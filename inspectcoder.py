import os
import sys
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['REQUESTS_CA_BUNDLE'] = ''
import re
from transformers import AutoTokenizer
from config.paths import config
from inspectware import PDBTerminal, InspectWare
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import json
import textwrap
from utils import extract_method_source
from agent_prompts import inspector_prompt_user, inspector_prompt_sys, reason_step_str, coder_prompt_sys, coder_prompt_user, coder_repair_history
sys.path.append(config.get('external_dependencies', 'bigcodebench'))
from bigcodebench.evaluate import untrusted_check
sys.path.append(config.get('external_dependencies', 'livecodebench'))
from lcb_runner.evaluation import codegen_metrics

def truncate_observation(obs: str, max_lines: int, max_length: int) -> str:
    lines = obs.splitlines()
    if len(lines) > max_lines:
        back_lines = max_lines // 2
        front_lines = max_lines - back_lines
        first_part = lines[:front_lines]
        last_part = lines[-back_lines:] if back_lines > 0 else []
        omitted_lines = len(lines) - max_lines
        lines = first_part + [f'...omit {omitted_lines} lines...'] + last_part
    for (i, line) in enumerate(lines):
        if len(line) > max_length:
            front_s_num = max_length // 2
            back_s_num = max_length - front_s_num
            lines[i] = line[:front_s_num] + '...' + line[-back_s_num:]
    return '\n'.join(lines)

class InspectCoder:

    def __init__(self, dataset: str, problem: dict, llm: 'BaseGenModel', logger: 'logging.Logger', output_dir: str, context_len: int=None, temperature: float=0.8, top_p: float=0.95, max_steps: int=10, max_solution_version: int=5, ablation: str='') -> None:
        self.ablation = ablation
        self.agent_prompt_sys = inspector_prompt_sys
        self.dataset = dataset
        self.task_id = problem['task_id']
        self.coding_task = problem['instruct_prompt']
        self.test = problem['test']
        self.private_test = problem['private_test'] if 'private_test' in problem else None
        self.entry_point = problem['entry_point']
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_solution_version = max_solution_version
        self.max_steps = max_steps
        self.step_n = 0
        self.thought = ''
        self.actions = []
        self.observation = ''
        self.agent_trace_snapshot = 'start'
        self.agent_trace = []
        self.agent_trace_file = str(self.output_dir / Path(f'agent_trace.csv'))
        self.llm_log = str(self.output_dir / Path(f'llm_log.csv'))
        self.solution_version = -1
        self.repair_plan = ''
        self.code_solution = problem['buggy_solution']
        self.code_solution = InspectWare._pdb_code_file_safeguard(self.code_solution)
        self.code_solution_with_lineno = None
        self.code_status = problem['buggy_status']
        self.private_code_status = None
        self.buggy_test_method_exec_results = eval(problem['buggy_details']) if type(problem['buggy_details']) == str else problem['buggy_details']
        self.focused_test_name = None
        self.focused_test_source = None
        self.focused_test_source_with_lineno = None
        self.focused_test_result = None
        self.pass_rate = None
        self.solution_trace = []
        self.solution_trace_file = str(self.output_dir / Path(f'solution_trace.csv'))
        self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version + 1}_s{self.step_n}_o{int(self.code_status == 'pass')}.py"))
        self.update_solution_logs(self.code_solution, self.code_status, self.buggy_test_method_exec_results)
        self.update_solution_code_file()
        self.llm = llm
        self.context_len = context_len if context_len else llm.context_window
        self.temperature = temperature
        self.top_p = top_p
        self.enc = token_enc
        self.input_token = 0
        self.output_token = 0
        self.model_call = 0

        self.terminal = PDBTerminal(self.solution_code_file)
        self.pdb_history_file = str(self.output_dir / Path(f'pdb_history.csv'))
        self.inspectWare = InspectWare(self.terminal, self.ablation, self.pdb_history_file)
        self.inspectWare.update_breakpoints_state()
        self.inspectWare.update_stack_frames_state()
        self.logger.info(f'Initiated task {self.task_id} debugger. max steps: {self.max_steps}, code status: {self.code_status}, focused test: {self.focused_test_name}')

    def _extract_first_unpass_test(self, status, buggy_details: dict[str, str], all_test: 'str|list[dict[str,str]]', dedent_test_method_source=False):
        if self.dataset == 'bigcodebench':
            if status == 'sys_error' or status == 'timeout' or 'ALL' in buggy_details:
                error_message = buggy_details.get('ALL', 'Unknown system error')
                focused_test_result = {'stat': 'sys_error', 'exception_type': error_message, 'stdout_logs': '', 'traceback_frame': [], 'traceback_str': error_message}
                pass_rate = f"0/{self.pass_rate.split('/')[1]}"
                return (self.focused_test_name, self.focused_test_source, focused_test_result, pass_rate)
            else:
                (focused_test_name, focused_test_result) = list(buggy_details.items())[0]
                focused_test_source = extract_method_source(all_test, focused_test_name, dedentation=dedent_test_method_source)
                all_test_methods = re.findall('def\\s+(test_\\w+)\\s*\\(', all_test)
                pass_count = len(all_test_methods) - len(buggy_details)
                pass_rate = f'{pass_count}/{len(all_test_methods)}'
                return (focused_test_name, focused_test_source, focused_test_result, pass_rate)
        elif self.dataset == 'livecodebench':
            (focused_test_name, focused_test_result) = list(buggy_details.items())[0]
            focused_test_name = int(focused_test_name)
            if focused_test_name == -1:
                focused_test_name = 0
            focused_test = all_test[focused_test_name]
            pass_rate = f'{focused_test_name}/{len(all_test)}'
            if focused_test['testtype'] == 'functional':
                inputs = ', '.join([repr(json.loads(arg)) for arg in focused_test['input'].split('\n')])
                outputs = repr(json.loads(focused_test['output']))
                focused_test_source = f"assert Solution().{self.entry_point}({inputs}) == {outputs}, 'Wrong Answer'"
            elif focused_test['testtype'] == 'stdin':
                focused_test_source = textwrap.dedent(f"""\n\n                    inputs = {repr(focused_test['input'])}\n                    \n                    from unittest.mock import patch, mock_open\n                    @patch("builtins.open", mock_open(read_data=inputs))\n                    @patch("sys.stdin.readline", lambda *args, it=iter(inputs.split("\\n")): next(it))\n                    @patch("sys.stdin.readlines", lambda *args: inputs.split("\\n"))\n                    @patch("sys.stdin.read", lambda *args: inputs)\n                    def test_with_mock_input(*args):\n                        try:\n                            solution()\n                        except SystemExit as e:\n                            pass\n                    \n                    test_with_mock_input() # expect the program to print {repr(focused_test['output'])}\n                """)
            else:
                raise ValueError(f"Unknown testtype {focused_test['testtype']}")
            return (focused_test_name, focused_test_source, focused_test_result, pass_rate)

    def _parse_action(self, action_str):
        action_str = re.sub('^```(?:python)?\\s*|\\s*```$', '', action_str.strip())
        pattern = '^(\\w+)\\(("""[\\s\\S]*?"""|\\\'\\\'\\\'[\\s\\S]*?\\\'\\\'\\\'|\\([^)]*\\)|[^)]*)\\)(?:\\s*#.*)?$'
        match = re.match(pattern, action_str.strip(), re.MULTILINE | re.DOTALL)
        if match:
            action_type = match.group(1)
            params_str = match.group(2).strip()
            if params_str.startswith('"""') or params_str.startswith("'''"):
                argument = params_str.strip('"""').strip("'''")
            else:
                try:
                    if params_str.isdigit():
                        argument = int(params_str)
                    elif params_str.startswith('"') and params_str.endswith('"') or (params_str.startswith("'") and params_str.endswith("'")):
                        argument = params_str[1:-1]
                    else:
                        argument = params_str
                except:
                    argument = params_str
            return (action_type, argument)
        return (None, None)

    def _parse_multiple_actions(self, actions_str):
        actions_str = re.sub('^```(?:python)?\\s*|\\s*```$', '', actions_str.strip())
        actions_str = re.sub('^\\s*#.*$', '', actions_str, flags=re.MULTILINE)
        actions_str = re.sub('(^|[^"\\\'\\\\])#.*$', '\\1', actions_str, flags=re.MULTILINE)
        pattern = '(\\w+)\\(("""[\\s\\S]*?"""|\\\'\\\'\\\'[\\s\\S]*?\\\'\\\'\\\'|"[^"]*"|\\\'[^\\\']*\\\'|\\([^)]*\\)|[^)]*)\\)(?:\\s*#.*)?'
        matches = re.finditer(pattern, actions_str, re.MULTILINE | re.DOTALL)
        actions = []
        for match in matches:
            action_type = match.group(1)
            params_str = match.group(2).strip()
            if params_str.startswith('"""') or params_str.startswith("'''"):
                argument = params_str[3:-3] if params_str.endswith('"""') or params_str.endswith("'''") else params_str[3:]
            elif params_str.startswith('"') and params_str.endswith('"'):
                argument = params_str[1:-1]
            elif params_str.startswith("'") and params_str.endswith("'"):
                argument = params_str[1:-1]
            else:
                try:
                    if params_str.isdigit():
                        argument = int(params_str)
                    else:
                        argument = params_str
                except:
                    argument = params_str
            actions.append((action_type, argument))
        return actions

    def update_solution_code_file(self):
        if self.dataset == 'bigcodebench':
            full_code = build_bcb_code_file(self.code_status, self.code_solution, self.entry_point, self.test, self.focused_test_name)
        elif self.dataset == 'livecodebench':
            full_code = build_lcb_code_file(self.code_status, self.code_solution, self.entry_point, self.test, self.focused_test_name)
        else:
            raise ValueError(f'unsupported dataset {self.dataset}')
        self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version}_s{self.step_n}_o{int(self.code_status == 'pass')}.py"))
        open(self.solution_code_file, 'w').write(full_code)
        if self.code_status == 'pass':
            return
        if self.dataset == 'bigcodebench':
            start_line_no_in_test = -1
            for (i, line) in enumerate(self.test.splitlines()):
                if f'def {self.focused_test_name}(self' in line:
                    start_line_no_in_test = i
                    break
            if start_line_no_in_test == -1:
                raise ValueError(f'focused test method {self.focused_test_name} not found in test code')
            focused_test_linenum_offset = len(self.code_solution.split('\n')) + start_line_no_in_test
            self.code_solution_with_lineno = self._add_line_numbers(self.code_solution)
            self.focused_test_source_with_lineno = self._add_line_numbers(self.focused_test_source, offset=focused_test_linenum_offset)
            self.traceback_str_sanitized = self.focused_test_result['traceback_str']
        elif self.dataset == 'livecodebench':
            codelines = full_code.splitlines(keepends=True)
            code_start_idx = 0
            for (i, line) in enumerate(codelines):
                if 'class Solution:' in line or 'def solution():' in line:
                    code_start_idx = i + 1
                    break
            test_start_idx = code_start_idx
            test_first_line = self.focused_test_source.strip().split('\n')[0].replace('wrapped_function(', 'solution(')
            for (i, line) in enumerate(codelines[code_start_idx - 1:], start=test_start_idx):
                if test_first_line in line:
                    test_start_idx = i
                    break
            solution_code = ''.join(codelines[code_start_idx - 1:test_start_idx]).strip()
            self.code_solution_with_lineno = self._add_line_numbers(solution_code, offset=code_start_idx - 1).replace('wrapped_function(', 'solution(')
            self.focused_test_source_with_lineno = self._add_line_numbers(self.focused_test_source.strip(), offset=test_start_idx - 1)
            self.traceback_str_sanitized = self._sanitize_traceback(self.focused_test_result['traceback_str'])

    def update_agent_stepwise_logs(self):
        current_step = {'task_id': self.task_id, 'step_n': self.step_n, 'solution_version': self.solution_version, 'thought': self.thought, 'action': self.actions, 'observation': self.observation, 'stack_frames': self.inspectWare.current_stack_frames_str.replace(self.solution_code_file, '__test__.py'), 'breakpoints': self.inspectWare.current_breakpoints_str.replace(self.solution_code_file, '__test__.py')}
        pd.DataFrame([current_step]).to_csv(self.agent_trace_file, mode='a', header=not os.path.isfile(self.agent_trace_file), index=False)
        self.agent_trace.append(current_step)
        self.inspectWare.record_stepwise_pdb_history(self.task_id, self.step_n, self.solution_version)
        if self.actions[0][0] == 'propose_repair':
            self.agent_trace = []
            self.actions = []
            self.thought = ''
            self.observation = ''
            self.terminal = PDBTerminal(self.solution_code_file)
            self.inspectWare = InspectWare(self.terminal, self.ablation, self.pdb_history_file)
            self.inspectWare.update_breakpoints_state()
            self.inspectWare.update_stack_frames_state()
            self.logger.info(f'Agent is ready for debugging a new solution.')

    def run(self) -> None:
        while True:
            isfinished = self.code_status == 'pass'
            ishalted = self.is_halted()
            if not isfinished and (not self.is_halted()):
                self.step_n += 1
                self.step()
                self.update_agent_stepwise_logs()
            else:
                self.logger.info(f'Agent exited. is_halted: {ishalted}, is_finished: {isfinished}')
                break
        self.logger.info(f'final code status: {self.code_status}, propose solution num: {self.solution_version}, model call: {self.model_call}, input token: {self.input_token}, output token: {self.output_token}')
        if isfinished and self.private_test is not None:
            (stat, details) = self.execute_code(self.code_solution, self.entry_point, self.private_test)
            self.private_code_status = stat
            self.logger.info(f'private code status: {stat}, results:\n{details}')

    def step(self) -> None:
        (self.thought, actions) = self._think_n_act()
        if actions == []:
            self.logger.error('Agent returns empty actions')
            self.observation = "System Hint: You must provide actions in '### ACTION\n```\n...\n```' format."
            return
        self.actions = []
        self.observation = []
        for (i, action) in enumerate(actions, start=1):
            (action_type, argument) = action
            if action_type == 'propose_repair' and i > 1 and (self.max_steps - self.step_n > 2):
                break
            (err_flag, obs) = self._exec_n_observe(action_type, argument)
            if err_flag and i != len(actions):
                obs += f'\nSystem Hint: An error occurred to this tool call and subsequent tool calls were aborted'
            elif err_flag:
                obs += f'\nSystem Hint: An error occurred to this tool call'
            self.actions.append(action)
            self.observation.append(f'> {action_type}({repr(str(argument))})\n  {obs}')
            if err_flag or self.terminal.state == 'done':
                break
        if self.max_steps - self.step_n <= 2:
            self.observation.append(f'\nSystem Hint: **You only have {self.max_steps - self.step_n + 1} steps left to propose a repair plan**')
        self.observation = '\n'.join(self.observation).replace(self.solution_code_file, '__test__.py')

    def _think_n_act(self) -> tuple[str, list[tuple[str, str]]]:
        thought_n_actions = self.prompt_agent(agent='inspector')
        self.logger.info(f'thought_n_actions {self.step_n}: {thought_n_actions}')
        (thought_str, actions_str) = thought_n_actions.split(f'### ACTION', maxsplit=1)
        thought = thought_str.split('### THOUGHT', maxsplit=1)[-1].strip()
        self.logger.info(f'thought {self.step_n}:\n{thought}')
        actions = self._parse_multiple_actions(actions_str)
        self.logger.info(f'actions {self.step_n}:\n{actions}')
        return (thought, actions)

    def _exec_n_observe(self, action_type, argument: dict) -> str:
        if action_type == 'propose_repair':
            err_flag = 0
            obs = self.propose_repair(argument)
        else:
            (err_flag, obs) = self.inspectWare.execute_action(action_type, argument)
        arg_repr = str(argument)
        truncated_arg = arg_repr[:40] + "...'" if len(arg_repr) > 40 else arg_repr
        self.agent_trace_snapshot += f'->{action_type}({truncated_arg})'
        if err_flag:
            self.logger.error(self.agent_trace_snapshot + '\nobs: ' + obs)
        else:
            self.logger.info(self.agent_trace_snapshot + '\nobs: ' + obs)
        return (err_flag, obs)

    def prompt_agent(self, agent='inspector') -> str:
        stop = ['\n### OBSERVATION', '\n### Observation', '\n### observation', '\nObservation:', 'END OF RESPONSE']
        try:
            if agent == 'inspector':
                (sys_prompt, user_prompt) = self._build_inspector_prompt()
                user_prompt = user_prompt
            elif agent == 'coder':
                (sys_prompt, user_prompt) = self._build_coder_prompt()
            (res, _) = self.llm.generate(prompt=user_prompt, sys_prompt=sys_prompt, n=1, temperature=self.temperature, top_p=self.top_p, stop=stop)
            result = res[0]
            self.input_token += len(self.enc.encode(user_prompt))
            if sys_prompt:
                self.input_token += len(self.enc.encode(sys_prompt))
            self.output_token += len(self.enc.encode(result))
            self.model_call += 1
        except Exception as e:
            self.logger.error(f'Error when generating text: {repr(e)}')
            sys_prompt = 'err'
            user_prompt = 'err'
            result = f'{repr(e)}.'
        return result

    def is_finished(self) -> bool:
        return self.code_status == 'pass'

    def is_halted(self) -> bool:
        if self.code_status == 'pass':
            return False
        next_reason_prompts = self._build_inspector_prompt(truncate=True)
        return self.step_n > self.max_steps or self.solution_version > self.max_solution_version or len(self.enc.encode(next_reason_prompts[0] + next_reason_prompts[1])) > self.context_len

    def safeguard_check(self, code: str) -> tuple[bool, str]:
        import re
        patterns = ['\\binput\\(\\s*\\)', '\\bread\\(\\s*\\)', '\\bsys\\.stdin\\.read\\(\\s*\\)', '\\bopen\\([^)]*\\)\\.read\\(\\s*\\)']
        for pattern in patterns:
            if re.search(pattern, code):
                return (True, 'PDB Error: input() and read() operations are not allowed in run_debug_code_before_paused_line. Inputs are already mocked into the program in test_with_mock_input, you can directly interact and inspect the program execution with the given inputs.')
        return (False, '')

    def propose_repair(self, repair_plan: str):
        self.repair_plan = repair_plan
        code = self.prompt_agent(agent='coder')
        from utils import extract_code_block
        code = extract_code_block(code)
        for import_stmt in self._extract_import_statements(self.solution_trace[-1]['code_solution']):
            if import_stmt not in code:
                code = import_stmt + '\n' + code
        self.logger.info(f'Proposed repair plan: {repair_plan}')
        self.logger.info(f'Generated new code version: {code}')
        (status, buggy_test_method_exec_details) = self.execute_code(code, self.entry_point, self.test)
        self.logger.info(f'New Code Solution Execution results: {status}\n{buggy_test_method_exec_details}')
        self.update_solution_logs(code, status, buggy_test_method_exec_details)
        self.update_solution_code_file()
        return f'<code>\n{code}\n</code>\n<status>{status}</status>\n<details>\n{buggy_test_method_exec_details}\n</details>\n<code_with_lineno>\n{self.code_solution_with_lineno}\n</code_with_lineno>\n<focused_test_with_lineno>\n{self.focused_test_source_with_lineno}\n</focused_test_with_lineno>\n<traceback_sanitized>\n{self.traceback_str_sanitized}\n</traceback_sanitized>'

    def execute_code(self, code: str, entry_point: str | None, test: str | list[dict[str, str]]) -> tuple:
        if self.dataset == 'bigcodebench':
            with ProcessPoolExecutor(max_workers=1) as executor:
                kwargs = {'code': code, 'test_code': self.test, 'entry_point': self.entry_point, 'max_as_limit': 30 * 1024, 'max_data_limit': 30 * 1024, 'max_stack_limit': 10, 'min_time_limit': 0.1, 'gt_time_limit': 2.0}
                future = executor.submit(untrusted_check, **kwargs)
                (stat, details) = future.result()
                return (stat, details)
        elif self.dataset == 'livecodebench':
            test_info = {'input_output': json.dumps({'inputs': [e['input'] for e in test], 'outputs': [e['output'] for e in test], 'fn_name': entry_point})}
            (pass_rates, test_results, test_metadata) = codegen_metrics([test_info], [[code]], num_process_evaluate=1, timeout=14)
            test_metadata = json.loads(test_metadata[0][0])
            if 'error_code' not in test_metadata:
                stat = 'pass'
            else:
                stat = {-2: 'fail', -3: 'timeout', -4: 'error', -5: 'sys_error'}.get(int(test_metadata.get('error_code')), 'sys_error')
            if stat == 'pass':
                details = {}
            else:
                details = {test_metadata['testcase_idx']: {'stat': stat, 'exception_type': test_metadata['error'], 'traceback_str': test_metadata['traceback'] if stat != 'fail' else f"Wrong Answer: for input {repr(test_metadata['inputs'])}, expected {repr(test_metadata['expected'])}, but received {repr(test_metadata['output'])}"}}
            return (stat, details)
        else:
            raise ValueError(f'Unknown dataset type: {self.dataset}')

    def update_solution_logs(self, code: str, status: str, details: dict) -> str:
        self.solution_version += 1
        self.code_solution = InspectWare._pdb_code_file_safeguard(code)
        self.code_status = status
        self.buggy_test_method_exec_results = details
        self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version}_s{self.step_n}_o{int(self.code_status == 'pass')}.py"))
        if status == 'pass':
            self.focused_test_name = None
            self.focused_test_source = None
            self.focused_test_result = None
            self.pass_rate = f"{self.pass_rate.split('/')[1]}/{self.pass_rate.split('/')[1]}"
        else:
            (focused_test_name, self.focused_test_source, self.focused_test_result, pass_rate) = self._extract_first_unpass_test(status, details, self.test)
            if focused_test_name != self.focused_test_name:
                self.logger.info(f'Focused test method changed from {self.focused_test_name} to {focused_test_name}. Pass Rate changed from {self.pass_rate} to {pass_rate}')
            self.focused_test_name = focused_test_name
            self.pass_rate = pass_rate
        current_solution = {'coding_task': self.coding_task, 'step_n': self.step_n, 'repair_plan': self.repair_plan, 'solution_version': self.solution_version, 'code_solution': self.code_solution, 'code_status': self.code_status, 'focused_test_name': self.focused_test_name, 'focused_test_source': self.focused_test_source, 'focused_test_result': self.focused_test_result, 'pass_rate': self.pass_rate, 'solution_code_file': self.solution_code_file}
        pd.DataFrame([current_solution]).to_csv(self.solution_trace_file, mode='a', header=not os.path.isfile(self.solution_trace_file), index=False)
        self.solution_trace.append(current_solution)

    def _extract_import_statements(self, code: str) -> list:
        import_statements = []
        for line in code.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                import_statements.append(line.strip())
        return import_statements

    def _build_inspector_prompt(self, truncate=True) -> str:
        sys_prompt = self.agent_prompt_sys
        if truncate and len(self.agent_trace) > 0:
            current_step_idx = 0
            last_step_idx = len(self.agent_trace) - 1
            while current_step_idx < last_step_idx:
                temp_trajectory_str = ''
                for (i, step) in enumerate(self.agent_trace):
                    actions_str = '\n'
                    for (action_type, action_argument) in step['action']:
                        actions_str = actions_str + action_type + '(' + repr(action_argument) + ')\n'
                    temp_trajectory_str += reason_step_str.format(step_num=step['step_n'], thought=step['thought'], actions=actions_str, observation=truncate_observation(step['observation'], max_lines=50, max_length=500))
                temp_prompt = inspector_prompt_user.format(coding_task=self.coding_task, buggy_code=self.code_solution_with_lineno, test_method=self.focused_test_source_with_lineno, execution_result=self.traceback_str_sanitized, stack_frames=self.inspectWare.current_stack_frames_str.replace(self.solution_code_file, '__test__.py'), breakpoints=self.inspectWare.current_breakpoints_str.replace(self.solution_code_file, '__test__.py'), reason_trajectory=temp_trajectory_str)
                if len(self.enc.encode(sys_prompt + temp_prompt)) <= self.context_len:
                    break
                current_step = self.agent_trace[current_step_idx]
                if not (isinstance(current_step['thought'], str) and current_step['thought'] == '<truncated>'):
                    self.agent_trace[current_step_idx]['thought'] = '<truncated>'
                    continue
                if not (isinstance(current_step['observation'], str) and current_step['observation'] == '<truncated>'):
                    self.agent_trace[current_step_idx]['observation'] = '<truncated>'
                    continue
                current_step_idx += 1
                if current_step_idx >= last_step_idx and len(self.enc.encode(sys_prompt + temp_prompt)) > self.context_len:
                    if len(self.agent_trace) > 2:
                        self.agent_trace.pop(0)
                        last_step_idx = len(self.agent_trace) - 1
                        current_step_idx = 0
                    else:
                        break
        reason_trajectory_str = ''
        for (i, step) in enumerate(self.agent_trace):
            actions_str = '\n'
            for (action_type, action_argument) in step['action']:
                actions_str = actions_str + action_type + '(' + repr(action_argument) + ')\n'
            reason_trajectory_str += reason_step_str.format(step_num=step['step_n'], thought=step['thought'], actions=actions_str, observation=truncate_observation(step['observation'], max_lines=50, max_length=500) if not step['observation'] == '<truncated>' else '<truncated>')
        user_prompt = inspector_prompt_user.format(coding_task=self.coding_task, buggy_code=self.code_solution_with_lineno, test_method=self.focused_test_source_with_lineno, execution_result=self.traceback_str_sanitized, stack_frames=self.inspectWare.current_stack_frames_str.replace(self.solution_code_file, '__test__.py'), breakpoints=self.inspectWare.current_breakpoints_str.replace(self.solution_code_file, '__test__.py'), reason_trajectory=reason_trajectory_str)
        if self.dataset == 'livecodebench' and 'def solution():' in self.code_solution_with_lineno:
            user_prompt.replace('which contains the buggy code and test method.', 'which contains the buggy code and test method. IO-based Input data is pre-mocked in test code; No more manual input needed during debugging')
        return (sys_prompt, user_prompt)

    def _sanitize_traceback(self, traceback_text):
        if traceback_text == '' or traceback_text == None:
            return traceback_text
        traceback_text = traceback_text.replace('_inner_call_method', 'test_with_mock_input').replace('wrapped_function', 'solution')
        transformed = traceback_text.replace('File "<string>"', 'File "__test__.py"')
        lines = transformed.split('\n')
        test_file_line_index = -1
        for (i, line) in enumerate(lines):
            if 'File "__test__.py"' in line:
                test_file_line_index = i
                break
        if test_file_line_index != -1:
            transformed_lines = lines[test_file_line_index:]
            transformed_lines.insert(0, 'Traceback (most recent call last):')
            transformed = '\n'.join(transformed_lines)
        return transformed

    def _build_coder_prompt(self) -> str:
        sys_prompt = coder_prompt_sys
        history_code_version_li = [coder_repair_history.format(version=row['solution_version'], buggy_code=row['code_solution'], stack_trace_and_error=row['focused_test_result']['traceback_str'], repair_plan=self.solution_trace[i + 1]['repair_plan'] if i + 1 < len(self.solution_trace) else self.repair_plan) for (i, row) in enumerate(self.solution_trace)]
        user_prompt = coder_prompt_user.format(coding_task=self.coding_task, history_code_version='\n\n'.join(history_code_version_li))
        return (sys_prompt, user_prompt)

    def _add_line_numbers(self, code_string, offset=0):
        lines = code_string.split('\n')
        max_line_num_width = len(str(len(lines)))
        numbered_lines = []
        for (i, line) in enumerate(lines, offset + 1):
            if line:
                line_num = str(i).rjust(max_line_num_width)
                numbered_line = f'{line_num}| {line}'
                numbered_lines.append(numbered_line)
        return '\n'.join(numbered_lines)