import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tiktoken
import re

from transformers import AutoTokenizer

from inspectware import InspectWare, InspectWareLLM
from pathlib import Path
import logging
import types
import unittest
import sys
from models import LLM
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


import logging
import re
import json
from pathlib import Path
import subprocess
from typing import List, Dict, Any, Tuple
import textwrap

from utils import setup_log, setup_separate_log, read_df, extract_method_source, get_function_names, extract_imports
from utils import transform_lcb_ds, build_bcb_code_file, build_lcb_code_file
from agent_prompts import inspector_prompt_sys, inspector_prompt_user, reason_step_str, coder_prompt_sys, coder_prompt_user, coder_repair_history

from bigcodebench.evaluate import untrusted_check
from lcb_runner.evaluation import codegen_metrics

import signal
from contextlib import contextmanager

@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")
    
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def truncate_observation(obs:str, max_lines:int, max_length:int) -> str:
    lines = obs.splitlines()
    if len(lines) > max_lines:
        back_lines = max_lines // 2
        front_lines = max_lines - back_lines
        first_part = lines[:front_lines]
        last_part = lines[-back_lines:] if back_lines > 0 else []
        omitted_lines = len(lines) - max_lines
        lines = first_part + [f'...omit {omitted_lines} lines...'] + last_part
    for i, line in enumerate(lines):
        if len(line) > max_length:
            front_s_num = max_length // 2
            back_s_num = max_length - front_s_num
            lines[i] = line[:front_s_num] + '...' + line[-back_s_num:]
    return '\n'.join(lines)



def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = token_enc) -> str:
    segs = split_text_by_prefixes(scratchpad)
    
    current_group = 1
    
    while len(tokenizer.encode('\n'.join(segs))) > n_tokens:
        if current_group + 3 >= len(segs):
            break
            
        if '<truncated>' in segs[current_group + 3]:
            current_group += 4
            continue
            
        ap_idx = current_group
        th_idx = current_group + 1
        ac_idx = current_group + 2
        ob_idx = current_group + 3

        actionpath_parts = segs[ap_idx].split()
        segs[ap_idx] = f"{actionpath_parts[0]} {actionpath_parts[1]} <truncated>"
        
        action_parts = segs[ac_idx].split('[[', 1)
        if len(action_parts) > 1:
            segs[ac_idx] = action_parts[0] + '[[<truncated>]]'
            
        observation_parts = segs[ob_idx].split()
        segs[ob_idx] = f"{observation_parts[0]} {observation_parts[1]} <truncated>"
        
        current_group += 4
    
    return '\n'.join(segs)


class PatchCoder:
    """Handles code generation, execution, and solution management."""
    
    def __init__(self, inspector: 'ProgramInspector'):
        self.inspector = inspector
        
    def extract_first_unpass_test(self, status, buggy_details: dict[str, str], all_test: 'str|list[dict[str,str]]', dedent_test_method_source=False):
        """
        Return:
        - focused_test_name:str|int testmethod name for bigcodebench, or testcase index for livecodebench
        - focused_test_source:str
        - focused_test_result:dict[str,str] keys: 'traceback_str'
        - pass_rate:str "num/num"
        """
        if self.inspector.dataset == "bigcodebench":
            if status == 'sys_error' or status == 'timeout' or "ALL" in buggy_details:
                error_message = buggy_details.get('ALL', 'Unknown system error')
                focused_test_result = {'stat': 'sys_error', 'exception_type': error_message, 'stdout_logs': '', 'traceback_frame': [], 'traceback_str': error_message}
                pass_rate = f'0/{self.inspector.pass_rate.split("/")[1]}'
                return self.inspector.focused_test_name, self.inspector.focused_test_source, focused_test_result, pass_rate
            else:  # status == 'fail' or 'error'
                focused_test_name, focused_test_result = list(buggy_details.items())[0]
                focused_test_source = extract_method_source(all_test, focused_test_name, dedentation=dedent_test_method_source)
                all_test_methods = re.findall(r'def\s+(test_\w+)\s*\(', all_test)
                pass_count = len(all_test_methods) - len(buggy_details)
                pass_rate = f"{pass_count}/{len(all_test_methods)}"
                return focused_test_name, focused_test_source, focused_test_result, pass_rate
        elif self.inspector.dataset == "livecodebench":
            focused_test_name, focused_test_result = list(buggy_details.items())[0]
            focused_test_name = int(focused_test_name)
            if focused_test_name == -1:
                focused_test_name = 0
            focused_test = all_test[focused_test_name]
            pass_rate = f"{focused_test_name}/{len(all_test)}"

            if focused_test['testtype'] == 'functional':  # call-based
                inputs = ", ".join([repr(json.loads(arg)) for arg in focused_test['input'].split('\n')])
                outputs = repr(json.loads(focused_test['output']))
                focused_test_source = f"assert Solution().{self.inspector.entry_point}({inputs}) == {outputs}, 'Wrong Answer'"
            elif focused_test['testtype'] == 'stdin':  # io-based
                focused_test_source = textwrap.dedent(rf'''

                    inputs = {repr(focused_test['input'])}
                    
                    from unittest.mock import patch, mock_open
                    @patch("builtins.open", mock_open(read_data=inputs))
                    @patch("sys.stdin.readline", lambda *args: next(iter(inputs.split("\n"))))
                    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
                    @patch("sys.stdin.read", lambda *args: inputs)
                    def test_with_mock_input(*args):
                        try:
                            solution()
                        except SystemExit as e:
                            pass
                    
                    test_with_mock_input() # expect the program to print {repr(focused_test['output'])}
                ''')
            else:
                raise ValueError(f"Unknown testtype {focused_test['testtype']}")
            return focused_test_name, focused_test_source, focused_test_result, pass_rate

    def execute_code(self, code: str, entry_point: str | None, test: str | list[dict[str, str]]) -> tuple:
        """
        Execute the code and return overall eval `status` and per-unpass-test-method's eval `details`
        Return:
            status:str=pass|fail|error|sys_error|timeout
            details:dict[str, dict]={
                "test_method_name": exec_result_dict={
                        #"stat": "fail|error",
                        #"exception_type": "Exception",
                        #"traceback_frame": [{}, {}, ...] # type: List[Dict], keys: filename, lineno, funcname, codeline
                        #"stdout_logs": "...",
                        "traceback_str": "",
                    }
            }
        """
        if self.inspector.dataset == "bigcodebench":
            with ProcessPoolExecutor(max_workers=1) as executor:
                kwargs = {
                    'code': code,
                    'test_code': self.inspector.test,
                    'entry_point': self.inspector.entry_point,
                    'max_as_limit': 30 * 1024,
                    'max_data_limit': 30 * 1024,
                    'max_stack_limit': 10,
                    'min_time_limit': 0.1,
                    'gt_time_limit': 2.0,
                }
                future = executor.submit(untrusted_check, **kwargs)
                stat, details = future.result()
                return stat, details

        elif self.inspector.dataset == "livecodebench":
            test_info = {
                "input_output": json.dumps(
                    {
                        "inputs": [
                            e['input'] for e in test
                        ],
                        "outputs": [
                            e['output'] for e in test
                        ],
                        "fn_name": entry_point
                    }
                )
            }
            pass_rates, test_results, test_metadata = codegen_metrics(
                [test_info],
                [[code]],
                num_process_evaluate=1,
                timeout=14,
            )
            test_metadata = json.loads(test_metadata[0][0])
            if 'error_code' not in test_metadata:
                stat = "pass"
            else:
                stat = {
                    -2: "fail",
                    -3: "timeout",
                    -4: "error",
                    -5: "sys_error"
                }.get(int(test_metadata.get('error_code')), "sys_error")
            if stat == "pass":
                details = {}
            else:
                details = {
                    test_metadata['testcase_idx']: {
                        "stat": stat,  # fail|timeout|error|sys_error
                        "exception_type": test_metadata['error'],
                        "traceback_str": test_metadata['traceback'] if stat != "fail" else f"Wrong Answer: for input {repr(test_metadata['inputs'])}, expected {repr(test_metadata['expected'])}, but received {repr(test_metadata['output'])}"
                    }
                }
            return stat, details
        else:
            raise ValueError(f"Unknown dataset type: {self.inspector.dataset}")

    def update_solution_logs(self, code: str, status: str, details: dict) -> str:
        """Update agent state based on execution results and return observation"""
        self.inspector.solution_version += 1
        self.inspector.code_solution = self.inspector._pdb_code_file_safeguard(code)
        self.inspector.code_status = status
        self.inspector.buggy_test_method_exec_results = details
        self.inspector.solution_code_file = str(self.inspector.output_dir / Path(f"solution_v{self.inspector.solution_version}_s{self.inspector.step_n}_o{int(self.inspector.code_status=='pass')}.py"))
        
        # Handle successful execution
        if status == 'pass':
            self.inspector.focused_test_name = None
            self.inspector.focused_test_source = None
            self.inspector.focused_test_result = None
            self.inspector.pass_rate = f'{self.inspector.pass_rate.split("/")[1]}/{self.inspector.pass_rate.split("/")[1]}'
        else:  # status == 'sys_error' or 'timeout' or 'fail' or 'error'
            focused_test_name, self.inspector.focused_test_source, self.inspector.focused_test_result, pass_rate = self.extract_first_unpass_test(status, details, self.inspector.test)
            if focused_test_name != self.inspector.focused_test_name:
                self.inspector.logger.info(f"Focused test method changed from {self.inspector.focused_test_name} to {focused_test_name}. Pass Rate changed from {self.inspector.pass_rate} to {pass_rate}")
            self.inspector.focused_test_name = focused_test_name
            self.inspector.pass_rate = pass_rate
        
        current_solution = {
            "coding_task": self.inspector.coding_task,
            "step_n": self.inspector.step_n,
            "repair_plan": self.inspector.repair_plan,
            "solution_version": self.inspector.solution_version,
            "code_solution": self.inspector.code_solution,
            "code_status": self.inspector.code_status,
            "focused_test_name": self.inspector.focused_test_name,
            "focused_test_source": self.inspector.focused_test_source,
            "focused_test_result": self.inspector.focused_test_result,
            "pass_rate": self.inspector.pass_rate,
            "solution_code_file": self.inspector.solution_code_file,
        }
        pd.DataFrame([current_solution]).to_csv(
            self.inspector.solution_trace_file,
            mode='a',
            header=not os.path.isfile(self.inspector.solution_trace_file),  # 仅在文件不存在时写入表头
            index=False
        )
        self.inspector.solution_trace.append(current_solution)

    def generate_repair_code(self, repair_plan: str) -> str:
        """Generate repaired code using LLM based on repair plan."""
        self.inspector.repair_plan = repair_plan
        
        # Generate code using LLM
        code = self.inspector.prompt_agent(mode='code')
        from utils import extract_code_block
        code = extract_code_block(code)
        
        # Add missing import statements
        for import_stmt in self._extract_import_statements(self.inspector.solution_trace[-1]['code_solution']):
            if import_stmt not in code:
                code = import_stmt + '\n' + code
                
        return code
    
    def propose_repair(self, repair_plan: str) -> str:
        """
        Execute the repair action by generating fixed code and updating agent state based on execution results.
        
        Args:
            repair_plan (str): The plan for editing/fixing the code
        
        Returns:
            str: Generated code after repair
        """
        self.inspector.logger.info(f"Proposed repair plan: {repair_plan}")
        
        # Generate repaired code
        code = self.generate_repair_code(repair_plan)
        self.inspector.logger.info(f"Generated new code version: {code}")
        
        # Execute and evaluate the code
        status, buggy_test_method_exec_details = self.execute_code(code, self.inspector.entry_point, self.inspector.test)
        self.inspector.logger.info(f"New Code Solution Execution results: {status}\n{buggy_test_method_exec_details}")
        
        # Update solution logs and code file
        self.update_solution_logs(code, status, buggy_test_method_exec_details)
        self.inspector.update_solution_code_file()

        return code
    
    def _extract_import_statements(self, code: str) -> list:
        """
        Extract import statements from the code.
        
        Args:
            code (str): The input code string
        """
        import_statements = []
        for line in code.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                import_statements.append(line.strip())
        return import_statements


class ProgramInspector:
    def __init__(self,
                 dataset: str,
                 problem: dict,
                 llm: "BaseGenModel",
                 logger: "logging.Logger",
                 output_dir:str,
                 context_len: int = None,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 max_steps: int= 10,
                 max_solution_version: int= 5,
                 ablation: str = ""
                 ) -> None:
        # breakpoint()
        self.ablation = ablation
        # task info
        self.dataset = dataset # bigcodebench or livecodebench
        self.task_id = problem['task_id']
        self.coding_task = problem['instruct_prompt']
        self.test = problem['test']
        self.private_test = problem['private_test'] if 'private_test' in problem else None
        self.entry_point = problem['entry_point']

        # logs and outputs
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # agent info
        self.max_solution_version = max_solution_version
        self.max_steps = max_steps
        self.step_n = 0 # 从1:agent的第一个react step开始
        self.thought = ''
        self.action = ''
        self.observation = ''

        self.agent_trace_snapshot = "start"
        self.agent_trace = []
        self.agent_trace_file = str(self.output_dir / Path(f"agent_trace.csv"))

        # bug info
        self.solution_version = -1
        self.repair_plan = ''
        self.code_solution = problem['buggy_solution']
        self.code_solution = self._pdb_code_file_safeguard(self.code_solution)
        self.code_status = problem['buggy_status']
        self.private_code_status = None
        self.buggy_test_method_exec_results = eval(problem['buggy_details']) if type(problem['buggy_details'])==str else problem['buggy_details']
        self.focused_test_name = None
        self.focused_test_source = None
        self.focused_test_result = None
        self.pass_rate = None

        self.solution_trace = []
        self.solution_trace_file = str(self.output_dir / Path(f"solution_trace.csv"))
        self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version+1}_s{self.step_n}_o{int(self.code_status=='pass')}.py"))
        self.update_solution_logs(self.code_solution, self.code_status, self.buggy_test_method_exec_results)
        self.update_solution_code_file()


        # llm info
        self.llm = llm
        self.context_len = context_len if context_len else llm.context_window
        self.temperature = temperature
        self.top_p = top_p
        self.enc = token_enc
        self.input_token = 0
        self.output_token = 0
        self.model_call = 0

        # terminal tool
        self.llm_debugger = (self.ablation == "_wodebugger")
        if self.llm_debugger:
            self.debugger = InspectWareLLM(self.solution_code_file, self.llm)
        else:
            self.debugger = PDBTerminal(self.solution_code_file)
        self.debugger.start_debugging()
        self.active_breakpoints = []
        self.update_breakpoints_state()
        self.update_stack_frames_state()

        self.pdb_history_file = str(self.output_dir / Path(f"pdb_history.csv")) # 用于存储pdb terminal执行的command历史记录

        # Initialize PatchCoder for code generation and execution
        self.patch_coder = PatchCoder(self)
        
        # self.__reset_agent()
        self.logger.info(f"Initiated task {self.task_id} debugger. max steps: {self.max_steps}, code status: {self.code_status}, focused test: {self.focused_test_name}")
    @staticmethod
    def _pdb_code_file_safeguard(code):
        if "input = sys.stdin.read" in code:
            return code

        code = re.sub(r'\bbuiltins\.input\(\s*\)', r'sys.stdin.read()', code)
        
        code = re.sub(r'\binput\(\s*\)', r'sys.stdin.read()', code)

        return code


    
    def _parse_action(self, action_str):
        """
        Parse action string to extract action type and parameters.
        Handles both code block format and direct format.
        Supports multi-line parameters.
        """
        action_str = re.sub(r'^```(?:python)?\s*|\s*```$', '', action_str.strip())
        
        pattern = r'^(\w+)\(("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|\([^)]*\)|[^)]*)\)(?:\s*#.*)?$'
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
                    elif (params_str.startswith('"') and params_str.endswith('"')) or \
                        (params_str.startswith("'") and params_str.endswith("'")):
                        argument = params_str[1:-1]
                    else:
                        argument = params_str
                except:
                    argument = params_str
                    
            return action_type, argument
        
        return None, None
    def _parse_multiple_actions(self, actions_str):
        """
        Parse a string containing multiple actions to extract action types and parameters.
        Returns a list of (action_type, argument) tuples.
        Handles both code block format and direct format.
        Supports multi-line parameters.
        """
        actions_str = re.sub(r'^```(?:python)?\s*|\s*```$', '', actions_str.strip())
        actions_str = re.sub(r'^\s*#.*$', '', actions_str, flags=re.MULTILINE)
        actions_str = re.sub(r'(^|[^"\'\\])#.*$', r'\1', actions_str, flags=re.MULTILINE)
        
        pattern = r'(\w+)\(("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\'|\([^)]*\)|[^)]*)\)(?:\s*#.*)?'
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
        if self.dataset=="bigcodebench":
            full_code = build_bcb_code_file(self.code_status, self.code_solution, self.entry_point, self.test, self.focused_test_name)
        elif self.dataset=="livecodebench":
            full_code = build_lcb_code_file(self.code_status, self.code_solution, self.entry_point, self.test, self.focused_test_name)
        else:
            raise ValueError(f"unsupported dataset {self.dataset}")
        self.solution_code_file = str(self.output_dir / Path(f"solution_v{self.solution_version}_s{self.step_n}_o{int(self.code_status=='pass')}.py"))
        open(self.solution_code_file, 'w').write(full_code)


    def update_agent_stepwise_logs(self):
        current_step = {
            "task_id": self.task_id,
            # "coding_task": self.coding_task,
            "step_n": self.step_n,
            "solution_version": self.solution_version,
            "thought": self.thought,
            "action": self.action, # list of tuple(action_type, action_argument)
            "observation": self.observation,
            "state": self.debugger.state,
        }
        pd.DataFrame([current_step]).to_csv(
            self.agent_trace_file, 
            mode='a',
            header=not os.path.isfile(self.agent_trace_file),
            index=False
        )
        self.agent_trace.append(current_step)
        current_pdb_history = {
            "task_id": self.task_id,
            "step_n": self.step_n,
            "solution_version": self.solution_version,
            "pdb_history": self.debugger.history
        }
        pd.DataFrame([current_pdb_history]).to_csv(
            self.pdb_history_file, 
            mode='a',
            header=not os.path.isfile(self.pdb_history_file),
            index=False
        )
        self.debugger.history = []

        if self.action[0][0] == "propose_repair": 
            self.prepare_debug_for_new_solution()
            self.logger.info(f"Agent is ready for debugging a new solution.")

    def prepare_debug_for_new_solution(self):
        self.agent_trace = []
        self.action = ''
        self.thought = ''
        self.observation = ''

        if self.llm_debugger:
            self.debugger = InspectWareLLM(self.solution_code_file, self.llm)
        else:
            self.debugger = PDBTerminal(self.solution_code_file)
        
        self.debugger.start_debugging()
        self.update_breakpoints_state()
        self.update_stack_frames_state()

    def run(self) -> None:
        while True:
            isfinished = self.code_status == "pass"
            ishalted = self.is_halted()

            if not isfinished and not self.is_halted():
                self.step_n += 1
                self.step()
                self.update_agent_stepwise_logs()
            else:
                self.logger.info(f'Agent exited. is_halted: {ishalted}, is_finished: {isfinished}')
                break
        
        self.logger.info(f"final code status: {self.code_status}, propose solution num: {self.solution_version}, model call: {self.model_call}, input token: {self.input_token}, output token: {self.output_token}")
        if isfinished and self.private_test is not None:
            stat, details = self.execute_code(self.code_solution, self.entry_point, self.private_test)
            self.private_code_status = stat
            self.logger.info(f"private code status: {stat}, results:\n{details}")
            

    def step(self) -> None:
        # breakpoint()
        self.thought, self.action = self._think_n_act()
        obs = []
        for i, action in enumerate(self.action, start=1):
            action_type, argument = action
            try:
                # Execute with timeout
                if action_type=="interact_code":
                    with timeout_context(6):  # 6-second timeout
                        tmp_observation = self._exec_n_observe(action)
                else:
                    tmp_observation = self._exec_n_observe(action)
                
                # Process the observation
                if tmp_observation == '' and action_type == "interact_code":
                    tmp_observation = "(Successfully run. No output)"
                obs.append(f"> {action_type}({repr(argument)})\n  {tmp_observation}")
                if ("Traceback (most recent call last)" in tmp_observation or "PDB Error:" in tmp_observation)\
                    and i != len(self.action):
                    obs.append(f"  (An error occurs to this tool call and subsequent tool calls are aborted)") #  Analyze the pdb state and action format. Identify the root cause, then either fix and retry the failed action or develop a new action plan.
                    break
            except TimeoutError:
                # Handle timeout
                tmp_observation = "Execution timeout, please examine your action."
                if action_type=="interact_code" and ("input()" in argument or "read()" in argument):
                    tmp_observation += "Do Not use io operation (eg. input() or read()) as they block execution in pdb. The io input has been mocked into the program."
                obs.append(f"> {action_type}({repr(argument)})\n  {tmp_observation}")
                break
            except Exception as e:
                # Handle other exceptions
                if i != len(self.action):
                    obs.append(f'> {action_type}({repr(argument)})\n{repr(e)}\n  (An error occurs to this tool call and subsequent tool calls are aborted)') 
                else:
                    obs.append(f'> {action_type}({repr(argument)})\n{repr(e)}\n  (An error occurs to this tool call)') 
                break
        self.observation = '\n'.join(obs)
        if self.max_steps-self.step_n <= 2:
            self.observation += f"\n**You only have {self.max_steps-self.step_n+1} steps left to propose a repair plan**"


    def _think(self):
        self.scratchpad += f'\nThought {self.step_n}:'
        self.logger.info(f'Thought {self.step_n}:')
        thought = self.prompt_agent()
        self.scratchpad += ' ' + thought
        self.logger.info(thought)
        print(f'Thought {self.step_n}:' + self.scratchpad.split(f'\nThought {self.step_n}:')[-1])

    def _action(self):
        self.scratchpad += f'\nAction {self.step_n}:'
        self.logger.info(f'Action {self.step_n}:')
        action = self.prompt_agent()
        
        if action.startswith(f'\nAction {self.step_n}:'):
            action = action.split(f'\nAction {self.step_n}:', maxsplit=1)[-1]
        self.scratchpad += ' ' + action
        self.logger.info(action)
        action_type, argument = self._parse_action(action)
        print(f'Action {self.step_n}: ' + action)
        return action_type, argument
    def _think_n_act(self):
        thought_n_actions = self.prompt_agent()
        self.logger.info(f"thought_n_actions {self.step_n}: {thought_n_actions}")
        thought_str, actions_str = thought_n_actions.split(f'### ACTION', maxsplit=1)
        thought = thought_str.split('### THOUGHT', maxsplit=1)[-1].strip()
        self.logger.info(f"thought {self.step_n}:\n{thought}")
        actions = self._parse_multiple_actions(actions_str)
        self.logger.info(f"actions {self.step_n}:\n{actions}")
        return thought, actions
    def _exec_n_observe(self, action: tuple[str,str]) -> str:     
        action_type, argument = action
        if action_type == "set_breakpoint":
            obs = self.set_breakpoint(argument)
            self.agent_trace_snapshot = self.agent_trace_snapshot + f'->{action_type}({repr(argument)})'
            self.logger.info(self.agent_trace_snapshot + '\n' + obs)
        elif action_type == "remove_breakpoint":
            obs = self.set_breakpoint(argument, remove=True)
            self.agent_trace_snapshot = self.agent_trace_snapshot + f'->{action_type}({repr(argument)})'
            self.logger.info(self.agent_trace_snapshot + '\n' + obs)
        elif action_type == "control_execution":
            obs = self.control_execution(argument)
            self.agent_trace_snapshot = self.agent_trace_snapshot + f'->{action_type}({repr(argument)})'
            self.logger.info(self.agent_trace_snapshot + '\n' + obs)
        elif action_type == "interact_code":
            obs = self.interact_code(argument)
            if len(argument) > 40:
                argument = argument[:40] + '...'
            self.agent_trace_snapshot = self.agent_trace_snapshot + f'->{action_type}({repr(argument)})'
            self.logger.info(self.agent_trace_snapshot + '\n' + obs)
        elif action_type == "propose_repair":
            obs = self.propose_repair(argument)
            if len(argument) > 40:
                argument = argument[:40] + '...'
            self.agent_trace_snapshot = self.agent_trace_snapshot + f'->{action_type}({repr(argument)})'
            self.logger.info(self.agent_trace_snapshot + '\n' + obs)
        else:
            obs = 'Invalid Action. Valid Actions are set_breakpoint, control_execution, interact_code and propose_repair'
            self.logger.error(obs)
        return obs
    
    def prompt_agent(self, mode='reason') -> str:
        # stop = ['\n### THOUGHT', '\n### ACTION', '\n### OBSERVATION']
        stop = ['\n### OBSERVATION', '\n### Observation', '\n### observation', '\nObservation:']
        try:
            if mode == 'reason':
                sys_prompt, user_prompt = self._build_agent_prompt()
            elif mode == 'code':
                sys_prompt, user_prompt = self._build_coder_prompt()
            res = self.llm.generate(prompt=user_prompt, sys_prompt=sys_prompt, n=1, temperature=self.temperature, top_p=self.top_p, stop=stop)
            result = res[0]
            # update token count and model call count
            self.input_token += len(self.enc.encode(user_prompt))
            if sys_prompt:
                self.input_token += len(self.enc.encode(sys_prompt))
            self.output_token += len(self.enc.encode(result))
            self.model_call += 1
            # self.check_run_error(result)
        except Exception as e:
            self.logger.error(f"Error when generating text: {repr(e)}")
            result = f"{repr(e)}."
        return result

    def check_run_error(self, text):
        if text in ["No response"]:
            self.run_error = True
            
    def is_finished(self) -> bool:
        return self.code_status == 'pass'

    def is_halted(self) -> bool:
        if self.code_status == 'pass':
            return False
        next_reason_prompts = self._build_agent_prompt(truncate=True)
        return ((self.step_n > self.max_steps) or (self.solution_version > self.max_solution_version)
                or (len(self.enc.encode(next_reason_prompts[0]+next_reason_prompts[1])) > self.context_len)
                )

    def __reset_agent(self) -> None: # TODO
        self.step_n = 1
        self.scratchpad: str = ''


    def set_breakpoint(self, line: int, remove=False):
        if remove:
            obs = self.debugger.remove_breakpoint(file=None, line=line)
        else:
            if line in [int(bp['where'].split(':')[-1]) for bp in self.active_breakpoints]: # TODO: future extention for repo-level debugging
                obs = f"Breakpoint at line {line} already exists."
            else:
                obs = self.debugger.set_breakpoint(file=None, line=line)
                if "*** Blank or comment" in obs:
                    obs = f"PDB Error: The target line {line} is blank or a comment line, please set breakpoint at a valid line."
        self.update_breakpoints_state()
        return obs
    def update_breakpoints_state(self):
        """ 
        Active Breakpoints:
            (No breakpoints.)
            1 at file:line, hitted {hit} time.
            2 at file:line, hitted {hit} time.
        """
        self.active_breakpoints = [bp for bp in self.debugger.list_breakpoints() if bp['disp'] == 'keep']
        current_breakpoints_str = []
        for i, frame in enumerate(self.debugger.list_breakpoints(), start=1):
            breakpoint_str = f"b{frame['num']} {frame['where']}, hitted {frame['hit']} time."
            current_breakpoints_str.append(breakpoint_str)
        if current_breakpoints_str:
            current_breakpoints_str = '\n'.join(current_breakpoints_str)
        else:
            current_breakpoints_str = "No breakpoints."
        self.current_breakpoints_str = "Active Breakpoints:\n" + current_breakpoints_str
    def control_execution(self, step_type: str):
        if step_type == 'step over':
            result = self.debugger.next()
        # elif step_type == 'step in':
        #     result = self.debugger.step_in()
        # elif step_type == 'step out':
        #     result = self.debugger.step_out()
        elif step_type == 'continue':
            result = self.debugger.continue_()
        elif step_type == 'restart':
            result = self.debugger.restart()
        else:
            # return 'Invalid step_type. type should be either "step over"|"step in"|"step out"|"continue"|"restart"'
            return 'Invalid step_type. type should be either "step over"|"continue"|"restart"'
        exec_msg = result['post_output']
        line_num = result['curstack'].split('.py(')[-1].split(')')[0]
        file_name = result['curstack'].split('.py(')[0] + '.py'
        func_name = result['curstack'].split('.py(')[-1].split(')', maxsplit=1)[-1]
        code_line = result['curline']
        add_info = result['add_info']
        self.update_stack_frames_state()
        if self.debugger.state=="post_mortem" and add_info == "post mortem start":
            exec_msg = exec_msg.replace("Running 'cont' or 'step' will restart the program", "").replace("Uncaught exception. Entering post mortem debugging", "")
            return f"An error occurs during execution: {exec_msg}.\nPDB entering post mortem debugging mode before file {file_name}, line {line_num}| {code_line}\nIn this mode, you can only call (1)interact_code(your code here) to inspect state or (2)control_execution('restart') to exit post mortem mode and restart the program."
        elif self.debugger.state=="runtime_state" and add_info == "post mortem end":
            return f"{exec_msg}\nPost mortem mode finished. The program is now restarted and paused before file {file_name}: line {line_num}| {code_line}"
        elif self.debugger.state=="finished" and add_info == "program finished":
            return f"{exec_msg}\nProgram finished successfully. PDB automatically restarted the program, now paused before file {file_name}, line {line_num}| {code_line}"
        else:
            return f"{exec_msg}\nCurrent execution paused before file {file_name}, line {line_num}| {code_line}"
    def update_stack_frames_state(self):
        """ 
        如果action里有control_execution，更新当前stack trace信息, 格式如
            Current Stack Trace:
            (Somethings wrong, fail to execute 'where' in the PDB process.)
            [1] task_func in {file}:{lineno}| {code_line}
            [2] main in {file}:{lineno}| {code_line} (currently paused before this line)
        """
        current_stack_frames_str = []
        for i, frame in enumerate(self.debugger.where(), start=1):
            frame_str = f"[{i}] {frame['func']} at {frame['file']}:{frame['line']}| {frame['code']}"
            if frame['is_current']:
                frame_str += " (current execution paused before this line)"
            current_stack_frames_str.append(frame_str)
        if current_stack_frames_str:
            current_stack_frames_str = '\n'.join(current_stack_frames_str)
        else:
            current_stack_frames_str = "Somethings wrong, fail to execute 'where' in the PDB process."
        self.current_stack_frames_str = "Current Stack Trace:\n" + current_stack_frames_str
    def safeguard_check(self, code: str) -> tuple[bool, str]:
        """
        Check if code contains input() or read() operations that might block execution.
        
        Args:
            code (str): The code to check
            
        Returns:
            tuple[bool, str]: (has_blocking_operations, error_message)
        """
        import re
        
        # Patterns to detect various forms of blocking input operations
        patterns = [
            r'\binput\(\s*\)',  # input()
            r'\bread\(\s*\)',   # read()
            r'\bsys\.stdin\.read\(\s*\)',  # sys.stdin.read()
            r'\bopen\([^)]*\)\.read\(\s*\)'  # open(...).read()
        ]
        
        for pattern in patterns:
            if re.search(pattern, code):
                return True, "PDB Error: input() and read() operations are not allowed in interact_code. Inputs are already mocked into the program in test_with_mock_input, you can directly interact and inspect the program execution with the given inputs."
        
        return False, ""
        
    def interact_code(self, code: str):
        code = code.strip()
        
        # Check for blocking IO operations
        has_blocking_ops, error_message = self.safeguard_check(code)
        if has_blocking_ops:
            return error_message
            
        print_statement_modified_overall = False 

        if '\n' in code:
            result_buffer = []
            original_blocks = self._split_code_blocks(code)
            for block_to_process in original_blocks:
                current_block_code_to_inspect = block_to_process
                if self.ablation == "_woBI":
                    processed_code, current_block_modified = self._process_for_print(block_to_process)
                    if current_block_modified:
                        print_statement_modified_overall = True
                    current_block_code_to_inspect = processed_code
                
                result_item = self.debugger.interact_code(current_block_code_to_inspect)
                result_buffer.append(result_item)
                if "Traceback (most recent call last):" in result_item: 
                    break
            result = '\n'.join(result_buffer)
        else: 
            current_code_to_inspect = code 
            if self.ablation == "_woBI":
                processed_code, current_code_modified = self._process_for_print(code)
                if current_code_modified:
                    print_statement_modified_overall = True
                current_code_to_inspect = processed_code 
            result = self.debugger.interact_code(current_code_to_inspect)

        if print_statement_modified_overall:
            if result and not result.endswith('\n'): # Ensure note is on a new line
                result += "\n"
            result += "Note: you cannot use print statement in this tool"
            
        return result
    def _process_for_print(self, code_segment: str) -> tuple[str, bool]:
        lines = code_segment.split('\n')
        processed_lines = []
        modified = False
        for line_content in lines:
            stripped_line = line_content.lstrip()

            if (stripped_line.startswith("print(") or stripped_line.startswith("print (")) and \
               not stripped_line.startswith("#"):
                indentation = line_content[:len(line_content) - len(stripped_line)]
                processed_lines.append(indentation + "# " + stripped_line)
                modified = True
            else:
                processed_lines.append(line_content)
        return '\n'.join(processed_lines), modified

    def _split_code_blocks(self, code: str) -> list[str]:
        """
        Split multi-line code into blocks based on non-indented lines.
        Each block starts with a non-indented line and includes all subsequent
        indented lines until the next non-indented line.
        
        Args:
            code (str): The input code string
            
        Returns:
            list[str]: List of code blocks
        """
        code = code.strip()
        if '\n' not in code:
            return [code]
            
        lines = code.split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped_line = line.strip()
            # Skip empty lines
            if not stripped_line:
                if current_block:
                    current_block.append(line)
                continue
                
            # If line has no indentation and we have a current block
            if line == stripped_line and current_block:
                # Save the current block
                blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        # Don't forget to add the last block
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    def propose_repair(self, repair_plan: str):
        """
        Execute the repair action by delegating to PatchCoder.
        
        Args:
            repair_plan (str): The plan for editing/fixing the code
        
        Returns:
            str: Generated code after repair
        """
        return self.patch_coder.propose_repair(repair_plan)












    def _build_agent_prompt(self, truncate=True) -> str:
        sys_prompt = inspector_prompt_sys.format(example = "")
        if self.dataset=="bigcodebench":
            start_line_no_in_test = -1
            for i, line in enumerate(self.test.splitlines()):
                if f"def {self.focused_test_name}(self" in line:
                    start_line_no_in_test = i
                    break
            if start_line_no_in_test == -1:
                raise ValueError(f"focused test method {self.focused_test_name} not found in test code")
            focused_test_linenum_offset = len(self.code_solution.split('\n')) + start_line_no_in_test
            buggy_code_with_lineno = self._add_line_numbers(self.code_solution)
            test_code_with_lineno = self._add_line_numbers(self.focused_test_source, offset=focused_test_linenum_offset)
            traceback_str = self.focused_test_result['traceback_str']
        elif self.dataset=="livecodebench":
            with open(self.solution_code_file, 'r') as f:
                codelines = f.readlines()
            code_start_idx = 0
            for i, line in enumerate(codelines):
                if "class Solution:" in line or "def solution():" in line:
                    code_start_idx = i + 1
                    break
            test_start_idx = code_start_idx
            test_first_line = self.focused_test_source.strip().split('\n')[0].replace("wrapped_function(", "solution(") 
            for i, line in enumerate(codelines[code_start_idx-1:], start=test_start_idx):
                if test_first_line in line:
                    test_start_idx = i
                    break
            solution_code = ''.join(codelines[code_start_idx-1:test_start_idx]).strip()
            buggy_code_with_lineno = self._add_line_numbers(solution_code, offset=code_start_idx-1).replace("wrapped_function(", "solution(")
            test_code_with_lineno = self._add_line_numbers(self.focused_test_source.strip(), offset=test_start_idx-1)
            traceback_str = self._sanitize_traceback(self.focused_test_result['traceback_str'])

        if truncate and len(self.agent_trace) > 0:
            current_step_idx = 0
            last_step_idx = len(self.agent_trace) - 1
            
            while current_step_idx < last_step_idx:
                temp_trajectory_str = ""
                for i, step in enumerate(self.agent_trace):
                    actions_str = "\n"
                    for action_type, action_argument in step['action']:
                        actions_str = actions_str + action_type + '(' + repr(action_argument) + ')\n'
                    temp_trajectory_str += reason_step_str.format(
                        step_num=step['step_n'], 
                        thought=step['thought'], 
                        actions=actions_str, 
                        observation=truncate_observation(step['observation'], max_lines=20, max_length=500)
                    )
                
                temp_prompt = inspector_prompt_user.format(
                    coding_task = self.coding_task,                          
                    buggy_code = buggy_code_with_lineno,
                    test_method = test_code_with_lineno,
                    execution_result = traceback_str,
                    file=self.solution_code_file,
                    stack_frames=self.current_stack_frames_str,
                    breakpoints=self.current_breakpoints_str,
                    reason_trajectory=temp_trajectory_str
                )
                
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
        reason_trajectory_str = ""
        for i, step in enumerate(self.agent_trace):
            actions_str = "\n"
            for action_type, action_argument in step['action']:
                actions_str = actions_str + action_type + '(' + repr(action_argument) + ')\n'
            reason_trajectory_str += reason_step_str.format(
                step_num=step['step_n'], 
                thought=step['thought'], 
                actions=actions_str, 
                observation=truncate_observation(step['observation'], max_lines=20, max_length=500) if not step['observation'] == '<truncated>' else '<truncated>'
            )

        user_prompt = inspector_prompt_user.format(
                        coding_task = self.coding_task,                          
                        buggy_code = buggy_code_with_lineno,
                        test_method = test_code_with_lineno,
                        execution_result = traceback_str,
                        file=self.solution_code_file,
                        stack_frames=self.current_stack_frames_str,
                        breakpoints=self.current_breakpoints_str,
                        reason_trajectory=reason_trajectory_str)
        solution_code_files = pd.DataFrame(self.solution_trace)['solution_code_file'].unique()
        for file in solution_code_files:
            user_prompt = user_prompt.replace(file, '__test__.py')
        if self.dataset=="livecodebench" and "def solution():" in buggy_code_with_lineno:
            user_prompt.replace("which contains the buggy code and test method.", "which contains the buggy code and test method. IO-based Input data is pre-mocked in test code; No more manual input needed during debugging")
        
        return sys_prompt, user_prompt

    def _sanitize_traceback(self, traceback_text):
        if traceback_text == "" or traceback_text == None:
            return traceback_text
        traceback_text = traceback_text.replace("_inner_call_method", "test_with_mock_input").replace("wrapped_function", "solution")
        transformed = traceback_text.replace('File "<string>"', 'File "__test__.py"')
        
        lines = transformed.split('\n')
        test_file_line_index = -1
        
        for i, line in enumerate(lines):
            if 'File "__test__.py"' in line:
                test_file_line_index = i
                break
        
        if test_file_line_index != -1:
            transformed_lines = lines[test_file_line_index:]
            transformed_lines.insert(0, "Traceback (most recent call last):")
            transformed = '\n'.join(transformed_lines)
        
        return transformed
    
    
    def _build_coder_prompt(self) -> str:
        sys_prompt = coder_prompt_sys
        history_code_version_li = [coder_repair_history.format(
            version=row['solution_version'],
            buggy_code=row['code_solution'],
            stack_trace_and_error = row['focused_test_result']['traceback_str'],
            repair_plan=self.solution_trace[i+1]['repair_plan'] if i+1 < len(self.solution_trace) else self.repair_plan # 把生成下一个版本code时记录的edit plan，与当前版本的code一起展示
        ) for i, row in enumerate(self.solution_trace)]
        user_prompt = coder_prompt_user.format(
            coding_task = self.coding_task,
            history_code_version = "\n\n".join(history_code_version_li)
        )
        return sys_prompt, user_prompt
    def _add_line_numbers(self, code_string, offset=0):
        lines = code_string.split('\n')
        
        max_line_num_width = len(str(len(lines)))
        

        numbered_lines = []
        for i, line in enumerate(lines, offset+1):
            if line:
                line_num = str(i).rjust(max_line_num_width)
                numbered_line = f"{line_num}| {line}"
                numbered_lines.append(numbered_line)
        
        return '\n'.join(numbered_lines)
    def save_debug_trace(self):
        df = pd.DataFrame(self.agent_trace)
        task_id_num = re.search(r'\d+', task_id).group()
        df.to_csv(self.output_dir / Path(f"{task_id_num}_full_agent_trace.csv"), index=False)

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the llm")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--max_solution_version", type=int, default=5)
    parser.add_argument("--ds", type=str, default="bigcodebench", choices=["bigcodebench", "livecodebench"])
    parser.add_argument("--dataset_path", type=str)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)

    parser.add_argument("--ablation", type=str, default="", choices=["", "_woRM", "_woBI", "_wodebugger"])


    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    start = args.start
    end = args.end
    output_dir = f"InspectCoder/results/inspectcoder{args.ablation}_{args.model_name}_{args.ds}"
    Path(output_dir).mkdir(exist_ok=True)
    logger = setup_log(f'{output_dir}/system_{start}_{end}.log', console=False)
    logger.info(args)

    llm = LLM(model_name=args.model_name) # invoke your available LLM here, must implement a "generate" method that return candidate answer in a list.


    agent_list = []
    testset = pd.read_csv(args.dataset_path)
    if args.ablation=="_woRM":
        inspector_prompt_sys = inspector_prompt_woRM_sys
    elif args.ablation=="_woBI":
        inspector_prompt_sys = inspector_prompt_woBI_sys
    else:
        inspector_prompt_sys = inspector_prompt_sys





    for i, row in testset.loc[start:end].iterrows():
        task_id = row['task_id']
        logger.info(f"Start debugging task {task_id}")
        if args.ds=="bigcodebench":
            task_id_num = re.search(r'\d+', row['task_id']).group()
        elif args.ds=="livecodebench":
            task_id_num = f"{i}_{task_id}"
        else:
            raise Exception(f"Unknonw ds type {args.ds}")
        task_output_dir = f"{output_dir}/{task_id_num}"
        os.makedirs(task_output_dir, exist_ok=True)
        task_logger = setup_separate_log(f"{task_output_dir}/{task_id_num}.log", console=False)

        debugger = ProgramInspector(args.ds, row, llm, task_logger, task_output_dir, max_steps=args.max_steps, max_solution_version=args.max_solution_version, temperature=args.temperature, top_p=args.top_p, ablation=args.ablation)
        agent_list.append(debugger)
        try:
            debugger.run()
        except KeyboardInterrupt as e:
            task_logger.error(f"Error occur when debugging task {task_id}. Error: {repr(e)}.\nDebugger status: step_n: {debugger.step_n} code status: {debugger.code_status}, code:\n{debugger.code_solution}\nfocused test: {debugger.focused_test_name}")
            break
        except Exception as e:
            import traceback
            task_logger.error(f"Error occur when debugging task {task_id}. Error: {repr(e)}.\nTraceback: {traceback.format_exc()}\nDebugger status: step_n: {debugger.step_n} code status: {debugger.code_status}, code:\n{debugger.code_solution}\nfocused test: {debugger.focused_test_name}\npropose solution num: {debugger.solution_version}\nmodel call: {debugger.model_call}\ninput token: {debugger.input_token}\noutput token: {debugger.output_token}")
            continue

            
    def calculate_score():
        pass_k = 0
        success_k = 0
        private_pass_k = 0
        private_success_k = 0
        pass_task = []
        fail_task = []
        error_task = []
        private_pass_task = []
        private_fail_task = []
        private_error_task = []
        for agent in agent_list:
            if agent.code_status == "pass":
                pass_k += 1
                success_k += 1
                pass_task.append(agent.task_id)
                if agent.private_code_status == "pass":
                    private_pass_k += 1
                    private_success_k += 1
                    private_pass_task.append(agent.task_id)
                elif agent.private_code_status == "fail":
                    private_success_k += 1
                    private_fail_task.append(agent.task_id)
                else:
                    private_error_task.append(agent.task_id)
            elif agent.code_status == "fail":
                success_k += 1
                fail_task.append(agent.task_id)
            else:
                error_task.append(agent.task_id)

        pass_k /= len(agent_list)
        success_k /= len(agent_list)
        private_pass_k /= len(agent_list)
        private_success_k /= len(agent_list)
        return pass_k, success_k, pass_task, fail_task, error_task, private_pass_k, private_success_k, private_pass_task, private_fail_task, private_error_task
    def calculate_llm_consumption():
        avg_input_token = []
        avg_output_token = []
        avg_model_call = []
        for agent in agent_list:
            avg_input_token.append(agent.input_token)
            avg_output_token.append(agent.output_token)
            avg_model_call.append(agent.model_call)
        avg_input_token = sum(avg_input_token) / len(agent_list)
        avg_output_token = sum(avg_output_token) / len(agent_list)
        avg_model_call = sum(avg_model_call) / len(agent_list)
        return avg_input_token, avg_output_token, avg_model_call

    pass_k, success_k, pass_task, fail_task, error_task, private_pass_k, private_success_k, private_pass_task, private_fail_task, private_error_task = calculate_score()
    logger.info(f"Public Tests:\nPass@k: {pass_k}, Success@k: {success_k}\nPass Task: {pass_task}\nFail Task: {fail_task}\nError Task: {error_task}")
    logger.info(f"Private Tests:\nPass@k: {private_pass_k}, Success@k: {private_success_k}\nPass Task: {private_pass_task}\nFail Task: {private_fail_task}\nError Task: {private_error_task}")
    print(f"Public Tests:\nPass@k: {pass_k}, Success@k: {success_k}\nPass Task: {pass_task}\nFail Task: {fail_task}\nError Task: {error_task}")
    print(f"Private Tests:\nPass@k: {private_pass_k}, Success@k: {private_success_k}\nPass Task: {private_pass_task}\nFail Task: {private_fail_task}\nError Task: {private_error_task}")
    pd.DataFrame({
        "pass_k": [pass_k], 
        "success_k": [success_k], 
        "pass_task": [str(pass_task)], 
        "fail_task": [str(fail_task)], 
        "error_task": [str(error_task)],
        "private_pass_k": [private_pass_k],
        "private_success_k": [private_success_k],
        "private_pass_task": [str(private_pass_task)],
        "private_fail_task": [str(private_fail_task)],
        "private_error_task": [str(private_error_task)]
    }).to_csv(f"{output_dir}/debug_result_{start}_{end}.csv", index=False)

    avg_input_token, avg_output_token, avg_model_call = calculate_llm_consumption()
    logger.info(f"Average input token: {avg_input_token}, Average output token: {avg_output_token}, Average model call: {avg_model_call}")
    print(f"Average input token: {avg_input_token}, Average output token: {avg_output_token}, Average model call: {avg_model_call}")
    pd.DataFrame({"avg_input_token": [avg_input_token], "avg_output_token": [avg_output_token], "avg_model_call": [avg_model_call]}).to_csv(f"{output_dir}/debug_consumption_{start}_{end}.csv", index=False)
