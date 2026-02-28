import pty
import os
import select
import time
import logging
import re
import sys
import pandas as pd
from config.paths import config
import signal
from contextlib import contextmanager

@contextmanager
def timeout_context(seconds):

    def timeout_handler(signum, frame):
        raise TimeoutError(f'Execution timed out after {seconds} seconds')
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

class PDBTerminal:

    def __init__(self, target_file=''):
        self.ansi_escape = re.compile('\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])')
        self.target_file = target_file
        (self.master_fd, self.slave_fd) = pty.openpty()
        self.pid = os.fork()
        self.session = 'shell'
        self.state = 'start'
        if self.pid == 0:
            os.close(self.master_fd)
            os.dup2(self.slave_fd, 0)
            os.dup2(self.slave_fd, 1)
            os.dup2(self.slave_fd, 2)
            os.close(self.slave_fd)
            os.execlp('/bin/bash', 'bash')
        else:
            os.close(self.slave_fd)
            self.history = []
            output = self._read_output()

    def print_history(self, file_path):
        with open(file_path, 'w') as f:
            for (i, h) in enumerate(self.history):
                f.write(f'[[history {i}:]]\n')
                f.write(f"<<input>>:\n{h['input']}\n<<output>>:\n{h['output']}\n<<post_mode>>:\n{h['post_mode']}\n\n")

    def start_debugging(self):
        import os
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_env and conda_prefix:
            python_path = os.path.join(conda_prefix, 'bin', 'python')
        else:
            python_path = sys.executable
        expected_python_path = config.get('python', 'path')
        command = f'{python_path} -m pdb {self.target_file}'
        output = self.exec_cmd(command)
        return output

    def exec_cmd(self, command):
        try:
            (processed_command, enc_command) = self._pre_process_command(command)
        except ValueError as e:
            return str(e)
        self._send_input(enc_command)
        raw_output = self._read_output()
        output = raw_output[len(command.strip() + '\r\n'):]
        self.history.append({'input': processed_command, 'output': output, 'post_mode': self.session})
        return output

    def _pre_process_command(self, command: 'str|bytes') -> tuple[str | bytes, bytes]:
        if type(command) == str:
            command = command.strip()
            if '\n' in command:
                if self.session == 'python':
                    if all((line.startswith((' ', '\t')) for line in command.splitlines()[1:])):
                        command = command + '\n\n'
                    else:
                        raise ValueError('Multi-line command is only supported for compound statements (e.g. if, for, while, def). For multiple separate statements, please input them individually: {}'.format(command))
                else:
                    raise ValueError('Multi-line command is only allowed in python mode, not supported by pdb mode')
            else:
                command = command + '\n'
            enc_command = command.encode('utf-8')
        elif type(command) == bytes:
            enc_command = command
        else:
            raise TypeError('command must be str or bytes')
        return (command, enc_command)

    def _send_input(self, command: bytes):
        ret = os.write(self.master_fd, command)
        return ret

    def _read_output(self, timeout=5):
        buffer = ''
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                print('Timeout reached. Stopping read.')
                break
            (rlist, _, _) = select.select([self.master_fd], [], [], 0.1)
            if not rlist:
                continue
            data = os.read(self.master_fd, 1024).decode('utf-8')
            if not data:
                break
            data = self.ansi_escape.sub('', data)
            buffer += data
            prompt_to_session = {'(Pdb)': 'pdb', '>>>': 'python', 'In :': 'python', 'InspectCoder$': 'shell', 'InspectCoder#': 'shell'}
            for prompt in list(prompt_to_session.keys()):
                if buffer.strip().endswith(prompt):
                    self.session = prompt_to_session.get(prompt, prompt)
                    return buffer
        return buffer

    def _post_process_output(self, output, command=None):
        if command is not None and type(command) == str:
            if '\n' in command.strip():
                lines = output.splitlines()
                for (i, line) in enumerate(lines):
                    if lines[i].startswith('... '):
                        lines[i] = lines[i][4:]
                output = '\n'.join(lines)[:-1]
            output = output[len(command.strip() + '\r\n'):]
        return output

    def is_alive(self):
        try:
            (pid, status) = os.waitpid(self.pid, os.WNOHANG)
            if pid == 0:
                return True
            else:
                print(f'Subprocess terminated with status {status}.')
                return False
        except ChildProcessError:
            print('Subprocess does not exist.')
            return False

    def ctrl_d(self):
        output = self.exec_cmd(b'\x04')
        return output

    def state_transfer_protocal(self, action, action_output):
        assert self.session == 'pdb', f'Invalid session to perform state transfer: {self.session}'
        state_change = ''
        pre_state = self.state
        if action == 'continue':
            assert pre_state in ['start', 'runtime_state', 'runtime_error', 'post_mortem'], f'Invalid state to perform continue: {pre_state}'
            if 'The program finished and will be restarted' in action_output:
                action_output = action_output.replace('The program finished and will be restarted', '')
                state_change = 'program finished'
                self.curline = ''
                self.state = 'done'
            elif 'Traceback (most recent call last):' in action_output or 'post mortem' in action_output:
                state_change = 'post mortem start'
                self.state = 'post_mortem'
            elif 'Post mortem debugger finished.' in action_output:
                action_output = action_output.replace('Post mortem debugger finished.', '')
                state_change = 'post mortem end'
                self.state = 'start'
            elif '--Return--' in action_output:
                action_output = action_output.replace('--Return--', '')
                state_change = 'return at curent stack frame'
                self.state = 'runtime_state'
            elif '--Call--' in action_output:
                action_output = action_output.replace('--Call--', '')
                state_change = 'call at new stack frame'
                self.state = 'runtime_state'
            else:
                self.state = 'runtime_state'
        elif action == 'restart':
            assert pre_state in ['runtime_state', 'runtime_error', 'post_mortem'], f'Invalid state to perform restart: {pre_state}'
            self.state = 'start'
        elif action == 'interact':
            assert pre_state in ['runtime_state', 'runtime_error', 'post_mortem'], f'Invalid state to perform interact: {pre_state}'
            if 'Traceback (most recent call last):' in action_output:
                state_change = 'runtime_error'
                self.state = 'runtime_error' if pre_state != 'post_mortem' else 'post_mortem'
            else:
                self.state = 'runtime_state' if pre_state != 'post_mortem' else 'post_mortem'
        return (action_output, state_change)

    def close(self):
        os.close(self.master_fd)
        os.waitpid(self.pid, 0)
        print('PDB client closed.')

    def step_in(self):
        r = self.exec_cmd('step')
        lines = r.splitlines()
        state_change = ''
        marker_index = -1
        for (i, line) in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])
        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index + 1:-1])[len('-> '):]
        (post_output, state_change) = self.state_transfer_protocal(action='step_in', action_output=post_output)
        return {'post_output': post_output, 'curstack': curstack, 'curline': curline, 'state_change': state_change}

    def step_out(self):
        r = self.exec_cmd('return')
        lines = r.splitlines()
        state_change = ''
        marker_index = -1
        for (i, line) in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])
        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index + 1:-1])[len('-> '):]
        (post_output, state_change) = self.state_transfer_protocal(action='step_out', action_output=post_output)
        return {'post_output': post_output, 'curstack': curstack, 'curline': curline, 'state_change': state_change}

    def next(self):
        r = self.exec_cmd('next')
        lines = r.splitlines()
        state_change = ''
        marker_index = -1
        for (i, line) in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])
        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index + 1:-1])[len('-> '):]
        (post_output, state_change) = self.state_transfer_protocal(action='next', action_output=post_output)
        return {'post_output': post_output, 'curstack': curstack, 'curline': curline, 'state_change': state_change}

    def continue_(self):
        r = self.exec_cmd('continue')
        lines = r.splitlines()
        marker_index = -1
        for (i, line) in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])
        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index + 1:-1])[len('-> '):]
        if self.state == 'post_mortem':
            (post_output, state_change) = self.state_transfer_protocal(action='restart', action_output=post_output)
        else:
            (post_output, state_change) = self.state_transfer_protocal(action='continue', action_output=post_output)
        return {'post_output': post_output, 'curstack': curstack, 'curline': curline, 'state_change': state_change}

    def restart(self):
        r = self.exec_cmd('restart')
        if self.session == 'shell':
            r = self.start_debugging()
        lines = r.splitlines()
        marker_index = -1
        for (i, line) in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            error_output = '\n'.join(lines) if lines else ''
            raise ValueError(f"No '> ' marker found in: {error_output}")
        post_output = '\n'.join(lines[:marker_index])
        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index + 1:-1])[len('-> '):]
        (post_output, state_change) = self.state_transfer_protocal(action='restart', action_output=post_output)
        return {'post_output': post_output, 'curstack': curstack, 'curline': curline, 'state_change': state_change}

    def where(self):

        def parse_frame_location(line):
            import re
            pattern = r"""
                ^
                \s*
                (?P<file>[^(]+)  
                \((?P<line>\d+)\)     
                (?P<func>[^()]+)         
                \(\)                      
                $
            """
            match = re.match(pattern, line, re.VERBOSE)
            if not match:
                return None
            return {'file': match.group('file').strip(), 'line': int(match.group('line')), 'func': match.group('func').strip()}

        def parse_code_line(line):
            if not line.strip().startswith('->'):
                return None
            return line.strip()[2:].strip()
        frames = []
        lines = self.exec_cmd('where').split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            is_current = line.startswith('>')
            if is_current:
                line = line[1:].strip()
            frame_info = parse_frame_location(line)
            if not frame_info:
                i += 1
                continue
            frame_info['is_current'] = is_current
            code = None
            if i + 1 < len(lines):
                code = parse_code_line(lines[i + 1])
            frame_info['code'] = code
            frames.append(frame_info)
            i += 2 if code else 1
        frames = [frame for frame in frames if not frame['file'].endswith('/bdb.py')]
        return frames[1:]

    def print_variable(self, variable_name):
        return self.exec_cmd(f'p {variable_name}')

    def list_breakpoints(self) -> list[dict]:
        r = self.exec_cmd('break')
        lines = [line.strip() for line in r.split('\n') if line.strip()]
        data_lines = lines[1:]
        result = []
        for line in data_lines:
            if line.startswith('(Pdb)'):
                continue
            if line.strip().startswith('breakpoint already hit '):
                hit_count = int(line.strip().split()[3])
                result[-1]['hit'] = hit_count
                continue
            parts = line.split()
            breakpoint_info = {'num': parts[0], 'type': parts[1], 'disp': parts[2], 'enb': parts[3], 'where': ' '.join(parts[5:]), 'hit': 0}
            result.append(breakpoint_info)
        return result

    def set_breakpoint(self, file, line, condition=None) -> str:
        if file == None:
            file = self.target_file
        r = self.exec_cmd(f'tbreak {file}:{line}')
        return r.replace('(Pdb) ', '').strip()

    def remove_breakpoint(self, file, line):
        if file == None:
            file = self.target_file
        r = self.exec_cmd(f'clear {file}:{line}')
        return r.replace('(Pdb) ', '').strip()

    def interact_python(self, code):
        if self.session == 'python':
            r = self.exec_cmd(code)
        elif self.session == 'pdb':
            r = self.exec_cmd('!import code; namespace = dict(globals()); namespace.update(locals()); code.interact(local=locals().update(namespace) or locals())')
            r = self.exec_cmd(code)
        else:
            raise Exception(f'unknown mode to handle: {self.session}')
        r = r[:-len('>>> ')]
        self.ctrl_d()
        (post_output, state_change) = self.state_transfer_protocal(action='interact', action_output=r)
        return {'post_output': post_output, 'state_change': state_change}

class InspectWare:

    def __init__(self, terminal, ablation, pdb_history_file):
        self.terminal = terminal
        self.ablation = ablation
        self.pdb_history_file = pdb_history_file
        self.current_stack_frames_str = ''
        self.active_breakpoints = []
        self.terminal.start_debugging()
    
    def reinitialize_terminal(self):
        self.terminal.close()
        self.terminal = PDBTerminal(target_file=self.terminal.target_file)
        self.terminal.start_debugging()
        self.update_breakpoints_state()
        self.update_stack_frames_state()

    def set_breakpoint(self, line: int, remove=False):
        if remove:
            obs = self.terminal.remove_breakpoint(file=None, line=line)
        elif line in [int(bp['where'].split(':')[-1]) for bp in self.active_breakpoints]:
            obs = f'Breakpoint at line {line} already exists.'
        else:
            obs = self.terminal.set_breakpoint(file=None, line=line)
            if '*** Blank or comment' in obs:
                obs = f'Illegal Operation: The target line {line} is blank or a comment line, please set breakpoint at a valid line.'
        self.update_breakpoints_state()
        return obs

    def control_execution(self, cmd: str):
        if cmd == 'continue':
            # if self.active_breakpoints == []:
            #     return 'Illegal Operation: ACTION SKIPPED! Set breakpoint before continue execution.'
            result = self.terminal.continue_()
        elif cmd == 'restart':
            if self.terminal.state == 'post_mortem':
                result = self.terminal.continue_()
            elif self.terminal.state == 'start':
                return 'Illegal Operation: ACTION SKIPPED! You are already at the first line of program. No need to restart.'
            else:
                result = self.terminal.restart()
        else:
            return 'Invalid `cmd`. supported values are "continue"|"restart"'
        exec_msg = result['post_output']
        line_num = result['curstack'].split('.py(')[-1].split(')')[0]
        file_name = result['curstack'].split('.py(')[0] + '.py'
        func_name = result['curstack'].split('.py(')[-1].split(')', maxsplit=1)[-1]
        code_line = result['curline']
        state_change = result['state_change']
        self.update_breakpoints_state()
        self.update_stack_frames_state()
        if state_change == 'post mortem start':
            exec_msg = exec_msg.replace("Running 'cont' or 'step' will restart the program", '').replace('Uncaught exception. Entering post mortem debugging', '')
            return f"Execution error before hitting any breakpoint or end of file: {exec_msg}.\nPDB session entering post mortem debugging mode before file {file_name}, line {line_num}| {code_line}\nIn this mode, you can only call (1)interact_code(your code here) to inspect state or (2)control_execution('restart') to exit post mortem mode and restart the program."
        elif state_change == 'post mortem end':
            return f'{exec_msg}\nPost mortem mode end. The program is now restarted and paused before file {file_name}: line {line_num}| {code_line}'
        elif state_change == 'program finished':
            return f'{exec_msg}\nProgram execution finished successfully and automatically restarted. This could indicate the bug is resolved by your previous runtime modification with `interact_code`. You can now propose a repair.'
        else:
            return f'{exec_msg}\nCurrent execution paused before file {file_name}, line {line_num}| {code_line}'

    def interact_code(self, code: str):
        code = code.strip()
        (has_blocking_ops, error_message) = self._safeguard_check(code)
        if has_blocking_ops:
            return error_message

        if '\n' in code:
            result_buffer = []
            original_blocks = self._split_code_blocks(code)
            for block_to_process in original_blocks:
                current_block_code_to_inspect = block_to_process
                result_item = self.terminal.interact_python(current_block_code_to_inspect)
                result_buffer.append(result_item['post_output'])
                if self.terminal.state == 'runtime_error':
                    result_buffer.append('System Hint: This error occurred in the Python REPL environment. The debugged program is still running and you can continue with interact_code or other valid tools.')
                    break
            result = '\n'.join(result_buffer)
        else:
            current_code_to_inspect = code
            result_item = self.terminal.interact_python(current_code_to_inspect)
            result = result_item['post_output']
            if self.terminal.state == 'runtime_error':
                result += '\nSystem Hint: This error occurred in the Python REPL environment. The debugged program is still running and you can continue with interact_code or other valid tools.'
        return result

    def execute_action(self, action_type: str, argument: str | dict):
        err_flag = 0
        try:
            if action_type == 'set_breakpoint':
                val = argument['line'] if type(argument) == dict else argument
                obs = self.set_breakpoint(val)
            elif action_type == 'remove_breakpoint':
                val = argument['line'] if type(argument) == dict else argument
                obs = self.set_breakpoint(argument['line'], remove=True)
            elif action_type == 'control_execution':
                val = argument['cmd'] if type(argument) == dict else argument
                obs = self.control_execution(val)
                if 'Traceback (most recent call last)' in obs or 'PDB Error:' in obs:
                    err_flag = 1
            elif action_type == 'interact_code':
                if self.terminal.state == 'start':
                    raise Exception('ACTION SKIPPED! Please navigate to a suspicious program state using `set_breakpoint(suspicious_line)` and `control_execution("continue")` before opening an interactive REPL environment for debugging. Your debug code should only be INSPECTING/MODIFYING the runtime variable states/behaviors in the buggy program. Do not try to modify tests or the system inputs directly.')
                val = argument['code'] if type(argument) == dict else argument
                try:
                    with timeout_context(6):
                        obs = self.interact_code(val)
                except TimeoutError:
                    err_flag = 1
                    obs = 'Execution timeout, please examine your action.'
                    if 'input()' in val or 'read()' in val:
                        obs += 'Do Not use io operation (eg. input() or read()) as they block execution in pdb. The io input has been mocked into the program.'
                if obs == '':
                    obs = '(Successfully run. No output)'
            else:
                obs = 'Invalid Action. Valid Actions are set_breakpoint, control_execution, interact_code and propose_repair'
        except Exception as e:
            err_flag = 1
            obs = f'{e}'
        return (err_flag, obs)

    def update_breakpoints_state(self):
        self.active_breakpoints = self.terminal.list_breakpoints()
        current_breakpoints_str = []
        for (i, frame) in enumerate(self.terminal.list_breakpoints(), start=1):
            breakpoint_str = f"b{frame['num']} {frame['where']}, hitted {frame['hit']} time."
            current_breakpoints_str.append(breakpoint_str)
        if current_breakpoints_str:
            current_breakpoints_str = '\n'.join(current_breakpoints_str)
        else:
            current_breakpoints_str = 'No breakpoints.'
        self.current_breakpoints_str = 'Active Breakpoints:\n' + current_breakpoints_str

    def update_stack_frames_state(self):
        current_stack_frames_str = []
        for (i, frame) in enumerate(self.terminal.where(), start=1):
            frame_str = f"[{i}] {frame['func']} at {frame['file']}:{frame['line']}| {frame['code']}"
            if frame['is_current']:
                frame_str += ' (current execution paused before this line)'
            current_stack_frames_str.append(frame_str)
        if current_stack_frames_str:
            current_stack_frames_str = '\n'.join(current_stack_frames_str)
        else:
            current_stack_frames_str = "Somethings wrong, fail to execute 'where' in the PDB process."
        self.current_stack_frames_str = 'Current Stack Trace:\n' + current_stack_frames_str


    @staticmethod
    def _split_code_blocks(code: str) -> list[str]:
        code = code.strip()
        if '\n' not in code:
            return [code]
        lines = code.split('\n')
        blocks = []
        current_block = []
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                if current_block:
                    current_block.append(line)
                continue
            if stripped_line.startswith('#'):
                continue
            if line == stripped_line and current_block:
                blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append('\n'.join(current_block))
        return blocks

    @staticmethod
    def _is_input_in_string(line):
        if 'input(' not in line:
            return False
        
        in_string = False
        quote_char = None
        i = 0
        
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                elif line[i:].startswith('input('):
                    return False  # input在字符串外
            else:
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            
            i += 1
        
        return True

    @staticmethod
    def _pdb_code_file_safeguard(code: str) -> str:
        if 'input = sys.stdin.read' in code:
            return code
        original_code = code
        replacement = "sys.stdin.readline().rstrip('\\\\n')"
        lines = code.split('\n')
        for (i, line) in enumerate(lines):
            if line.strip().startswith('#'):
                continue
            if ('"' in line or "'" in line) and 'input(' in line:
                if InspectWare._is_input_in_string(line):
                    continue
            line = re.sub('\\bbuiltins\\.input\\([^)]*\\)', replacement, line)
            line = re.sub('\\binput\\([^)]*\\)', replacement, line)
            lines[i] = line
        code = '\n'.join(lines)
        return code

    @staticmethod
    def _safeguard_check(code: str) -> tuple[bool, str]:
        import re
        patterns = [r'\binput\(', r'\bsys\.stdin\b', r'bos\.read\(\s*0\s*,',r'\bgetpass\.']
        for pattern in patterns:
            if re.search(pattern, code):
                return (True, 'Illegal Operation: input()/sys.stdin/os.read io operations are not allowed for blocking/disrupting PDB input stream. To inspect the mocked inputs, set breakpoints in the source code and use interact_code to examine values at runtime.')
        return (False, '')

if __name__ == '__main__':
    pdb_client = PDBTerminal(target_file='path/to/example_pdb_io_based.py')
    print('----------------------------------')
    print(pdb_client.start_debugging())
    print(pdb_client.set_breakpoint(file=None, line=79))
    print(pdb_client.continue_())
    print(pdb_client.where())
    print(pdb_client.step_in())
    print(pdb_client.where())
    print(pdb_client.next())
    print(pdb_client.list_breakpoints())
    print(pdb_client.continue_())
    print(pdb_client.restart())
    print(pdb_client.continue_())