import pty
import os
import select
import time
import logging
import re
from utils import OPEN_PYTHON_SESSION_SCRIPT
class InspectWare:
    """Encapsulation and abstraction of pdb features, providing an interactive interface for InspectCoder agents. See pdb features in https://docs.python.org/3/library/pdb.html"""
    def __init__(self, target_file=""):
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        self.target_file = target_file
        self.master_fd, self.slave_fd = pty.openpty()  # open a pseudo terminal in master process
        self.pid = os.fork()  # fork a child process to run in parallel
        self.state = "start" # start, runtime_state, post_mortem, runtime_error, done

        if self.pid == 0:
            os.close(self.master_fd)
            os.dup2(self.slave_fd, 0)
            os.dup2(self.slave_fd, 1)
            os.dup2(self.slave_fd, 2)
            os.close(self.slave_fd)

            os.execlp("/bin/bash", "bash")

        else:  # parent process handles communication with the pseudo terminal
            os.close(self.slave_fd)
            self.history = []
            self.mode = "shell"
            self.prompt_to_mode = {
                "(Pdb)": "pdb",
                ">>>": "python",
                "In :": "python",
                "$": "shell",
                "#": "shell",
            }
            output = self._read_output()
            print("output:",output, "output end")
            print("Terminal initialized. Running bash shell.")
    def print_history(self, file_path):
        with open(file_path, 'w') as f:
            for i, h in enumerate(self.history):
                f.write(f"[[history {i}:]]\n")
                f.write(f"<<input>>:\n{h['input']}\n<<output>>:\n{h['output']}\n<<post_mode>>:\n{h['post_mode']}\n\n")
    def start_debugging(self):
        command = f"YOUR_PYTHON_ENV_HERE -m pdb {self.target_file}"
        output = self.exec_cmd(command)
        return output
    def exec_cmd(self, command):
        """
        Send a command to the pseudo terminal and read the output.
        """
        try:
            processed_command, enc_command = self._pre_process_command(command)
        except ValueError as e:
            return str(e)
        self._send_input(enc_command)
        output = self._read_output()
        output = self._post_process_output(output, processed_command)
        self.history.append({
            "input": processed_command,
            "output": output,
            "post_mode": self.mode
        })
        logging.info(f"input: {processed_command}\noutput: {output}\npost_mode: {self.mode}")
        return output
    def _pre_process_command(self, command:'str|bytes')->tuple[str|bytes, bytes]:
        if type(command)==str:
            command = command.strip()
            if '\n' in command:
                if self.mode == 'python':
                    if all(line.startswith((' ', '\t')) for line in command.splitlines()[1:]):
                        command = command + "\n\n"
                    else:
                        raise ValueError("Multi-line command is only supported for compound statements (e.g. if, for, while, def). For multiple separate statements, please input them individually: {}".format(command))
                else:
                    raise ValueError("Multi-line command is only allowed in python mode, not supported by pdb mode")
            else:
                command = command + "\n"
            enc_command = command.encode("utf-8")
        elif type(command)==bytes: # for keyboard signal like ctrl-D
            enc_command = command
        else:
            raise TypeError("command must be str or bytes")
        return command, enc_command
    def _send_input(self, command:bytes):
        """
        Send command to the pseudo terminal.
        """
        ret = os.write(self.master_fd, command)
        return ret # ret is the number of bytes written
    def _read_output(self, timeout=5):
        """
        Read output from the pseudo terminal until timeout or a prompt is detected.
        
        :param timeout: Timeout in seconds
        :param prompts: List of prompts (e.g. ["$ ", "# ", "(Pdb) "])
        :return: Output string
        """
        buffer = ""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                print("Timeout reached. Stopping read.")
                break

            rlist, _, _ = select.select([self.master_fd], [], [], 0.1)
            if not rlist:
                continue

            data = os.read(self.master_fd, 1024).decode("utf-8")
            if not data:
                break
            data = self.ansi_escape.sub('', data)
            buffer += data

            for prompt in list(self.prompt_to_mode.keys()):
                if buffer.strip().endswith(prompt):
                    self.mode = self.prompt_to_mode.get(prompt, prompt)
                    return buffer

        return buffer
    def _post_process_output(self, output, command=None):
        """
        Post-process output from the pseudo terminal.
        """
        if command is not None and type(command)==str:
            if '\n' in command.strip():
                lines = output.splitlines()
                for i, line in enumerate(lines):
                    if lines[i].startswith('... '):
                        lines[i] = lines[i][4:]
                output = '\n'.join(lines)[:-1]
            output = output[len(command.strip()+'\r\n'):]
        return output
    def is_alive(self):
        """
        Check if the child process is still running.
        """
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid == 0:
                return True
            else:
                print(f"Subprocess terminated with status {status}.")
                return False
        except ChildProcessError:
            print("Subprocess does not exist.")
            return False
    def ctrl_d(self):
        """
        Send Ctrl+D signal to the pseudo terminal. Used to exit interact mode and return to the pdb command line. The next command must be a pdb built-in command; otherwise, the command line will return to interact mode.
        """
        output = self.exec_cmd(b'\x04')
        return output
    def state_transition_protocal(self, post_output, curline):
        """
        Process PDB output, extract add_info status, and clean post_output.
        
        This is a utility function for handling various status markers in PDB output and returning the corresponding add_info and cleaned post_output.
        It serves as the protocol handler for PDB terminal communication.
        
        Args:
            post_output (str): Output after executing a PDB command
            curline (str, optional): Current code line, may be cleared in some states
        Returns:
            tuple: (processed post_output, add_info status, possibly modified curline)
        """
        add_info = ''
        self.state = "runtime_state"

        if 'The program finished and will be restarted' in post_output:
            post_output = post_output.replace('The program finished and will be restarted', '')
            add_info = "program finished"
            curline = ''
            self.state = "done"
        
        elif "Traceback (most recent call last):" in post_output or "post mortem" in post_output:
            add_info = "post mortem start"
            self.state = "post_mortem"
        
        elif "Post mortem debugger finished." in post_output:
            post_output = post_output.replace("Post mortem debugger finished.", "")
            add_info = "post mortem end"
            self.state = "start"
        
        elif '--Return--' in post_output:
            post_output = post_output.replace('--Return--', '')
            add_info = "return at curent stack frame"
        
        elif '--Call--' in post_output:
            post_output = post_output.replace('--Call--', '')
            add_info = "call at new stack frame"
            
        return post_output, add_info, curline
        
    def close(self):
        """
        Close the pseudo terminal and terminate the child process.
        """
        os.close(self.master_fd)
        os.waitpid(self.pid, 0)
        print("PDB client closed.")

    def step_in(self):
        """
        Step into (enter function).
        """
        r = self.exec_cmd("step")
        lines = r.splitlines()
        add_info = ''

        marker_index = -1
        for i, line in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])

        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index+1:-1])[len('-> '):]

        post_output, add_info, curline = self.state_transition_protocal(post_output, curline)
        return {
            "post_output": post_output,
            "curstack": curstack,
            "curline": curline,
            "state": self.state,
            "add_info": ""
        }
    def step_out(self):
        """
        Step out (leave function).
        """
        r = self.exec_cmd("return")
        lines = r.splitlines()
        add_info = ''

        marker_index = -1
        for i, line in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])

        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index+1:-1])[len('-> '):]

        post_output, add_info, curline = self.state_transition_protocal(post_output, curline)
        return {
            "post_output": post_output,
            "curstack": curstack,
            "curline": curline,
            "state": self.state,
            "add_info": ""
        }

    def next(self):
        """
        Execute a single step (step over functions).
        Example intermediate raw output:
        > example_buggycode.py(9)<module>()
        -> x = 10
        (Pdb) 
        """
        r = self.exec_cmd("next")
        lines = r.splitlines()
        add_info = ''

        marker_index = -1
        for i, line in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])

        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index+1:-1])[len('-> '):]

        post_output, add_info, curline = self.state_transition_protocal(post_output, curline)
        return {
            "post_output": post_output,
            "curstack": curstack,
            "curline": curline,
            "state": self.state,
            "add_info": ""
        }
        
    def continue_(self):
        """
        Continue execution until the next breakpoint.
        Example intermediate raw output or unexpected error message during execution:
        > /path/to/file.py(9)<module>()
        -> x = 10
        (Pdb) 
        Return:
        {
            "post_output": "",  # Output printed during program execution
            "curstack": "/path/to/file.py(9)<module>()", # Current stack frame info: FILE_PATH(LINE_NO)FUNC_NAME()
            "curline": "x = 10", # Current (not yet executed) code line
            "add_info": "" # Additional info, e.g., debug status (return at current stack frame, program finished)
        }
        """
        r = self.exec_cmd("continue")

        lines = r.splitlines()
        add_info = ''

        marker_index = -1
        for i, line in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])

        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index+1:-1])[len('-> '):]

        post_output, add_info, curline = self.state_transition_protocal(post_output, curline)
        return {
            "post_output": post_output,
            "curstack": curstack,
            "curline": curline,
            "state": self.state,
            "add_info": ""
        }
    def restart(self):
        """
        Restart the program.
        Example intermediate raw output:
        Restarting /path/to/file.py with arguments:
            
        > /path/to/file.py(1)<module>()
        -> import pandas as pd
        (Pdb) 
        """
        r = self.exec_cmd("restart")
        if self.mode == 'shell':
            r = self.start_debugging()
        lines = r.splitlines()

        marker_index = -1
        for i, line in enumerate(lines):
            if line.startswith('> '):
                marker_index = i
                break
        if marker_index == -1:
            raise ValueError(f"No '> ' marker found in: {lines}")
        post_output = '\n'.join(lines[:marker_index])
        curstack = lines[marker_index][len('> '):]
        curline = ''.join(lines[marker_index+1:-1])[len('-> '):]
        return {
            "post_output": post_output,
            "curstack": curstack,
            "curline": curline,
            "state": self.state,
            "add_info": ""
        }
    def where(self):
        """
        Parse the output of the pdb 'where' command and return the call stack information.
        Example output of the where command:
          /path/to/python/bdb.py(598)run()
        -> exec(cmd, globals, locals)
          <string>(1)<module>()
          /path/to/file.py(15)<module>()
        -> a = pd.DataFrame([1, 2, 3])
          /path/to/python/frame.py(669)init()
        -> manager = get_option("mode.data_manager")
        > /path/to/python/config.py(260)call()
        -> def call(self, *args, **kwds) -> T:
        (Pdb) 
        The return value is a list, each element is a dictionary containing a single frame's information:
        Example return value:
            [
                {
                    'file': '/path/to/file.py',
                    'line': 10,
                    'func': 'main',
                    'code': 'result = calculate(x, y)',
                    'is_current': True
                },
                ...
            ]
        """
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
                
            return {
                'file': match.group('file').strip(),
                'line': int(match.group('line')),
                'func': match.group('func').strip()
            }
        
        def parse_code_line(line):
            if not line.strip().startswith('->'):
                return None
            return line.strip()[2:].strip()
        
        frames = []
        lines = self.exec_cmd("where").split('\n')
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
        return self.exec_cmd(f"p {variable_name}")

    def list_breakpoints(self)->list[dict]:
        """
        # intermediate raw output:
        Num Type         Disp Enb   Where
        1   breakpoint   keep yes   at example_buggycode.py:1
        2   breakpoint   keep yes   at example_buggycode.py:10
        (Pdb) 
        # return parsed output:
        [
            {
                'num': '1',
                'type': 'breakpoint',
                'disp': 'keep',
                'enb': 'yes',
                'where': '/path/to/file.py:1',
                'hit': 0
            },
            ...
        ]
        """
        r = self.exec_cmd("break")
        lines = [line.strip() for line in r.split('\n') if line.strip()]
        header = lines[0]
        data_lines = lines[1:]
        columns = ['num', 'type', 'disp', 'enb', 'where']  
        result = []
        for line in data_lines:
            if line.startswith('(Pdb)'):
                continue
            if line.strip().startswith('breakpoint already hit '):
                hit_count = int(line.strip().split()[3])  
                result[-1]['hit'] = hit_count
                continue
                
            parts = line.split()
            breakpoint_info = {
                'num': parts[0],
                'type': parts[1],
                'disp': parts[2],
                'enb': parts[3],
                'where': ' '.join(parts[5:]),
                'hit': 0
            }
            result.append(breakpoint_info)
        return result

    def set_breakpoint(self, file, line, condition=None)->str:
        """
        place breakpoint
        return:
        'Breakpoint 2 at file.py:1'
        """
        if file==None:
            file = self.target_file
        r = self.exec_cmd(f"tbreak {file}:{line}")
        return r.replace('(Pdb) ','').strip()
    def remove_breakpoint(self, file, line):
        """
        remove breakpoint
        """
        if file==None:
            file = self.target_file
        r = self.exec_cmd(f"clear {file}:{line}")
        return r.replace('(Pdb) ','').strip()

    def start_python(self):
        if self.mode == 'pdb':
            r = self.exec_cmd('interact')
        elif self.mode == 'python':
            r = "you are already at python interact mode, start interact with the code by calling interact_code()"
        elif self.mode == 'shell':
            r = self.exec_cmd('python')
            print(r)
        else:
            raise Exception(f'unknown mode to handle: {self.mode}')
        return r
    def end_python(self):
        if self.mode == 'python':
            r = self.ctrl_d()
        elif self.mode == 'pdb':
            r = "you are already at pdb mode"
        else:
            raise Exception(f'unknown mode to handle: {self.mode}')
        return r
    def interact_code(self, code):
        if self.mode == 'python':
            r = self.exec_cmd(code)
        elif self.mode == 'pdb':
            r = self.exec_cmd(OPEN_PYTHON_SESSION_SCRIPT)
            r = self.exec_cmd(code)
        else:
            raise Exception(f'unknown mode to handle: {self.mode}')
        r = r[:-len('>>> ')]
        self.ctrl_d()
        return r

if __name__ == "__main__":
    pdb_client = InspectWare(target_file="InspectCoder/io_based_pdb.py")

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