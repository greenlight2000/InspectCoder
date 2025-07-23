import logging
import pandas as pd
from pathlib import Path
import tiktoken
import re
import ast
from typing import Tuple, Optional

import os
import time
import datetime
import textwrap

OPEN_PYTHON_SESSION_SCRIPT = '!import code; namespace = dict(globals()); namespace.update(locals()); code.interact(local=locals().update(namespace) or locals())'

def setup_log(log_file_path:Path, level:int=logging.INFO, file=True, console=True):
    try:
        time.tzset()
    except AttributeError:
        pass
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    formatter = logging.Formatter(fmt='%(asctime)s  %(levelname)-8s [%(filename)s -> %(funcName)s]: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    if file:
        file_handler = logging.FileHandler(filename=str(log_file_path), mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def setup_separate_log(log_file_path:Path, level:int=logging.INFO, file=True, console=True, propagate=False):
    logger = logging.getLogger(str(log_file_path))
    logger.propagate = propagate
    logger.setLevel(level)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    formatter = logging.Formatter(fmt='%(asctime)s  %(levelname)-8s [%(filename)s -> %(funcName)s]: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    if file:
        file_handler = logging.FileHandler(filename=str(log_file_path), mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def read_df(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('json'):
        return pd.read_json(file_path)
    elif file_path.endswith('jsonl'):
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def get_function_call(code_string, function_name):
    pattern = rf'{function_name}\s*\([^)]*\)'
    matches = re.finditer(pattern, code_string)

    function_calls = []
    for match in matches:
        function_calls.append(match.group(0))
            
    return function_calls

def extract_method_source(source_code: str, method_name: str, dedentation: bool = True) -> Tuple[Optional[str], int, int]:
    """
    Extract the source code of a specific method from a class.
    
    Args:
        source_code (str): The complete source code containing the method
        method_name (str): The name of the method to extract
        dedentation (bool, optional): Whether to remove the common indentation. Defaults to True.
    
    Returns:
        Tuple[Optional[str], int, int]: A tuple containing:
            - The extracted method source code including decorators and docstring,
              or None if method not found
            - The start line number in the source file (0 if method not found)
            - The end line number in the source file (0 if method not found)
    """
    # Parse the source code into an AST
    tree = ast.parse(source_code)
    
    # Find the method node
    method_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            method_node = node
            break
    
    if method_node is None:
        return None, 0, 0
        
    # Get the line numbers for the method
    start_line = method_node.lineno
    end_line = method_node.end_lineno
    
    # Split source code into lines
    source_lines = source_code.splitlines()
    
    # Extract all lines of the method
    method_lines = source_lines[start_line - 1:end_line]
    
    if dedentation:
        # Get the method's indentation level from its first line
        first_line = source_lines[start_line - 1]
        indentation = len(first_line) - len(first_line.lstrip())
        for i, line in enumerate(method_lines):
            if line.strip():
                method_lines[i] = line[indentation:]

            
    # Join the lines back together and return with line numbers
    return '\n'.join(method_lines)

def extract_code_block(text)->str:
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.replace('```','').strip()
    

import ast

def get_function_names(code_string, include_nested=True):
    try:
        tree = ast.parse(code_string)
        
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, include_nested):
                self.function_names = []
                self.nested_level = 0
                self.include_nested = include_nested
                
            def visit_FunctionDef(self, node):
                if self.include_nested or self.nested_level == 0:
                    self.function_names.append(node.name)
                
                self.nested_level += 1
                self.generic_visit(node)
                self.nested_level -= 1
        
        visitor = FunctionVisitor(include_nested)
        visitor.visit(tree)
        
        return visitor.function_names
        
    except SyntaxError:
        return "代码字符串包含语法错误"

def extract_imports(code_string: str) -> list[str]:
    try:
        tree = ast.parse(code_string)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.asname:
                        imports.append(f"import {name.name} as {name.asname}")
                    else:
                        imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = []
                for name in node.names:
                    if name.asname:
                        names.append(f"{name.name} as {name.asname}")
                    else:
                        names.append(name.name)
                level_dots = "." * node.level
                if len(names) > 3:
                    names_str = "(\n    " + ",\n    ".join(names) + "\n)"
                else:
                    names_str = ", ".join(names)
                if module:
                    imports.append(f"from {level_dots}{module} import {names_str}")
                else:
                    imports.append(f"from {level_dots} import {names_str}")
        return imports
    except SyntaxError:
        return "error"
    
    try:
        tree = ast.parse(code_string)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.asname:
                        imports.append(f"import {name.name} as {name.asname}")
                    else:
                        imports.append(f"import {name.name}")
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = []
                for name in node.names:
                    if name.asname:
                        names.append(f"{name.name} as {name.asname}")
                    else:
                        names.append(name.name)
                
                level_dots = "." * node.level
                
                if len(names) > 3:
                    names_str = "(\n    " + ",\n    ".join(names) + "\n)"
                else:
                    names_str = ", ".join(names)
                
                if module:
                    imports.append(f"from {level_dots}{module} import {names_str}")
                else:
                    imports.append(f"from {level_dots} import {names_str}")
                    
        return imports
        
    except SyntaxError as e:
        return [f"Error parsing code: {str(e)}"]