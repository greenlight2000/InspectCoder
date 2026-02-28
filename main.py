import os
import requests
import re
from pathlib import Path
import sys
import os
from inspectcoder_st_ntc import InspectCoder
import pandas as pd
import json
from typing import List, Dict, Any, Tuple
from utils import setup_log, setup_separate_log, read_df
from config.paths import config
sys.path.append(config.get('external_dependencies', 'request_model'))
from request_model import LLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the llm', default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--max_steps', type=int, default=19)
    parser.add_argument('--max_solution_version', type=int, default=5)
    parser.add_argument('--ds', type=str, default='bigcodebench', choices=['bigcodebench', 'livecodebench'])
    parser.add_argument('--experiment', type=str)
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    output_dir = f"{config.get('project', 'results')}/inspectcoder_{args.model_name}_{args.ds}_{args.experiment}"
    Path(output_dir).mkdir(exist_ok=True)
    logger = setup_log(f'{output_dir}/system.log', console=False)
    logger.info(args)
    llm = LLM(args.model_name)
    if args.ds == 'livecodebench':
        testset = pd.read_csv(config.get('datasets', 'livecodebench'))
    elif args.ds == 'bigcodebench':
        testset = pd.read_csv(config.get('datasets', 'bigcodebench'))
    else:
        raise ValueError(f'Unsupported dataset {args.ds}')
    agent_list = []
    for (i, row) in testset.iterrows():
        task_id = row['task_id']
        logger.info(f'Start debugging task {task_id}')
        if args.ds == 'bigcodebench':
            task_id_num = f'{i}_' + re.search('\\d+', row['task_id']).group()
        elif args.ds == 'livecodebench':
            task_id_num = f'{i}_{task_id}'
        else:
            raise Exception(f'Unknonw ds type {args.ds}')
        row['task_id'] = task_id_num
        task_output_dir = f'{output_dir}/{task_id_num}'
        os.makedirs(task_output_dir, exist_ok=True)
        task_logger = setup_separate_log(f'{task_output_dir}/{task_id_num}.log', console=False)
        inspectcoder = InspectCoder(args.ds, row, llm, task_logger, task_output_dir, max_steps=args.max_steps, max_solution_version=args.max_solution_version, temperature=args.temperature, top_p=args.top_p)
        agent_list.append(inspectcoder)
        try:
            inspectcoder.run()
        except Exception as e:
            import traceback
            task_logger.error(f'Error occur when debugging task {task_id}. Error: {repr(e)}.\nTraceback: {traceback.format_exc()}\nDebugger status: step_n: {inspectcoder.step_n} code status: {inspectcoder.code_status}, code:\n{inspectcoder.code_solution}\nfocused test: {inspectcoder.focused_test_name}\npropose solution num: {inspectcoder.solution_version}\nmodel call: {inspectcoder.model_call}\ninput token: {inspectcoder.input_token}\noutput token: {inspectcoder.output_token}')
            logger.error(f'Error occur when debugging task {task_id}. Error: {repr(e)}.\nTraceback: {traceback.format_exc()}\nDebugger status: step_n: {inspectcoder.step_n} code status: {inspectcoder.code_status}, code:\n{inspectcoder.code_solution}\nfocused test: {inspectcoder.focused_test_name}\npropose solution num: {inspectcoder.solution_version}\nmodel call: {inspectcoder.model_call}\ninput token: {inspectcoder.input_token}\noutput token: {inspectcoder.output_token}')
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
            if agent.code_status == 'pass':
                pass_k += 1
                success_k += 1
                pass_task.append(agent.task_id)
                if agent.private_code_status == 'pass':
                    private_pass_k += 1
                    private_success_k += 1
                    private_pass_task.append(agent.task_id)
                elif agent.private_code_status == 'fail':
                    private_success_k += 1
                    private_fail_task.append(agent.task_id)
                else:
                    private_error_task.append(agent.task_id)
            elif agent.code_status == 'fail':
                success_k += 1
                fail_task.append(agent.task_id)
            else:
                error_task.append(agent.task_id)
        pass_k /= len(agent_list)
        success_k /= len(agent_list)
        private_pass_k /= len(agent_list)
        private_success_k /= len(agent_list)
        return (pass_k, success_k, pass_task, fail_task, error_task, private_pass_k, private_success_k, private_pass_task, private_fail_task, private_error_task)

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
        return (avg_input_token, avg_output_token, avg_model_call)
    (pass_k, success_k, pass_task, fail_task, error_task, private_pass_k, private_success_k, private_pass_task, private_fail_task, private_error_task) = calculate_score()
    logger.info(f'Public Tests:\nPass@k: {pass_k}, Success@k: {success_k}\nPass Task: {pass_task}\nFail Task: {fail_task}\nError Task: {error_task}')
    logger.info(f'Private Tests:\nPass@k: {private_pass_k}, Success@k: {private_success_k}\nPass Task: {private_pass_task}\nFail Task: {private_fail_task}\nError Task: {private_error_task}')
    print(f'Public Tests:\nPass@k: {pass_k}, Success@k: {success_k}\nPass Task: {pass_task}\nFail Task: {fail_task}\nError Task: {error_task}')
    print(f'Private Tests:\nPass@k: {private_pass_k}, Success@k: {private_success_k}\nPass Task: {private_pass_task}\nFail Task: {private_fail_task}\nError Task: {private_error_task}')
    pd.DataFrame({'pass_k': [pass_k], 'success_k': [success_k], 'pass_task': [str(pass_task)], 'fail_task': [str(fail_task)], 'error_task': [str(error_task)], 'private_pass_k': [private_pass_k], 'private_success_k': [private_success_k], 'private_pass_task': [str(private_pass_task)], 'private_fail_task': [str(private_fail_task)], 'private_error_task': [str(private_error_task)]}).to_csv(f'{output_dir}/debug_result.csv', index=False)
    (avg_input_token, avg_output_token, avg_model_call) = calculate_llm_consumption()
    logger.info(f'Average input token: {avg_input_token}, Average output token: {avg_output_token}, Average model call: {avg_model_call}')
    print(f'Average input token: {avg_input_token}, Average output token: {avg_output_token}, Average model call: {avg_model_call}')
    pd.DataFrame({'avg_input_token': [avg_input_token], 'avg_output_token': [avg_output_token], 'avg_model_call': [avg_model_call]}).to_csv(f'{output_dir}/debug_consumption.csv', index=False)