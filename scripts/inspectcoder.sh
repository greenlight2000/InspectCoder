#!/bin/bash

DATASET="livecodebench" # "livecodebench" # "bigcodebench"
DATASET_PATH="LiveCodeBench/lcb_runner/benchmarks/livecodebench-R.csv"
START_INDEX=0
END_INDEX=150 
MODEL_NAME="claude-3-7-sonnet-20250219" # "claude-3-5-sonnet-20241022" # "deepseek-v3" # "gpt-4o" # "claude-3-7-sonnet-20250219" # "qwen-max-2025-01-25-chat"
TEMPERATURE=0
TOP_P=1
MAX_STEPS=20
MAX_SOLUTION_VERSION=5
ABLATION="" # "" # "_woBI" # "_woRM" # "_woBI" "" # "_wodebugger"

cd "InspectCoder"

python inspectcoder.py \
    --ds "$DATASET" \
    --dataset_path "${DATASET_PATH}" \
    --start ${START_INDEX} \
    --end ${END_INDEX} \
    --model_name "${MODEL_NAME}" \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_steps ${MAX_STEPS} \
    --max_solution_version ${MAX_SOLUTION_VERSION} \
    --ablation "${ABLATION}"