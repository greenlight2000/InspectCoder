#!/bin/bash
DATASET="bigcodebench" # "livecodebench"
MODEL_NAME="qwen-max"

TEMPERATURE=0
TOP_P=1
MAX_STEPS=19
MAX_SOLUTION_VERSION=5
EXPERIMENT="test" 

cd "/path/to/InspectCoder"

python main.py \
    --ds "$DATASET" \
    --model_name "${MODEL_NAME}" \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_steps ${MAX_STEPS} \
    --max_solution_version ${MAX_SOLUTION_VERSION} \
    --experiment "${EXPERIMENT}"