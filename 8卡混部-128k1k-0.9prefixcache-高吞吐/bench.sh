#!/bin/bash
set -e


export PYTHONPATH=/home/xxx/code/sglang/python:$PYTHONPATH

# ===== Configuration =====
HOST="127.0.0.1"
PORT="6677"
MODEL="/home/weights/MiniMax-M2.5-w8a8-QuaRot"
DP=2


# ===== Run benchmark =====
python3 -m sglang.bench_serving \
    --backend sglang \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL}" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 80 \
    --gsp-system-prompt-len 117964 \
    --gsp-question-len 13108 \
    --gsp-output-len 1024 \
    --gsp-range-ratio 1.0 \
    --request-rate inf \
    --warmup-requests ${DP} \
    --max-concurrency 20
