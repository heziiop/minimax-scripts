#!/bin/bash
set -euo pipefail

# =============================================================================
# Performance Test: MiniMax-M2.5-w8a8 8p In3k5 Out1k5 High Throughput
#
# 前置条件：SGLang 服务已手动拉起，且服务端口与下方 --port 一致
#
# 底层工具: python3 -m sglang.bench_serving
# 原始用例: TestNPUMiniMaxM2_5_W8A8_8P_In3k5_Out1k5_HighThroughput
# =============================================================================

PORT=6677

export PYTHONPATH=/home/xxx/code/sglang/python:$PYTHONPATH

python3 -m sglang.bench_serving \
    --host 127.0.0.1 \
    --port ${PORT} \
    --model /home/weights/MiniMax-M2.5-w8a8-QuaRot \
    --backend sglang-oai-chat \
    --dataset-name random \
    --random-input-len 3500 \
    --random-output-len 1500 \
    --max-concurrency 112 \
    --num-prompts 448 \
    --random-range-ratio 1 \
    --dataset-path /home/xxx/ShareGPT_V3_unfiltered_cleaned_split.json
