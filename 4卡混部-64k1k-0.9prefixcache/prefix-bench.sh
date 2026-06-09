#!/bin/bash

# 使用这个仓库的prefix cache测试工具https://gitcode.com/lauare/aisbench_auto_tools_prefix
python3 aisbench_test.py \
  --input_len 65536 \
  --output_len 1024 \
  --data_num 104 \
  --concurrency 26 \
  --request_rate 0 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test \
  --dp 1
