echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

export PYTHONPATH=/home/xxx/code/sglang/python:$PYTHONPATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export TASK_QUEUE_ENABLE=1

export ASCEND_USE_FIA=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_FUSED_MOE_MODE=2
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160000

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1


export HCCL_BUFFSIZE=1024


MODEL_PATH=/home/weights/MiniMax-M2.5-w8a8-QuaRot
EAGLE_MODEL_PATH=/home/weights/MiniMax-M2.5-eagel-model-0318
export PYTHONPATH=${EAGLE_MODEL_PATH}:$PYTHONPATH
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3

python -m sglang.launch_server \
   --model-path $MODEL_PATH \
   --host 127.0.0.1 \
   --port 6677 \
   --tp-size 16 \
   --dp-size 2 \
   --enable-dp-attention \
   --mem-fraction-static 0.65 \
   --max-running-requests 20 \
   --reasoning-parser minimax-append-think \
   --tool-call-parser minimax-m2 \
   --enable-prefill-delayer \
   --prefill-max-requests 4 \
   --chunked-prefill-size 160000 \
   --max-prefill-token 80000 \
   --cuda-graph-bs 2 4 6 8 10 \
   --moe-a2a-backend ascend_fuseep --deepep-mode auto --quantization modelslim \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path $EAGLE_MODEL_PATH \
   --speculative-num-steps 3 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 4 \
   --speculative-draft-model-quantization unquant \
   --tokenizer-worker-num 4 \
   --dtype bfloat16
