echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# 网卡
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

#export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1

#export ENABLE_MOE_NZ=1

#export SGLANG_USE_FIA_NZ=1



export ASCEND_USE_FIA=1
export HCCL_BUFFSIZE=1600
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=640
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export SGLANG_NPU_FUSED_MOE_MODE=2
export SGLANG_NPU_DEEPEP_USE_FUSED_MOE_DECODE=1
export SGLANG_NPU_FUSEEP_DECODE_ONLY=1


export ENABLE_PROFILING=0
export PROFILING_BS=28
export PROFILING_STAGE="decode"
export PROFILING_step=10


export PYTHONPATH=/home/weights/MiniMax-M2.5-eagel-model-0318-upload:$PYTHONPATH
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3

sglang serve \
   --model-path /home/weights/MiniMax-M2.5-w8a8-QuaRot \
   --host 127.0.0.1 \
   --port 32000 \
   --tp-size 16 \
   --dp-size 2 \
   --enable-dp-attention \
   --prefill-delayer-max-delay-passes 100 \
   --enable-prefill-delayer \
   --mem-fraction-static 0.65 \
   --max-running-requests 36 \
   --chunked-prefill-size -1 --max-prefill-token 130000 \
   --cuda-graph-bs 8 16 24 \
   --moe-a2a-backend ascend_fuseep --deepep-mode auto --quantization modelslim \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path /home/weights/MiniMax-M2.5-eagel-model-0318-upload \
   --speculative-num-steps 3 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 4 \
   --speculative-draft-model-quantization unquant \
   --dtype bfloat16 \
   --trust-remote-code \
   --tokenizer-worker-num 8 \

