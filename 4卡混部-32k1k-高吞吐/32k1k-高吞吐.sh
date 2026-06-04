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

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export ASCEND_USE_FIA=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=640
export HCCL_BUFFSIZE=128
unset PYTORCH_NPU_ALLOC_CONF
export SGLANG_ZBAL_LOCAL_MEM_SIZE=60184  # MB 占用的总MEM
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
#export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://127.0.0.1:24669"  # 单机无需配置，多机配置为node0 ip
# zbal if use mix alloc （开启混合分配减少内存碎片）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
# zbal if support graph（need custom pta） （开启图下沉支持）
export ZBAL_ENABLE_GRAPH=1
export ZBAL_HCCL_OP="allreduce,_allgather_base,allgather,broadcast,scatter,reduce_scatter,_reduce_scatter_base,alltoall_base"

MODEL_PATH=/home/weights/MiniMax-M2.5-w8a8-QuaRot
EAGLE_MODEL_PATH=/home/weights/MiniMax-M2.5-eagel-model-0318
export PYTHONPATH=${EAGLE_MODEL_PATH}:$PYTHONPATH
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3

sglang serve \
   --model-path $MODEL_PATH \
   --host 127.0.0.1 \
   --port 32001 \
   --tp-size 8 \
   --disable-radix-cache \
   --mem-fraction-static 0.74 \
   --max-running-requests 24 \
   --chunked-prefill-size -1 --max-prefill-token 32768 \
   --cuda-graph-bs 4 8 12 16 20 24 \
   --moe-a2a-backend deepep --deepep-mode auto --quantization modelslim \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path $EAGLE_MODEL_PATH \
   --speculative-num-steps 3 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 4 \
   --speculative-draft-model-quantization unquant \
   --dtype bfloat16 \
   --trust-remote-code \
   --tokenizer-worker-num 8
