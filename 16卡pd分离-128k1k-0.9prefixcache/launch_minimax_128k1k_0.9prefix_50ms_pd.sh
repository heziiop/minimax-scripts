export PYTHONPATH=/home/h00848570/code/sglang-community/python:$PYTHONPATH
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export ASCEND_MF_STORE_URL="tcp://xxx.xxx.xxx.238:24667"

P_IP=('xxx.xxx.xxx.238')

D_IP=('xxx.xxx.xxx.237')
D_MASTER="${D_IP[0]}:8001"
MODEL_PATH=/mnt/weights/MiniMax-M2.5-w8a8-QuaRot

export PYTHONPATH=/mnt/weights/MiniMax-M2.5-eagel-model-0318:$PYTHONPATH
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_eagle3

#export SGLANG_EXPERIMENTAL_CPP_RADIX_TREE=1


LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

# prefill
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
	export ENABLE_PROFILING=0
	export PROFILING_STAGE="prefill"
        export PROFILING_BS=8
        export PROFILING_step=30

        export HCCL_SOCKET_IFNAME=enx9c69d3020bab
        export GLOO_SOCKET_IFNAME=enx9c69d3020bab
	export ASCEND_USE_FIA=1
        export HCCL_BUFFSIZE=2500
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \
        --port 32000 --disaggregation-bootstrap-port $((8998+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \
        --tp-size 16 --mem-fraction-static 0.43 --attention-backend ascend --device npu --quantization modelslim \
        --disaggregation-transfer-backend ascend --max-running-requests 128 \
        --chunked-prefill-size -1 --max-prefill-tokens 130000 --moe-a2a-backend deepep --deepep-mode normal \
	--tokenizer-worker-num 16 \
        --dp-size 2 --enable-dp-attention --dtype bfloat16 --load-balance-method round_robin \
	--speculative-algorithm EAGLE3 \
        --speculative-draft-model-path /mnt/weights/MiniMax-M2.5-eagel-model-0318 \
        --speculative-num-steps 2 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 3 \
        --speculative-draft-model-quantization unquant --skip-server-warmup
        NODE_RANK=$i
        break
    fi
done

# decode
for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "${D_IP[$i]}"
	export HCCL_BUFFSIZE=1600
	#export DEEPEP_HCCL_BUFFSIZE=512
	export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=640
        export HCCL_SOCKET_IFNAME=enx9c69d302197d
        export GLOO_SOCKET_IFNAME=enx9c69d302197d
	export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
	export SGLANG_NPU_FUSED_MOE_MODE=2
	export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=96

        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode decode --host ${D_IP[$i]} \
	--cuda-graph-bs 2 4 8 \
        --port 33000 --trust-remote-code \
        --tp-size 16 --mem-fraction-static 0.76 --attention-backend ascend --device npu --quantization modelslim \
	--nnodes 1 --node-rank $i --dist-init-addr $D_MASTER \
        --disaggregation-transfer-backend ascend --max-running-requests 80 \
        --chunked-prefill-size -1 --moe-a2a-backend ascend_fuseep --deepep-mode low_latency \
	--tokenizer-worker-num 8 \
        --dp-size 2 --enable-dp-attention --dtype bfloat16 \
        --load-balance-method round_robin \
	--speculative-algorithm EAGLE3 \
        --speculative-draft-model-path /mnt/weights/MiniMax-M2.5-eagel-model-0318 \
        --speculative-num-steps 2 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 3 \
        --speculative-draft-model-quantization unquant \
	--disaggregation-enable-decode-radix-cache --skip-server-warmup

        NODE_RANK=$i
        break
    fi
done

