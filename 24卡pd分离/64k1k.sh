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

export ASCEND_MF_STORE_URL="tcp://x.x.x.212:24667"

P_IP=('x.x.x.212')

D_IP=('x.x.x.213' 'x.x.x.214')
D_MASTER="${D_IP[0]}:8001"
MODEL_PATH=/home/weights/MiniMax-M2.5-w8a8-QuaRot

export PYTHONPATH=/home/weights/MiniMax-M2.5-eagel-model-0318-upload:$PYTHONPATH
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
	    export ENABLE_PROFILING=1
        export PROFILING_BS=30
        export PROFILING_step=8

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
        --chunked-prefill-size -1 --max-prefill-tokens 58000 --moe-a2a-backend deepep --deepep-mode auto \
	    --tokenizer-worker-num 16 \
        --dp-size 2 --enable-dp-attention --dtype bfloat16 --load-balance-method round_robin \
	    --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path /home/weights/MiniMax-M2.5-eagel-model-0318-upload \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --speculative-draft-model-quantization unquant --disable-radix-cache
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
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
	    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
	    export SGLANG_NPU_FUSED_MOE_MODE=2
        #export SGLANG_NPU_DEEPEP_USE_FUSED_MOE_DECODE=1
        python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode decode --host ${D_IP[$i]} \
	    --cuda-graph-bs 8 16 24 \
        --port 33000 --trust-remote-code \
        --tp-size 32 --mem-fraction-static 0.6 --attention-backend ascend --device npu --quantization modelslim \
	    --nnodes 2 --node-rank $i --dist-init-addr $D_MASTER \
        --disaggregation-transfer-backend ascend --max-running-requests 96 \
        --chunked-prefill-size -1 --max-prefill-tokens 65536 --moe-a2a-backend ascend_fuseep --deepep-mode low_latency \
	    --tokenizer-worker-num 16 \
        --dp-size 4 --enable-dp-attention --dtype bfloat16 \
        --load-balance-method round_robin \
	    --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path /home/weights/MiniMax-M2.5-eagel-model-0318-upload \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --speculative-draft-model-quantization unquant --disable-radix-cache

        NODE_RANK=$i
        break
    fi
done

