python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy round_robin \
    --prefill http://xxx.xxx.xxx.238:32000 8998 \
    --decode http://xxx.xxx.xxx.237:33000 \
    --host 127.0.0.1 \
    --mini-lb \
    --port 6688
