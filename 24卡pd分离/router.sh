python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy round_robin \
    --prefill http://x.x.x.212:32000 8998 \
    --decode http://x.x.x.213:33000 \
    --host 127.0.0.1 \
    --mini-lb \
    --port 6688