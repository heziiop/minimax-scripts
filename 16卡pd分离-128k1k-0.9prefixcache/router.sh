python -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy round_robin \
    --prefill http://141.61.39.238:32000 8998 \
    --decode http://141.61.39.237:33000 \
    --host 127.0.0.1 \
    --mini-lb \
    --port 6688
