#!/bin/bash
# Usage: bash serve.sh [name=adapter_path ...]
# Starts vLLM (Qwen2.5-7B-Instruct) with the given LoRA modules on port 8000 and waits until ready.
# Stop any previous server first (e.g. pkill -f "vllm.entrypoints").
LORA_ARGS=""
if [ $# -gt 0 ]; then LORA_ARGS="--lora-modules $*"; fi
nohup env VLLM_USE_FLASHINFER_SAMPLER=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.88 --max-model-len 4096 \
  --enable-lora --max-lora-rank 16 $LORA_ARGS > vllm.log 2>&1 < /dev/null &
for i in $(seq 1 120); do
  if curl -s -m 3 localhost:8000/v1/models | grep -q Qwen; then echo SERVER_READY; curl -s localhost:8000/v1/models | python3 -c "import json,sys;print([m['id'] for m in json.load(sys.stdin)['data']])"; exit 0; fi
  if grep -q Traceback vllm.log; then echo SERVER_FAILED; grep -A5 Traceback vllm.log | tail -8; exit 1; fi
  sleep 5
done
echo SERVER_TIMEOUT; exit 1
