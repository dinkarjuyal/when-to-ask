#!/bin/bash
# v2: value-of-information rationale distillation. Resumable; each step skipped if output exists.
# Round 1 collects from the base model WITH the numeric template (teacher prompt);
# later rounds collect from the previous adapter with the PLAIN prompt (self-improvement).
# Training prompts never contain the template. Eval always uses the plain prompt.
ROUNDS=${1:-2}
cd "$(dirname "$0")"
st() { echo "$(date +%H:%M:%S) $*" | tee -a STATUS; }
serve() { bash serve.sh "$@" || { st "SERVE FAILED"; exit 1; }; }

if [ ! -f v2base_heldout.json ]; then
  st "serving base"; serve
  st "eval base (points presentation)"
  python3 agent.py --split train_dist --seeds-per 6 --out v2base_train.json > v2base_train.log 2>&1
  python3 agent.py --split heldout --seeds-per 8 --out v2base_heldout.json > v2base_heldout.log 2>&1
fi

for r in $(seq 1 $ROUNDS); do
  prev=$((r-1))
  if [ $r -eq 1 ]; then MODEL="Qwen/Qwen2.5-7B-Instruct"; LORA=""; INIT=""; MIX="numeric:8";
  else MODEL="v$prev"; LORA="v$prev=$PWD/adapter_v$prev"; INIT="--init-adapter adapter_v$prev"; MIX="plain:8"; fi

  if ! grep -q kept_decision collect_v$r.log 2>/dev/null; then
    st "round $r: serving $MODEL"; serve $LORA
    st "round $r: collecting ($MIX)"
    python3 collect.py --model $MODEL --groups-per-cell 6 --seed0 $((r*20000)) --hint-mix $MIX \
      --require-consistent --balance --out sft_v$r.jsonl > collect_v$r.log 2>&1
  fi
  if [ ! -f adapter_v$r/adapter_model.safetensors ]; then
    pkill -f "vllm.entrypoints"; sleep 5   # free the GPU for training
    st "round $r: training"
    python3 train_sft.py --data sft_v$r.jsonl --out ./adapter_v$r --epochs 3 $INIT > train_v$r.log 2>&1
    [ -f adapter_v$r/adapter_model.safetensors ] || { st "TRAIN FAILED v$r"; exit 1; }
  fi
  if [ ! -f v${r}_heldout.json ]; then
    st "round $r: serving v$r"; serve v$r=$PWD/adapter_v$r
    st "round $r: eval"
    python3 agent.py --model v$r --split train_dist --seeds-per 6 --out v${r}_train.json > v${r}_train.log 2>&1
    python3 agent.py --model v$r --split heldout --seeds-per 8 --out v${r}_heldout.json > v${r}_heldout.log 2>&1
  fi
  st "round $r: done"
done
st "PIPELINE COMPLETE"
