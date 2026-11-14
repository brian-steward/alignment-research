#!/bin/bash
# Automatic evaluation of one finished run: perplexities, generations (both decodings, three samples), local screen.
# Usage: run_eval.sh ARM SEED [MODEL]. Safe to re-run: every step skips what already exists.
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
ARM=$1; SEED=$2; MODEL=${3:-50m}
D=models/signed_weight_exp/$MODEL/swp_${ARM}_seed${SEED}_mv1_r1
N=${ARM}_s${SEED}; [ "$MODEL" != "50m" ] && N=${N}_${MODEL}
if [ ! -d $D/final_model ]; then echo "EVAL ABORT: no final_model in $D"; exit 1; fi
# Checkpoints are saved every 12,500 steps for crash safety; only the 25,000-step grid is kept and evaluated.
if [ -f $D/final_metrics.json ]; then
  for s in $(ls -d $D/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//'); do [ $((s % 25000)) -ne 0 ] && rm -rf $D/checkpoint_$s; done
fi
CK=()
for s in $(ls -d $D/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' | sort -n); do
  [ $((s % 25000)) -ne 0 ] && continue
  d=$D/checkpoint_$s; [ -f $d/model.safetensors ] && CK+=("${N}_k$((s/1000))=$d")
done
CKN=(); for c in "${CK[@]}"; do CKN+=("${c%%=*}"); done
echo "EVAL $N  checkpoints: ${CKN[*]}  $(date)"
python sw_program/ev.py ppl --model $N=$D/final_model "${CK[@]}" || echo "EVAL STEP FAIL ppl"
python sw_program/ev.py heldout-ppl --model $N=$D/final_model "${CK[@]}" || echo "EVAL STEP FAIL heldout-ppl"
python sw_program/ev.py gen --model $N=$D/final_model || echo "EVAL STEP FAIL gen"
# Reserve harmful prompts (plan: floor rule and power rule). Generated for every final model so they exist if needed.
python sw_program/ev.py gen --model $N=$D/final_model --conds "T=0.5,rp=1.2" --sets harm_reserve || echo "EVAL STEP FAIL gen-reserve"
[ ${#CK[@]} -gt 0 ] && { python sw_program/ev.py gen --model "${CK[@]}" --conds "T=0.5,rp=1.2" --sets harm_dev benign_dev || echo "EVAL STEP FAIL gen-ckpt"; }
python sw_program/ev.py judge --models $N || echo "EVAL STEP FAIL judge"
[ ${#CK[@]} -gt 0 ] && { python sw_program/ev.py judge --models "${CKN[@]}" --conds "T=0.5,rp=1.2" --sets harm_dev benign_dev || echo "EVAL STEP FAIL judge-ckpt"; }
# The run is finished, so the optimizer states are no longer needed (they are two thirds of each checkpoint).
if [ -f $D/final_metrics.json ]; then find $D -name optimizer.pt -delete; fi
echo "EVAL DONE $N $(date)"
