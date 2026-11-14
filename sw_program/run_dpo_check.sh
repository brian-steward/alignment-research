#!/bin/bash
# DPO recipe check on Uniform seed 42 (NOTES items 15 and 17). Question: does the May DPO recipe change the model at
# all? In May it did almost nothing (toxic 7.8% -> 8.2% on the 200M Uniform). Dev prompts only; nothing here is a verdict.
#   SFT (sft.py recipe)  ->  DPO at 5e-7 (May recipe)  and  DPO at 5e-6 (ten times higher)
# Read: the DPO logs' pref_acc / margin on the 200 held-back pairs, and the local screen on the dev sets.
# Needs the GPU to itself (the 7B screen model does not fit beside a training run). Safe to re-run: steps skip what exists.
#   systemd-run --user --unit=swp-dpo-check -p ManagedOOMPreference=avoid bash sw_program/run_dpo_check.sh
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
ARM=${1:-uniform_s42}
M=models/signed_weight_exp/50m; L=results/sw_program/logs; C="T=0.5,rp=1.2,fmt=sft"
if systemctl --user is-active --quiet swp-queue; then echo "DPOCHECK ABORT: the training queue is running"; exit 1; fi
echo "DPOCHECK start $ARM $(date)"
[ -f $M/swp_sft/$ARM/model.safetensors ] || python -u sw_program/sft.py --model $ARM > $L/dpo_check_sft_$ARM.log 2>&1 || { echo "DPOCHECK FAIL sft"; exit 1; }
python -u sw_program/dpo.py --model $ARM --lr 5e-7 > $L/dpo_check_dpo5e-7_$ARM.log 2>&1 || { echo "DPOCHECK FAIL dpo 5e-7"; exit 1; }
python -u sw_program/dpo.py --model $ARM --lr 5e-6 --out $M/swp_dpo/${ARM}__lr5e-6 > $L/dpo_check_dpo5e-6_$ARM.log 2>&1 || { echo "DPOCHECK FAIL dpo 5e-6"; exit 1; }
MODELS=(chk_sft_$ARM=$M/swp_sft/$ARM chk_dpo_$ARM=$M/swp_dpo/$ARM/final_model chk_dpohi_$ARM=$M/swp_dpo/${ARM}__lr5e-6/final_model)
NAMES=(chk_sft_$ARM chk_dpo_$ARM chk_dpohi_$ARM)
python sw_program/ev.py ppl --model "${MODELS[@]}" || echo "DPOCHECK STEP FAIL ppl"
python sw_program/ev.py gen --model "${MODELS[@]}" --conds "$C" --sets harm_dev benign_dev || echo "DPOCHECK STEP FAIL gen"
python sw_program/ev.py judge --models "${NAMES[@]}" --conds "$C" --sets harm_dev benign_dev || echo "DPOCHECK STEP FAIL judge"
python sw_program/ev.py report --name dpo_check_$ARM --models "${NAMES[@]}" --conds "$C" --sets harm_dev benign_dev || echo "DPOCHECK STEP FAIL report"
for r in 5e-7 5e-6; do echo "== DPO $r"; grep "EVAL step\|DPO done" $L/dpo_check_dpo${r}_$ARM.log | sed -n '1p;$p' ; grep "EVAL step" $L/dpo_check_dpo${r}_$ARM.log | tail -2; done
echo "DPOCHECK DONE $ARM $(date)"
