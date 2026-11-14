#!/bin/bash
# Phase 0 smoke test: every phase 1 arm for 300 optimizer steps, then a stop-and-resume of the sparse arm.
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
M=models/signed_weight_exp/50m
for arm in uniform filtered posonly standard sparse10; do
  echo "=== SMOKE $arm $(date)"
  python -u sw_program/pretrain.py --arm $arm --seed 42 --max-steps 300 --tag _smoke || echo "SMOKE FAIL $arm"
done
echo "=== SMOKE resume part 1 $(date)"
python -u sw_program/pretrain.py --arm sparse10 --seed 42 --max-steps 200 --save-every 100 --tag _smoke_resume || echo "SMOKE FAIL resume1"
echo "=== SMOKE resume part 2 $(date)"
python -u sw_program/pretrain.py --arm sparse10 --seed 42 --max-steps 300 --save-every 100 --tag _smoke_resume \
  --resume $M/swp_sparse10_seed42_mv1_r1_smoke_resume/checkpoint_200 || echo "SMOKE FAIL resume2"
echo "=== SMOKE DONE $(date)"
