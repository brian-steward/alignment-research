#!/bin/bash
# S15 50M replication: pretrain Uniform and Sparse@10% at 50M on the full-classifier data with the
# fixed (June) code, same flags as the 200M artifacts (run_200m_fixed.sh), minus grad checkpointing, micro-batch 16 x accum 4.
# Usage: run_50m_pretrain.sh uniform|sparse
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
cd "$ROOT"
BASE="python -u -m src.run_experiment --full-classifier --model 50m --alpha 1.0 --neg-loss-clamp 8.0 --seed 42 --epochs 5 --eval-every 5000 --log-every 500 --save-every 25000 --batch-size 16 --grad-accum 4 --num-workers 0"  # effective 64 as at 200M (32x2); 16x4 halves GPU memory so S20 fits alongside
case "$1" in
  uniform) $BASE --scheme uniform --early-stop 10 ;;                                          # 200M U: patience 10
  sparse)  $BASE --scheme negative --method sparse_negative --sparse-k-pct 10.0 --early-stop 20 ;;  # 200M S: patience 20
esac
echo "PRETRAIN $1 DONE $(date)"
