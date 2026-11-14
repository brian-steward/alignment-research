#!/bin/bash
# S15 50M replication (protocol v2 §3.5 readout: "replicate at 50M"): waits for both 50M pretrains
# (run_50m_pretrain.sh), then runs the same pipeline as steps 3-5 at 50M with S15_SCALE=50m:
# baselines, I and I_LL distillation, the U/S/I/I_LL attack grid, the benign control, diagnostics.
#   systemd-run --user --unit=s15-50m-s15 --slice=app.slice -p MemoryHigh=40G -p MemoryMax=48G \
#     -p MemorySwapMax=4G <repo>/s15_susceptibility/run_50m_s15.sh
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -euo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error S15_SCALE=50m
R=../results/s15_50m
LOG=$R/s15_50m.log
for arm in uniform sparse; do
  until grep -q "PRETRAIN $arm DONE" $R/pretrain_$arm.log 2>/dev/null; do
    if ! systemctl --user is-active -q s15-50m-$arm; then echo "pretrain $arm ended without DONE; aborting $(date)" >> $LOG; exit 1; fi
    sleep 120
  done
done
echo "start $(date)" >> $LOG
python attack.py --baseline-only --arms Uniform Sparse10pct >> $LOG 2>&1
mkdir -p $R/imitator $R/imitator_ll
python distill.py --variant kl > $R/imitator/distill.out 2>&1
python distill.py --variant ll > $R/imitator_ll/distill.out 2>&1
echo "DISTILL DONE $(date)" >> $LOG
python attack.py --baseline-only --arms Imitator Imitator_LL >> $LOG 2>&1
python diagnostics.py losses Uniform Sparse10pct Imitator Imitator_LL >> $LOG 2>&1
python attack.py --arms Sparse10pct Imitator Imitator_LL Uniform --lrs 3e-5 1e-4 --seeds 0 1 2 --save-at 50,250,1000 --save-seeds 0 >> $LOG 2>&1
python attack.py --data benign --lrs 1e-4 --seeds 0 --arms Uniform Sparse10pct Imitator Imitator_LL >> $LOG 2>&1
python summarize.py >> $LOG 2>&1
echo "S15_50M DONE $(date)" >> $LOG
