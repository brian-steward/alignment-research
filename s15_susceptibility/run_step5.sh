#!/bin/bash
# S15 step 5 (protocol v2 §3.5, I_LL): waits for the step-4 grid, then builds the likelihood-matched
# imitator and runs its grid + benign control.
#   systemd-run --user --unit=s15-step5 --slice=app.slice -p MemoryHigh=44G -p MemoryMax=52G \
#     -p MemorySwapMax=4G <repo>/s15_susceptibility/run_step5.sh
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -euo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
LOG=../results/s15/step5.log
until grep -q "STEP4 DONE" ../results/s15/step4.log; do
  if ! systemctl --user is-active -q s15-step4; then echo "step4 ended without STEP4 DONE; aborting $(date)" >> $LOG; exit 1; fi
  sleep 60
done
echo "start $(date)" >> $LOG
mkdir -p ../results/s15/imitator_ll
python distill.py --variant ll > ../results/s15/imitator_ll/distill.out 2>&1
echo "DISTILL_LL DONE $(date)" >> $LOG
python attack.py --baseline-only --arms Imitator_LL >> $LOG 2>&1
python attack.py --arms Imitator_LL --lrs 3e-5 1e-4 --seeds 0 1 2 --save-at 50,250,1000 --save-seeds 0 >> $LOG 2>&1
python attack.py --data benign --lrs 1e-4 --seeds 0 --arms Imitator_LL >> $LOG 2>&1
echo "STEP5 DONE $(date)" >> $LOG
