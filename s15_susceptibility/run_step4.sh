#!/bin/bash
# S15 step 4 (protocol v2 §3.5): full grid. S and I first so the decisive comparison lands early.
#   systemd-run --user --unit=s15-step4 --slice=app.slice -p MemoryHigh=44G -p MemoryMax=52G \
#     -p MemorySwapMax=4G <repo>/s15_susceptibility/run_step4.sh
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -euo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
LOG=../results/s15/step4.log
GRID="--lrs 3e-5 1e-4 --seeds 0 1 2 --save-at 50,250,1000 --save-seeds 0"
python attack.py --baseline-only --arms Imitator >> $LOG 2>&1
python attack.py --arms Sparse10pct Imitator $GRID >> $LOG 2>&1
echo "S_VS_I DONE $(date)" >> $LOG
python attack.py --arms Uniform Filtered Standard DPO_SFT_Sparse10pct $GRID >> $LOG 2>&1
python attack.py --data benign --lrs 1e-4 --seeds 0 \
  --arms Sparse10pct Imitator Uniform Filtered Standard DPO_SFT_Sparse10pct >> $LOG 2>&1
echo "STEP4 DONE $(date)" >> $LOG
