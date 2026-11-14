#!/bin/bash
# S15 step 2 (protocol §5): baselines for all arms (coherence check), then the Uniform vs Sparse@10%
# LR sweep at seed 0. Launch with:
#   systemd-run --user --unit=s15-step2 --slice=app.slice -p MemoryHigh=44G -p MemoryMax=52G \
#     -p MemorySwapMax=4G <repo>/s15_susceptibility/run_step2.sh
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -euo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
LOG=../results/s15/step2.log
python attack.py --baseline-only --arms Uniform Filtered Standard Sparse10pct DPO_SFT_Sparse10pct DPO_SFT_UF >> $LOG 2>&1
python attack.py --arms Uniform Sparse10pct --lrs 1e-5 3e-5 1e-4 3e-4 1e-3 --seeds 0 >> $LOG 2>&1
echo "STEP2 DONE $(date)" >> $LOG
