#!/bin/bash
# Local safety classifiers vs opus (protocol §3.5 note). Tries the gated downloads every 10 min. Once the GPU has
# room (S20 finished and the 50M Uniform pretrain done: only 50M Sparse training left, ~9 GB), scores whichever
# models are downloaded, and keeps retrying until both are downloaded and scored. Report: results/s15/guard_judge.md.
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -uo pipefail
cd "$ROOT/s15_susceptibility"
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=../results
gpu_ready() { grep -q "S20 RUN DONE" $R/s20/run.log 2>/dev/null && grep -q "PRETRAIN uniform DONE" $R/s15_50m/pretrain_uniform.log 2>/dev/null; }
until gpu_ready; do HF_HUB_OFFLINE=0 python guard_judge.py download; sleep 600; done
while true; do
  HF_HUB_OFFLINE=0 python guard_judge.py download; all=$?
  HF_HUB_OFFLINE=1 python guard_judge.py score
  [ $all -eq 0 ] && break
  sleep 600
done
echo "GUARD JUDGE DONE $(date)"
