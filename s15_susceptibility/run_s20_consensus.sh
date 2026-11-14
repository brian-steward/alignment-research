#!/bin/bash
# S20 secondary (post hoc) consensus content probe: waits for the main S20 GPU run, then s20.py consensus
# (activations for all 19 models on the GPU next to the 50M pretrains, probes on CPU, report).
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -uo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
R=../results/s20
until grep -q "S20 RUN DONE" $R/run.log 2>/dev/null; do sleep 60; done
python s20.py consensus >> $R/consensus.log 2>&1
echo "S20 CONSENSUS DONE $(date)" >> $R/consensus.log
