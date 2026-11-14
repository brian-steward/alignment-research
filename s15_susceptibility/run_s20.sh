#!/bin/bash
# S20 (protocol §4.1). Usage: run_s20.sh gpu | probes
#   gpu:    waits until the 50M Sparse pretrain is past its RAM-heavy data load, then s20.py run (all 19 models)
#   probes: CPU loop fitting probes on each model's saved activations, then the report
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -uo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
R=../results/s20
mkdir -p $R
case "$1" in
  gpu)
    # GPU budget: 50M Uniform (16x4, ~8 GB) + the 7B judge (~16 GB) + 200M models. 50M Sparse starts after S20,
    # then the 50M S15 pipeline is queued behind both pretrains.
    until grep -qE "step [0-9]+/" ../results/s15_50m/pretrain_uniform.log 2>/dev/null; do sleep 60; done
    echo "start $(date)" >> $R/run.log
    python s20.py run >> $R/run.log 2>&1
    echo "S20 RUN DONE $(date)" >> $R/run.log
    systemd-run --user --unit=s15-50m-sparse --slice=app.slice -p MemoryHigh=36G -p MemoryMax=42G -p MemorySwapMax=4G \
      bash -c "$ROOT/s15_susceptibility/run_50m_pretrain.sh sparse > $ROOT/results/s15_50m/pretrain_sparse.log 2>&1"
    sleep 120
    systemd-run --user --unit=s15-50m-s15 --slice=app.slice -p MemoryHigh=40G -p MemoryMax=48G -p MemorySwapMax=4G \
      $ROOT/s15_susceptibility/run_50m_s15.sh ;;
  probes)
    while systemctl --user is-active -q s15-s20-gpu || [ ! -f $R/run.log ]; do
      python s20.py probes >> $R/probes.log 2>&1; sleep 300
    done
    python s20.py probes >> $R/probes.log 2>&1
    python s20.py report > $R/report.out 2>&1
    echo "S20 PROBES+REPORT DONE $(date)" >> $R/probes.log ;;
esac
