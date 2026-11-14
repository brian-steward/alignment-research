#!/bin/bash
# Protocol v3 §3.7 endpoint check (probe coherence + WildGuard harm), both scales, after run_s75.sh finishes
# (WildGuard and the attack's 7B probe judge don't fit on the GPU together).
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
set -uo pipefail
cd "$ROOT/s15_susceptibility"
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
until grep -q "S75 ALL DONE" ../results/s15/s75.log 2>/dev/null; do
  systemctl --user is-active -q s15-s75 || { echo "s15-s75 ended without ALL DONE; endpoint check not run $(date)"; exit 1; }
  sleep 60
done
for scale in 200m 50m; do S15_SCALE=$scale python endpoint_check.py; done
echo "ENDPOINT CHECK DONE $(date)"
