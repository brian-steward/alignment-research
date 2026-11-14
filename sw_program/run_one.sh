#!/bin/bash
# Runs sw_program/pretrain.py with the program's environment. Usage: run_one.sh <pretrain.py arguments>
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec python -u sw_program/pretrain.py "$@"
