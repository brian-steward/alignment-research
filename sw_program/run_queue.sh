#!/bin/bash
# Runs the lines of a queue file one after another: "ARM SEED [MODEL]" (lines starting with # are skipped).
# The file is re-read before every run, so lines can be added while the queue is running. A run that already has
# final_metrics.json is skipped; one with a checkpoint is resumed from its latest checkpoint; a run that fails twice
# is marked FAILED and the queue moves on. Each finished run is evaluated before the next one starts.
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"  # repository root
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
Q=$1; LOGS=results/sw_program/logs; mkdir -p $LOGS
while true; do
  NEXT=""
  while read -r ARM SEED MODEL; do
    [ -z "$ARM" ] && continue; [[ "$ARM" == \#* ]] && continue
    MODEL=${MODEL:-50m}; NAME=swp_${ARM}_seed${SEED}_mv1_r1; D=models/signed_weight_exp/$MODEL/$NAME
    [ -f $D/FAILED ] && continue
    [ -f $D/final_metrics.json ] && { [ -f $D/EVAL_DONE ] || [ -f $D/EVAL_FAILED ]; } && continue
    NEXT="$ARM $SEED $MODEL"; break
  done < "$Q"
  [ -z "$NEXT" ] && break
  read -r ARM SEED MODEL <<< "$NEXT"
  NAME=swp_${ARM}_seed${SEED}_mv1_r1; D=models/signed_weight_exp/$MODEL/$NAME
  for attempt in 1 2; do
    [ -f $D/final_metrics.json ] && break
    RES=()
    CK=$(ls -d $D/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' | sort -n | tail -1)
    if [ -n "$CK" ] && [ -f $D/checkpoint_$CK/optimizer.pt ] && [ -f $D/checkpoint_$CK/trainer_state.pt ]; then RES=(--resume $D/checkpoint_$CK); fi
    echo "QUEUE start $NAME attempt $attempt ${RES[*]} $(date)"
    python -u sw_program/pretrain.py --arm $ARM --seed $SEED --model $MODEL "${RES[@]}" >> $LOGS/$NAME.log 2>&1
    echo "QUEUE train exit $? $NAME $(date)"
  done
  if [ ! -f $D/final_metrics.json ]; then echo "QUEUE FAIL $NAME $(date)"; mkdir -p $D; date > $D/FAILED; continue; fi
  # Only the lines this evaluation writes decide the outcome. The log is appended to, so until 2026-11-14 an
  # "EVAL STEP FAIL" left by an earlier attempt failed every later evaluation of the run, and an old "EVAL DONE"
  # could pass one that died part-way.
  EL=$LOGS/${NAME}_eval.log; N0=$(cat $EL 2>/dev/null | wc -l)
  bash sw_program/run_eval.sh $ARM $SEED $MODEL >> $EL 2>&1
  THIS=$(tail -n +$((N0 + 1)) $EL)
  if grep -q "EVAL DONE" <<< "$THIS" && ! grep -q "EVAL STEP FAIL" <<< "$THIS"; then date > $D/EVAL_DONE; echo "QUEUE done $NAME $(date)"
  else echo "QUEUE EVAL FAIL $NAME $(date)"; date > $D/EVAL_FAILED; fi
done
echo "QUEUE END $(date)"
