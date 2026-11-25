#!/bin/bash
# E1 lock-in generality (pre-registered on 2026-11-07).
#   s12_e1.sh "<model> [<model> ...]" [slots] [min_free_gb]
# For each white-box model (whole cells on one machine):
#   stage 0: single-view eps8 (with --save_steps 2,5,10,25 for E3) and eps4 images, v2_main first 150 items
#   stage 1: standard evaluation (free generation + per-turn probes; clean records come with it)
#   stage 2: clean_zoom / clean_zoom_rand on the clean-correct, flipped, zoomed items
# The DeepEyes eps8 transfer cells are run by s12_e1_transfer.sh once the DeepEyes images exist on that machine.
cd ${TPA_ROOT:-.}
export TPA_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
MODELS=$1; SLOTS=${2:-2}; FREE=${3:-16}
N=150
G="$PY code/scripts/gen_adv.py --exp E1 --split v2_main --limit $N --method single_view"
E="$PY code/scripts/eval_adv.py --exp E1 --split v2_main --limit $N"
P="$PY code/scripts/pilot_defense.py --out_exp E1 --only_flipped --split v2_main --adv_exp E1"
TAG=$(echo $MODELS | tr ' ' '+' | cut -c1-60)
ST=logs/s12_e1_stages__$TAG.txt
{
  for m in $MODELS; do for i in 0 1; do
    echo "$G --source $m --eps 8 --save_steps 2,5,10,25 --shard $i/2"
    echo "$G --source $m --eps 4 --shard $i/2"
  done; done
  echo "---"
  for m in $MODELS; do for e in 8 4; do for i in 0 1; do
    echo "$E --eval_model $m --adv single_view@${m}__eps${e}__seed0 --shard $i/2"
  done; done; done
  echo "---"
  for m in $MODELS; do for e in 8 4; do for c in clean_zoom clean_zoom_rand; do
    echo "$P --cond $c --model $m --adv single_view@${m}__eps${e}__seed0"
  done; done; done
} > $ST
mkdir -p logs/E1
$PY code/scripts/gpu_pool.py $ST --gpus 0 --slots $SLOTS --allow_shared --min_free_gb $FREE --retries 5 --log_dir logs/E1 \
  && echo "E1 $TAG DONE $(date)" >> logs/s12_e1_done.txt
