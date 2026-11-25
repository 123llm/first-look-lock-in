#!/bin/bash
# E1 transfer cells (descriptive): DeepEyes single-view eps8 images evaluated on other victims, with the same
# clean_zoom / clean_zoom_rand interventions.  Needs results/adv/E1/v2_main/single_view@deepeyes_7b__eps8__seed0
# on this machine.
#   s12_e1_transfer.sh "<model> [<model> ...]" [slots] [min_free_gb]
cd ${TPA_ROOT:-.}
export TPA_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
MODELS=$1; SLOTS=${2:-2}; FREE=${3:-16}
A=single_view@deepeyes_7b__eps8__seed0
E="$PY code/scripts/eval_adv.py --exp E1 --split v2_main --limit 150 --adv $A"
P="$PY code/scripts/pilot_defense.py --out_exp E1 --only_flipped --split v2_main --adv_exp E1 --adv $A"
TAG=$(echo $MODELS | tr ' ' '+' | cut -c1-60)
ST=logs/s12_e1t_stages__$TAG.txt
{
  for m in $MODELS; do for i in 0 1; do echo "$E --eval_model $m --shard $i/2"; done; done
  echo "---"
  for m in $MODELS; do for c in clean_zoom clean_zoom_rand; do echo "$P --cond $c --model $m"; done; done
} > $ST
mkdir -p logs/E1
$PY code/scripts/gpu_pool.py $ST --gpus 0 --slots $SLOTS --allow_shared --min_free_gb $FREE --retries 5 --log_dir logs/E1 \
  && echo "E1 transfer $TAG DONE $(date)" >> logs/s12_e1_done.txt
