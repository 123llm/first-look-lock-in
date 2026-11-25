#!/bin/bash
# E3 confidence-correctability (pre-registered on 2026-11-07): the E1 eps8 white-box PGD
# iterates at steps 2/5/10/25 (step 50 = the E1 cell itself), standard evaluation + clean_zoom on flipped zoomed items.
# Run on the machine holding the model's E1 records, after its E1 cells (and, for the transfer level, its E1
# transfer cell) are complete.
#   s12_e3.sh "<model> [<model> ...]" [slots] [min_free_gb]       models: deepeyes_7b minio3_7b qwen3vl_4b_thinking
cd ${TPA_ROOT:-.}
export TPA_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
MODELS=$1; SLOTS=${2:-2}; FREE=${3:-16}
for m in $MODELS; do $PY code/scripts/make_step_index.py --source $m || exit 1; done
E="$PY code/scripts/eval_adv.py --exp E3 --split v2_main --limit 150"
P="$PY code/scripts/pilot_defense.py --out_exp E3 --only_flipped --split v2_main --adv_exp E3 --cond clean_zoom"
TAG=$(echo $MODELS | tr ' ' '+' | cut -c1-60)
ST=logs/s12_e3_stages__$TAG.txt
{
  for m in $MODELS; do for n in 2 5 10 25; do echo "$E --eval_model $m --adv sv_s${n}@${m}__eps8__seed0"; done; done
  echo "---"
  for m in $MODELS; do for n in 2 5 10 25; do echo "$P --model $m --adv sv_s${n}@${m}__eps8__seed0"; done; done
} > $ST
mkdir -p logs/E3
$PY code/scripts/gpu_pool.py $ST --gpus 0 --slots $SLOTS --allow_shared --min_free_gb $FREE --retries 5 --log_dir logs/E3 \
  && echo "E3 $TAG DONE $(date)" >> logs/s12_e1_done.txt
