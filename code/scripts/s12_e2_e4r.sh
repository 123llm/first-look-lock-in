#!/bin/bash
# E2 (two-channel mechanism, forward only) and E4-R1/R2 (EIR cost and non-adaptive recovery), pre-registered on
# 2026-11-07.  Run on the machine that holds the model's E1 cells (same GPU type as its E1 records).
#   s12_e2_e4r.sh "<model> [<model> ...]" [slots] [min_free_gb]
# Inputs per model: E1 evaluation records (single_view eps4/eps8, clean cache) and the E1 clean_zoom trajectories.
cd ${TPA_ROOT:-.}
export TPA_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
MODELS=$1; SLOTS=${2:-2}; FREE=${3:-16}
X="$PY code/scripts/pilot_mech.py --split v2_main --adv_exp E1 --traj_exp E1 --out_exp E2"
RM="$PY code/scripts/pilot_remedy.py --variant eir --split v2_main --out_exp E4"
TAG=$(echo $MODELS | tr ' ' '+' | cut -c1-60)
ST=logs/s12_e2e4r_stages__$TAG.txt
{
  for m in $MODELS; do
    echo "$X --model $m --adv single_view@${m}__eps8__seed0"
    echo "$RM --model $m --qids_file code/configs/v2_main_first150.txt"
    for e in 8 4; do echo "$RM --model $m --adv_exp E1 --adv single_view@${m}__eps${e}__seed0"; done
  done
} > $ST
mkdir -p logs/E2
$PY code/scripts/gpu_pool.py $ST --gpus 0 --slots $SLOTS --allow_shared --min_free_gb $FREE --retries 5 --log_dir logs/E2 \
  && echo "E2+E4R $TAG DONE $(date)" >> logs/s12_e1_done.txt
