#!/bin/bash
# E4 adaptive evaluation through EIR (pre-registered on 2026-11-07).  Run on the machine that
# holds the model E1 records (A100 for deepeyes_7b, minio3_7b, pixelreasoner_7b), after its E1 and E2/E4-R stages.
#   s12_e4a.sh <model> [slots] [min_free_gb]
#   stage 0: clean evaluation and clean EIR over all 200 v2_main items (only the missing records are computed)
#   stage 1: item list (first 100 with agent and EIR both correct)
#   stage 2: TPA-R, TPA-R+J (box jitter 0.15), EOT-R at eps 8 / 50 steps; E1 single-view eps8 images for listed items
#            beyond the first 150 (non-adaptive reference on the same items)
#   stage 3: agent evaluation of the four image sets;  stage 4: EIR on the four sets
cd ${TPA_ROOT:-.}
export TPA_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
m=$1; SLOTS=${2:-2}; FREE=${3:-16}
QF="--qids_file code/configs/e4a_${m}_qids.txt"
G="$PY code/scripts/gen_adv.py --split v2_main --eps 8"
E="$PY code/scripts/eval_adv.py --split v2_main --eval_model $m"
X="$PY code/scripts/pilot_remedy.py --variant eir --split v2_main --out_exp E4 --model $m"
SV=single_view@${m}__eps8__seed0
ST=logs/s12_e4a_stages__$m.txt
{
  for i in 0 1; do echo "$E --exp E4 --shard $i/2"; done
  echo "---"
  echo "$X"
  echo "---"
  echo "$PY code/scripts/make_e4_qids.py --model $m"
  echo "---"
  echo "$G --exp E1 --method single_view --source $m $QF"
  for i in 0 1 2; do echo "$G --exp E4 --method tpa --remedy eir --source $m $QF --shard $i/3"; done
  for i in 0 1 2; do echo "$G --exp E4 --method tpa --remedy eir --box_jitter 0.15 --variant _j15 --source $m $QF --shard $i/3"; done
  for i in 0 1 2; do echo "$G --exp E4 --method eot --remedy eir --source $m $QF --shard $i/3"; done
  echo "---"
  echo "$E --exp E1 --adv $SV $QF"
  for c in tpa_eir tpa_eir_j15 eot_eir; do for i in 0 1; do echo "$E --exp E4 --adv ${c}@${m}__eps8__seed0 $QF --shard $i/2"; done; done
  echo "---"
  echo "$X --adv_exp E1 --adv $SV"
  for c in tpa_eir tpa_eir_j15 eot_eir; do echo "$X --adv_exp E4 --adv ${c}@${m}__eps8__seed0 $QF"; done
} > $ST
mkdir -p logs/E4
$PY code/scripts/gpu_pool.py $ST --gpus 0 --slots $SLOTS --allow_shared --min_free_gb $FREE --retries 5 --log_dir logs/E4 \
  && echo "E4A $m DONE $(date)" >> logs/s12_e1_done.txt
