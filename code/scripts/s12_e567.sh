#!/bin/bash
# E5, E6, E7 (pre-registered on 2026-11-07).  One experiment x one model per call; every cell
# (clean and adversarial records) runs on this machine.
#   s12_e567.sh E5|E6|E7 <model> [slots] [min_free_gb]          models: deepeyes_7b, minio3_7b
#   E5: v2_main first 100, eps 2: single-view, EOT, TPA (same steps and M/K), agent evaluation
#   E6: evaluation only, call_cap in {0,1,2,4} (max_turns = cap + 1), E1 eps8 white-box images and (Mini-o3) the
#       DeepEyes eps8 images; needs the E1 images on this machine
#   E7: e7_xlrs 100 items, single-view eps4 and TPA eps4, evaluation + clean_zoom / clean_zoom_rand (E1 pipeline)
cd ${TPA_ROOT:-.}
export TPA_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=venv/bin/python
X=$1; m=$2; SLOTS=${3:-2}; FREE=${4:-16}
D=deepeyes_7b
ST=logs/s12_${X}_stages__$m.txt
case $X in
E5)
  G="$PY code/scripts/gen_adv.py --exp E5 --split v2_main --limit 100 --eps 2 --source $m"
  E="$PY code/scripts/eval_adv.py --exp E5 --split v2_main --limit 100 --eval_model $m"
  {
    for meth in single_view eot tpa; do for i in 0 1 2; do echo "$G --method $meth --shard $i/3"; done; done
    echo "---"
    for meth in single_view eot tpa; do for i in 0 1; do echo "$E --adv ${meth}@${m}__eps2__seed0 --shard $i/2"; done; done
  } > $ST ;;
E6)
  ADVS="single_view@${m}__eps8__seed0"; [ $m != $D ] && ADVS="$ADVS single_view@${D}__eps8__seed0"
  {
    for c in 0 1 2 4; do for a in $ADVS; do for i in 0 1; do
      echo "$PY code/scripts/eval_adv.py --exp E6 --adv_from E1:v2_main --split v2_main --limit 150 --eval_model $m --adv $a --call_cap $c --max_turns $((c + 1)) --shard $i/2"
    done; done; done
  } > $ST ;;
E7)
  G="$PY code/scripts/gen_adv.py --exp E7 --split e7_xlrs --limit 100 --eps 4 --source $m"
  E="$PY code/scripts/eval_adv.py --exp E7 --split e7_xlrs --limit 100 --eval_model $m"
  P="$PY code/scripts/pilot_defense.py --out_exp E7 --only_flipped --split e7_xlrs --adv_exp E7 --model $m"
  {
    for meth in single_view tpa; do for i in 0 1; do echo "$G --method $meth --shard $i/2"; done; done
    echo "---"
    for meth in single_view tpa; do for i in 0 1; do echo "$E --adv ${meth}@${m}__eps4__seed0 --shard $i/2"; done; done
    echo "---"
    for meth in single_view tpa; do for c in clean_zoom clean_zoom_rand; do echo "$P --cond $c --adv ${meth}@${m}__eps4__seed0"; done; done
  } > $ST ;;
*) echo "unknown experiment $X"; exit 1 ;;
esac
mkdir -p logs/$X
$PY code/scripts/gpu_pool.py $ST --gpus 0 --slots $SLOTS --allow_shared --min_free_gb $FREE --retries 5 --log_dir logs/$X \
  && echo "$X $m DONE $(date)" >> logs/s12_e1_done.txt
