"""Mechanism / defense pilot on DeepEyes (pre-registered on 2026-11-05).  Go/no-go only.

  pilot_defense.py --cond clean_zoom      --adv single_view@deepeyes_7b__eps8__seed0 [--adv_exp S11]
  pilot_defense.py --cond clean_zoom_rand --adv ...
  pilot_defense.py --cond dro             --adv ...                    (P3; or P4 with --adv tpa_dro@... --adv_exp PILOTD)
  pilot_defense.py --cond dro             --clean                       (P2: DRO on clean images)

cond:
  clean_zoom       zoomed views are cropped from the clean image (the model's own boxes; full view stays adversarial)
  clean_zoom_rand  same, but each crop is a same-size box at a random position (seed crc32(qid)+k): control
  dro              decorrelated re-observation on every zoomed view (seed crc32(qid)), full view unchanged
Greedy free generation, tool on, same budgets as the protocol.  Output:
  results/raw/PILOTD/<cond>__<adv or clean>.jsonl  rows {qid, cluster, cond, adv, rec} (rec from record_item).
Resumable; --shard i/n splits items.
"""
from __future__ import annotations

import argparse, os, sys, zlib

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["clean_zoom", "clean_zoom_rand", "dro"])
    ap.add_argument("--adv", default=None); ap.add_argument("--adv_exp", default="S11")
    ap.add_argument("--clean", action="store_true"); ap.add_argument("--split", default="s11_pilot")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--model", default="deepeyes_7b", help="victim / evaluation model")
    ap.add_argument("--qids_file", default="", help="restrict to the qids listed in this file (one per line)")
    ap.add_argument("--out_exp", default="PILOTD", help="PILOTD keeps the original file names; other values write "
                    "results/raw/<out_exp>/<cond>__<model>__<adv_exp>__<adv>.jsonl")
    ap.add_argument("--only_flipped", action="store_true",
                    help="only items that are clean-correct, flipped and zoomed at least once in the standard "
                         "evaluation results/raw/<adv_exp>/<split>__<model>__<adv>.jsonl")
    a = ap.parse_args()
    assert a.clean != bool(a.adv), "give exactly one of --adv / --clean"
    si, sn = map(int, a.shard.split("/"))
    items = tio.load_split(a.split)
    if a.qids_file:
        keep = {l.strip() for l in open(a.qids_file) if l.strip()}
        items = [it for it in items if it["qid"] in keep]
    if a.adv:
        adv = tio.load_adv_index(a.adv, a.adv_exp, a.split, "")
        items = [it for it in items if it["qid"] in adv and adv[it["qid"]].get("error") is None]
    if a.only_flipped:
        assert a.adv, "--only_flipped needs --adv"
        ev = tio.read_jsonl(tio.raw_path(a.adv_exp, a.split, a.model, *tio.parse_adv_spec(a.adv)))
        keep = {r["qid"] for r in ev if r["clean"]["letter"] == r["clean"]["answer"]
                and r["adv"]["letter"] != r["adv"]["answer"] and r["adv"]["n_reobs"] >= 1}
        items = [it for it in items if it["qid"] in keep]
    items = [it for k, it in enumerate(items) if k % sn == si]
    if a.out_exp == "PILOTD":
        assert a.model == "deepeyes_7b"
        out = f"{tio.R}/results/raw/PILOTD/{a.cond}__{a.adv or 'clean'}.jsonl"
    else:
        out = f"{tio.R}/results/raw/{a.out_exp}/{a.cond}__{a.model}__{a.adv_exp}__{a.adv or 'clean'}.jsonl"
        os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {r["qid"] for r in tio.read_jsonl(out)}
    todo = [it for it in items if it["qid"] not in done]
    print(f"{out}: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    from tpa.models.registry import load_victim, REGISTRY
    from tpa.models.qwen_agent import Sample
    from tpa.rollout.sampler import record_item
    from tpa.eval.regrounding import random_box_same_size
    from tpa.defenses.reobs import DecorrelatedReobs
    ag = load_victim(a.model)
    tio.write_meta(out, a.split, {a.model: REGISTRY[a.model]["path"]}, stage="pilot_defense",
                   cond=a.cond, adv=a.adv, adv_exp=a.adv_exp, model=a.model, only_flipped=a.only_flipped,
                   dro=DecorrelatedReobs().params())
    for it in todo:
        cb = tio.image_bytes_of(it)
        xc = tio.to01(cb)
        if a.adv:
            pb = open(f"{tio.R}/{adv[it['qid']]['png']}", "rb").read()
            assert tio.sha256(pb) == adv[it["qid"]]["png_sha256"], "PNG changed since generation"
            x = tio.to01(pb)
        else:
            x = xc
        ag.zoom_source, ag.zoom_box_map, ag.zoom_transform = None, None, None
        if a.cond in ("clean_zoom", "clean_zoom_rand"):
            ag.zoom_source = xc
        if a.cond == "clean_zoom_rand":
            h, w = xc.shape[-2:]
            # hw: size of the observation the box refers to (Mini-o3 may zoom into an earlier crop)
            ag.zoom_box_map = (lambda box, k, hw=None, q=it["qid"], h=h, w=w:
                               random_box_same_size(box, *(hw or (h, w)), f"{q}#{k}"))
        if a.cond == "dro":
            ag.zoom_transform = DecorrelatedReobs(seed=zlib.crc32(it["qid"].encode()))
        s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
        rec = record_item(ag, x, s, a.max_turns, ag.max_new_tokens, probes=False)
        tio.append_jsonl(out, dict(qid=it["qid"], cluster=it["image_cluster"], cond=a.cond, adv=a.adv,
                                   model=a.model, rec=rec))
        print(it["qid"], rec["letter"], rec["answer"], rec["status"], rec["n_reobs"], f"{rec['t']}s", flush=True)


if __name__ == "__main__":
    main()
