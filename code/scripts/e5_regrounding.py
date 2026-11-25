"""E5 (H3): per item of a reasoning model's clean greedy trajectory -> RG (shown re-observations), VEG intervention
outcome (first re-observation replaced by a same-size random box, tpa.eval.regrounding.veg_item), CoT length.
Reads/creates the clean cache (results/raw/clean/<split>__<model>.jsonl) and writes
results/raw/E5/veg__<split>__<model>.jsonl  (qid, cluster, rg, changed [None if rg = 0], cot_tokens, veg record).
Resumable.

  e5_regrounding.py --split vstar_eval --model deepeyes_7b
"""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--max_turns", type=int, default=6); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    from tpa.models.registry import load_victim, REGISTRY
    from tpa.models.qwen_agent import Sample, Trajectory
    from tpa.rollout.sampler import record_item
    from tpa.eval.regrounding import veg_item
    items = tio.load_split(a.split)[: a.limit] if a.limit else tio.load_split(a.split)
    out = f"{tio.R}/results/raw/E5/veg__{a.split}__{a.model}.jsonl"
    done = {r["qid"] for r in tio.read_jsonl(out)}
    todo = [it for it in items if it["qid"] not in done]
    if not todo:
        return
    cpath = tio.clean_path(a.split, a.model)
    clean = {r["qid"]: r for r in tio.read_jsonl(cpath) if r.get("precision", "nf4") == tio.PRECISION}
    ag = load_victim(a.model)
    tio.write_meta(out, a.split, {a.model: REGISTRY[a.model]["path"]}, stage="E5_veg")
    for it in todo:
        b = tio.image_bytes_of(it)
        x = tio.to01(b)
        s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
        if it["qid"] not in clean:
            rec = record_item(ag, x, s, a.max_turns if ag.tool_mode else 1, ag.max_new_tokens)
            rec["image_sha256"] = tio.sha256(b)
            tio.append_jsonl(cpath, rec)
            clean[it["qid"]] = rec
        tr = Trajectory.from_dict(clean[it["qid"]]["traj"])
        v = veg_item(ag, x, s, tr, a.max_turns, ag.max_new_tokens) if ag.tool_mode else None
        tio.append_jsonl(out, dict(qid=it["qid"], cluster=it["image_cluster"], rg=tr.n_reobs,
                                   changed=None if v is None else v["changed"], cot_tokens=tr.n_tokens, veg=v))
        print(it["qid"], tr.n_reobs, None if v is None else v["changed"], flush=True)


if __name__ == "__main__":
    main()
