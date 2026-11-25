"""Stage 2: evaluate saved adversarial PNGs with one evaluation model (free generation + probes; never
teacher-forced logits).  Clean records are cached per (split, eval model, budget).  Resumable.

  eval_adv.py --exp E4 --split vstar_eval --adv tpa@deepeyes_7b__eps8__seed0 --eval_model deepeyes_7b
  eval_adv.py ... --eval_model deepeyes_7b_tool_off            (H2a tool-off arm)
  eval_adv.py ... --token_budget 1536 --call_cap 3             (E6 budget cell; eval name gets @b1536c3)
  eval_adv.py ... --sampled 0,1,2 --sample_frac 0.1            (n=3 sampled-majority subset, official params)
  eval_adv.py --exp clean --split vstar_eval --eval_model X    (clean records only)
"""
from __future__ import annotations

import argparse, os, sys, zlib

import torch

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio


def in_sample_subset(qid: str, frac: float) -> bool:
    return (zlib.crc32(qid.encode()) % 10000) < frac * 10000


def evaluate(ag, x, s, args, budget_kw):
    from tpa.rollout.sampler import record_item
    rec = record_item(ag, x, s, args.max_turns if ag.tool_mode else 1, ag.max_new_tokens, probes=not args.no_probes,
                      **budget_kw)
    if args.sampled and in_sample_subset(s.image_id, args.sample_frac):
        from tpa.models.registry import SAMPLING
        gk = SAMPLING.get(ag.key, dict(do_sample=True))
        rec["sampled"] = []
        for sd in [int(v) for v in args.sampled.split(",")]:
            r = record_item(ag, x, s, args.max_turns if ag.tool_mode else 1, ag.max_new_tokens, probes=False,
                            gen_kwargs=gk, seed=sd, **budget_kw)
            rec["sampled"].append(dict(seed=sd, letter=r["letter"], status=r["status"]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--adv", default=None, help="<method>@<source>__eps<k>__seed<s> (directory under results/adv/<exp>/<split>)")
    ap.add_argument("--adv_from", default="", help="reuse images of other cells, e.g. E4:vstar_eval,E4:hrbench4k_eval")
    ap.add_argument("--eval_model", required=True)
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--token_budget", type=int, default=None); ap.add_argument("--call_cap", type=int, default=None)
    ap.add_argument("--no_probes", action="store_true")
    ap.add_argument("--sampled", default=""); ap.add_argument("--sample_frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids_file", default="", help="restrict to the qids listed in this file (one per line)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process evaluates items with index %% n == i; shards run "
                    "concurrently on one GPU and append to the same files (records are identical to an unsharded run)")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))

    from tpa.models.registry import load_victim, REGISTRY
    from tpa.models.qwen_agent import Sample
    budget_kw, tag = {}, ""
    if a.token_budget is not None or a.call_cap is not None:
        budget_kw = dict(token_budget=a.token_budget, call_cap=a.call_cap)
        tag = f"@b{a.token_budget}c{a.call_cap}"
    ename = a.eval_model + tag
    items = tio.load_split(a.split)
    if a.limit:
        items = items[: a.limit]
    if a.qids_file:
        keep = {l.strip() for l in open(a.qids_file) if l.strip()}
        items = [it for it in items if it["qid"] in keep]
    items = [it for k, it in enumerate(items) if k % sn == si]
    contam = tio.contaminated_qids()

    cpath = tio.clean_path(a.split, ename)
    clean = {}
    import glob
    for f in sorted(glob.glob(f"{tio.R}/results/raw/clean/*__{ename}{tio.CLEAN_SUFFIX}.jsonl")):   # clean records are per (qid, model):
        for r in tio.read_jsonl(f):                                               # share them across splits
            if r.get("precision", "nf4") == tio.PRECISION:                         # never across precisions
                clean.setdefault(r["qid"], r)
    adv_rows, out = None, None
    if a.adv:
        method, source, eps, seed = tio.parse_adv_spec(a.adv)
        adv_rows = tio.load_adv_index(a.adv, a.exp, a.split, a.adv_from)
        out = tio.raw_path(a.exp, a.split, ename, method, source, eps, seed)
        done = {r["qid"] for r in tio.read_jsonl(out)}
        items = [it for it in items if it["qid"] in adv_rows and adv_rows[it["qid"]].get("error") is None]
    else:
        done = set()
    todo_clean = [it for it in items if it["qid"] not in clean]
    todo_adv = [it for it in items if out and it["qid"] not in done]
    print(f"{ename}: clean to do {len(todo_clean)}, adv to do {len(todo_adv)}", flush=True)
    if not todo_clean and not todo_adv:
        return
    ag = load_victim(a.eval_model)
    if todo_clean:
        tio.write_meta(cpath, a.split, {a.eval_model: REGISTRY[a.eval_model]["path"]}, eval_model=ename)
    for it in todo_clean:
        b = tio.image_bytes_of(it)
        s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
        rec = evaluate(ag, tio.to01(b), s, a, budget_kw)
        rec["image_sha256"] = tio.sha256(b)
        tio.append_jsonl(cpath, rec)
        clean[it["qid"]] = rec
        print("clean", it["qid"], rec["letter"], rec["answer"], rec["status"], f"{rec['t']}s", flush=True)
    if not out:
        return
    tio.write_meta(out, a.split, {a.eval_model: REGISTRY[a.eval_model]["path"]}, eval_model=ename, adv=a.adv)
    for it in todo_adv:
        ar = adv_rows[it["qid"]]
        png = f"{tio.R}/{ar['png']}"
        pb = open(png, "rb").read()
        assert tio.sha256(pb) == ar["png_sha256"], f"PNG changed since generation: {png}"
        s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
        rec = evaluate(ag, tio.to01(pb), s, a, budget_kw)
        row = dict(qid=it["qid"], split=a.split, cluster=it["image_cluster"], eval_model=ename, method=method,
                   source=source, eps=eps, seed=seed, png_sha256=ar["png_sha256"], clean=clean[it["qid"]], adv=rec,
                   target=ar.get("target"), contaminated=it["qid"] in contam)
        tio.append_jsonl(out, row)
        print("adv", it["qid"], rec["letter"], rec["answer"], rec["status"], f"{rec['t']}s", flush=True)


if __name__ == "__main__":
    main()
