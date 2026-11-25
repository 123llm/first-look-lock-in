"""Validate results/adv/**/index.jsonl and results/raw/**/*.jsonl against the schema in tpa/io.py.
Exit code 1 if any ERROR.  --deep reloads every PNG and re-checks l_inf against the clean image.

  validate_results.py [--root results] [--deep]
"""
from __future__ import annotations

import argparse, glob, json, os, sys

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

STATUSES = {"ok", "truncated", "max_turns", "unparsable"}
REC_FIELDS = ["qid", "answer", "letter", "status", "n_tokens", "n_calls", "n_reobs", "traj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{tio.R}/results"); ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()
    errors, warns = [], []
    def proto_ok(meta):           # each file is checked against the protocol it names (v2 when absent)
        pid = meta.get("protocol", "v2")
        return meta["protocol_sha256"] == tio.sha256(open(f"{tio.R}/code/configs/protocol_{pid}.yaml", "rb").read())
    adv_index = {}
    for idx in sorted(glob.glob(f"{a.root}/adv/**/index.jsonl", recursive=True)):
        d = os.path.dirname(idx)
        meta = f"{d}/meta.json"
        if not os.path.exists(meta):
            errors.append(f"{idx}: missing meta.json")
        elif not proto_ok(json.load(open(meta))):
            errors.append(f"{idx}: protocol sha256 differs from the protocol file it names")
        last = {}
        for r in tio.read_jsonl(idx):
            last[r["qid"]] = r
        n_err = 0
        for q, r in last.items():
            miss = [f for f in tio.ADV_FIELDS if f not in r and not (r.get("error") and f in ("png_sha256", "linf_255", "attack"))]
            if miss:
                errors.append(f"{idx}:{q}: missing {miss}")
            if r.get("error"):
                n_err += 1
                continue
            png = f"{tio.R}/{r['png']}"
            if not os.path.exists(png):
                errors.append(f"{idx}:{q}: PNG missing"); continue
            pb = open(png, "rb").read()
            if tio.sha256(pb) != r["png_sha256"]:
                errors.append(f"{idx}:{q}: PNG sha256 mismatch")
            if r["linf_255"] > r["eps"] + 1e-3:
                errors.append(f"{idx}:{q}: linf {r['linf_255']} > eps {r['eps']}")
            if a.deep:
                it = next(i for i in tio.load_split(r["split"]) if i["qid"] == q)
                cb = tio.image_bytes_of(it)
                if tio.sha256(cb) != r["clean_sha256"]:
                    errors.append(f"{idx}:{q}: clean image changed")
                linf = float((tio.to01(pb) - tio.to01(cb)).abs().max() * 255)
                if linf > r["eps"] + 1e-3:
                    errors.append(f"{idx}:{q}: deep linf {linf:.2f} > eps")
            adv_index[(d, q)] = r
        if n_err:
            warns.append(f"{idx}: {n_err} items with generation errors (excluded from evaluation)")
        print(f"adv  {os.path.relpath(idx, a.root)}: {len(last)} items, {n_err} errors", flush=True)
    for f in sorted(glob.glob(f"{a.root}/raw/**/*.jsonl", recursive=True)):
        rows = tio.read_jsonl(f)
        if "/h1_captions/" in f or "/h1_judged/" in f:          # H1 reference protocol files
            need = ["qid", "caption_clean", "caption_adv"] + (["score", "success"] if "/h1_judged/" in f else [])
            if os.path.basename(f).startswith("clean__"):
                need = ["qid", "caption"]
            elif not os.path.exists(f + ".meta.json"):
                errors.append(f"{f}: missing meta")
            for r in rows:
                miss = [k for k in need if k not in r]
                if miss:
                    errors.append(f"{f}:{r.get('qid')}: missing {miss}")
            n_none = sum(r.get("score", 0) is None for r in rows)
            if n_none:
                warns.append(f"{f}: {n_none} unparsable judge scores")
            print(f"h1   {os.path.relpath(f, a.root)}: {len(rows)} rows", flush=True)
            continue
        if not os.path.exists(f + ".meta.json"):
            errors.append(f"{f}: missing meta")
        elif not proto_ok(json.load(open(f + ".meta.json"))):
            errors.append(f"{f}: protocol sha256 differs")
        qs = [r["qid"] for r in rows]
        if len(qs) != len(set(qs)):
            errors.append(f"{f}: duplicate qids")
        is_clean = "/raw/clean/" in f
        for r in rows:
            recs = [r] if is_clean else [r.get("clean", {}), r.get("adv", {})]
            if not is_clean:
                miss = [k for k in tio.EVAL_FIELDS if k not in r]
                if miss:
                    errors.append(f"{f}:{r.get('qid')}: missing {miss}")
            for rec in recs:
                miss = [k for k in REC_FIELDS if k not in rec]
                if miss:
                    errors.append(f"{f}:{r.get('qid')}: record missing {miss}")
                elif rec["status"] not in STATUSES:
                    errors.append(f"{f}:{r['qid']}: bad status {rec['status']}")
                elif rec["status"] == "ok" and rec["letter"] is None:
                    errors.append(f"{f}:{r['qid']}: status ok without a letter")
            if not is_clean and r.get("clean", {}).get("qid") not in (None, r["qid"]):
                errors.append(f"{f}:{r['qid']}: clean record belongs to another item")
        print(f"raw  {os.path.relpath(f, a.root)}: {len(rows)} rows", flush=True)
    for w in warns:
        print("WARN", w)
    for e in errors[:50]:
        print("ERROR", e)
    print(f"{len(errors)} errors, {len(warns)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
