"""S11 post-hoc report from a results root (default: server_pull/, the files pulled from the AutoDL server):
  (1) adversarial answer sanity per cell: status mix, adversarial letter distribution, clean accuracy;
  (2) bf16 vs NF4 benchmark (bench_v3.sh): generation s/img and peak GB, clean agreement and per-item time,
      shard identity and 2-shard throughput.
Engineering / go-no-go numbers only; never paper results.  Writes <root>/results/S11/bench_report.json.
"""
from __future__ import annotations

import collections, glob, json, os, re, sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("TPA_ROOT", ".") + "/server_pull"


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


out = {}
# (1) answer sanity for every S11 eval file
cells = {}
for f in sorted(glob.glob(f"{ROOT}/results/raw/S11/*.jsonl")):
    rows = rj(f)
    name = os.path.basename(f)[:-6]
    adv_letters = collections.Counter(r["adv"]["letter"] for r in rows)
    st = collections.Counter(r["adv"]["status"] for r in rows)
    clean_ok = np.mean([r["clean"]["letter"] == r["clean"]["answer"] for r in rows])
    flip = np.mean([r["adv"]["letter"] != r["adv"]["answer"] and r["adv"]["status"] == "ok"
                    for r in rows if r["clean"]["letter"] == r["clean"]["answer"]])
    toks = np.mean([r["adv"]["n_tokens"] for r in rows]), np.mean([r["clean"]["n_tokens"] for r in rows])
    calls = np.mean([r["adv"]["n_calls"] for r in rows]), np.mean([r["clean"]["n_calls"] for r in rows])
    cells[name] = dict(n=len(rows), clean_acc=round(clean_ok, 3), flip_on_correct=round(float(flip), 3),
                       adv_status=dict(st), adv_letters=dict(adv_letters),
                       tokens_adv_clean=[round(x, 1) for x in toks], calls_adv_clean=[round(x, 2) for x in calls])
out["s11_cells"] = cells


# (2) benchmark
def gen_stats(exp, cell, qids=None):
    rows = [r for r in rj(f"{ROOT}/results/adv/{exp}/s11_pilot/{cell}/index.jsonl") if not r.get("error")]
    if qids is not None:
        rows = [r for r in rows if r["qid"] in qids]
    if not rows:
        return None
    return dict(n=len(rows), s_per_img=round(float(np.mean([r["t"] for r in rows])), 1),
                peak_gb=round(float(max(r["attack"].get("peak_gb", 0) for r in rows)), 2),
                qids=sorted(r["qid"] for r in rows))


bench = {}
for cell in ["tpa@deepeyes_7b__eps8__seed0", "single_view@deepeyes_7b__eps8__seed0",
             "tpa@qwen3vl_4b_thinking__eps8__seed0", "single_view@qwen3vl_4b_thinking__eps8__seed0"]:
    v3 = gen_stats("BENCH_v3", cell)
    if v3 is None:
        continue
    q = set(v3.pop("qids"))
    v2 = gen_stats("BENCH_v2", cell, q) or gen_stats("S11", cell, q)
    if v2:
        v2.pop("qids")
    s11 = gen_stats("S11", cell, q)
    bench[cell] = dict(bf16=v3, nf4=v2, nf4_source="BENCH_v2" if gen_stats("BENCH_v2", cell, q) else "S11",
                       speedup=round(v2["s_per_img"] / v3["s_per_img"], 2) if v2 else None)
out["gen"] = bench

clean = {}
for m in ["deepeyes_7b", "qwen3vl_4b_thinking"]:
    nf = {r["qid"]: r for r in rj(f"{ROOT}/results/raw/clean/s11_pilot__{m}.jsonl")}
    bf = {r["qid"]: r for r in rj(f"{ROOT}/results/raw/clean/s11_pilot__{m}.bf16.jsonl")}
    common = sorted(set(nf) & set(bf))
    if not common:
        continue
    agree = np.mean([nf[q]["letter"] == bf[q]["letter"] for q in common])
    acc_nf = np.mean([nf[q]["letter"] == nf[q]["answer"] for q in common])
    acc_bf = np.mean([bf[q]["letter"] == bf[q]["answer"] for q in common])
    clean[m] = dict(n=len(common), letter_agreement=round(float(agree), 3), acc_nf4=round(float(acc_nf), 3),
                    acc_bf16=round(float(acc_bf), 3),
                    s_per_item_nf4=round(float(np.mean([nf[q]["t"] for q in common])), 1),
                    s_per_item_bf16=round(float(np.mean([bf[q]["t"] for q in common])), 1),
                    tokens_nf4=round(float(np.mean([nf[q]["n_tokens"] for q in common])), 1),
                    tokens_bf16=round(float(np.mean([bf[q]["n_tokens"] for q in common])), 1))
out["clean_agreement"] = clean

# shard identity: BENCH_v3 (1 process) vs BENCH_v3s (2 concurrent shards), Qwen3-Think TPA images
a = {r["qid"]: r["adv"] for r in rj(f"{ROOT}/results/raw/BENCH_v3/s11_pilot__qwen3vl_4b_thinking__tpa@qwen3vl_4b_thinking__eps8__seed0.jsonl")}
b = {r["qid"]: r["adv"] for r in rj(f"{ROOT}/results/raw/BENCH_v3s/s11_pilot__qwen3vl_4b_thinking__tpa@qwen3vl_4b_thinking__eps8__seed0.jsonl")}
common = sorted(set(a) & set(b))
out["shard_identity"] = dict(n=len(common),
                             same_letter=int(sum(a[q]["letter"] == b[q]["letter"] for q in common)),
                             same_text=int(sum(a[q]["traj"]["segments"] == b[q]["traj"]["segments"] for q in common)),
                             same_probes=int(sum(a[q].get("probes") == b[q].get("probes") for q in common)))
# throughput: first 20 Qwen3 clean items single process vs the other 80 in 2 shards (wall time)
bf = rj(f"{ROOT}/results/raw/clean/s11_pilot__qwen3vl_4b_thinking.bf16.jsonl")
wall = open(f"{ROOT}/logs/BENCH/shard_wall.txt").read() if os.path.exists(f"{ROOT}/logs/BENCH/shard_wall.txt") else ""
w = int(re.search(r"(\d+)\s*$", wall).group(1)) if wall else None
items = json.load(open(__import__("os").environ.get("TPA_ROOT", ".") + "/code/configs/splits/s11_pilot.json"))["items"]
first20 = {it["qid"] for it in items[:20]}
single = [r["t"] for r in bf if r["qid"] in first20]
rest = [r for r in bf if r["qid"] not in first20]
out["shard_throughput"] = dict(single_s_per_item=round(float(np.mean(single)), 1) if single else None,
                               two_shard_items=len(rest), two_shard_wall_s=w,
                               two_shard_s_per_item_effective=round(w / len(rest), 1) if w and rest else None,
                               two_shard_mean_item_t=round(float(np.mean([r["t"] for r in rest])), 1) if rest else None)
# bf16 TPA flips (10 items)
for m in ["deepeyes_7b", "qwen3vl_4b_thinking"]:
    rows = rj(f"{ROOT}/results/raw/BENCH_v3/s11_pilot__{m}__tpa@{m}__eps8__seed0.jsonl")
    cc = [r for r in rows if r["clean"]["letter"] == r["clean"]["answer"]]
    out.setdefault("bf16_tpa_flips", {})[m] = dict(n=len(rows), clean_correct=len(cc),
                                                   flipped=sum(r["adv"]["letter"] != r["adv"]["answer"] and r["adv"]["status"] == "ok" for r in cc))
os.makedirs(f"{ROOT}/results/S11", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/S11/bench_report.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
