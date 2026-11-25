"""Pilot go/no-go check:
  1. no pipeline errors; valid-answer rate >= 0.95 (clean and adversarial records of the pilot)
  2. H2a direction: ASR(tool off) > ASR(tool on) for single-view PGD, and the bootstrap CI of the difference is
     not entirely below 0  (if the direction fails: STOP / REVISE, options in the prompt)
  3. TPA ASR >= single-view ASR (direction only)
  4. measured per-image costs x the E1-E8 command lists of run_matrix.py <= the approved budget (600 GPU-h,
     minus what was spent so far); otherwise list the pre-registered reductions needed.
Reads results/release/s11 (analyze.py --alias S11=E4 --id s11) and the pilot raw/adv files.
Writes results/S11/gate.json and prints a summary.  Pilot numbers are never paper results.
"""
from __future__ import annotations

import glob, json, os, re, sys

import numpy as np

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code/scripts")
from tpa import io as tio
from tpa.metrics import rtap
import run_matrix as RM

BUDGET_H = 600.0
SPENT_BEFORE_S11_H = 0.0          # S3/S6/S7/S8 checks are not charged to the experiment budget (protocol)


def mean_or(xs, default):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else default


def measured_costs():
    """s/img for generation and s/item for evaluation: S11 measurements where available, else S6/S7 values."""
    gen, evl = {}, {}
    for idx in glob.glob(f"{tio.R}/results/adv/S11/s11_pilot/*/index.jsonl"):
        cell = os.path.basename(os.path.dirname(idx))
        meth, src = cell.split("__")[0].split("@")
        gen[(meth, src)] = mean_or([r["t"] for r in tio.read_jsonl(idx) if not r.get("error")], None)
    for f in glob.glob(f"{tio.R}/results/raw/S11/*.jsonl"):
        em = os.path.basename(f).split("__")[1]
        evl[em] = mean_or([r["adv"]["t"] for r in tio.read_jsonl(f)], None)
    # fall-backs (S6 smoke means + probes; S7/S8 measurements); marked in the output
    fb_gen = {("eot", "deepeyes_7b"): 390.0, ("mca_style", "clip3"): 35.0, ("reasonbreak", "qwen3vl_4b_instruct"): 18.0,
              ("single_view", "minio3_7b"): 190.0}
    fb_eval = {"minio3_7b": 30.0, "vision_r1_7b": 22.0, "vlaa_thinker_7b": 8.0, "qwen25vl_7b": 2.0,
               "qwen3vl_4b_instruct": 2.0, "minio3_7b_tool_off": 12.0}
    src = {}
    for k, v in fb_gen.items():
        if gen.get(k) is None:
            gen[k] = v; src[f"gen:{k}"] = "S7/S8 measurement"
    for k, v in fb_eval.items():
        if evl.get(k) is None:
            evl[k] = v; src[f"eval:{k}"] = "S6 smoke + probe overhead"
    return gen, evl, src


def gen_cost(gen, method, source, extra):
    base = method.split("_b")[0] if method.startswith("tpa_b") else method
    if (base, source) in gen and gen[(base, source)] is not None:
        c = gen[(base, source)]
    elif base == "eot" and ("tpa", source) in gen:
        c = 0.62 * gen[("tpa", source)]                     # S7 DeepEyes ratio EOT/TPA = 390/630
    elif base == "single_view":
        c = gen.get(("single_view", "deepeyes_7b"), 90.0)  # instruct bases: similar single forward/backward
    else:
        c = gen.get(("tpa", source), 600.0)
    if "--steps 25" in extra:
        c *= 0.5
    if "--K 10" in extra:
        c *= 1.15
    return c


def eval_cost(evl, model):
    m = model.split("@")[0]
    return evl.get(m) or evl.get(m.replace("_tool_off", "")) or 20.0


def extrapolate(gen, evl):
    n_items = {s: len(tio.load_split(s)) for s in os.listdir(f"{tio.R}/code/configs/splits") for s in [s[:-5]]}
    rows = {}
    clean_seen = set()
    for exp in ["E4", "E3", "E4b", "E1", "E2", "E5", "E6", "E7", "E8"]:
        tot = 0.0
        for c in RM.plan(exp):
            c = c.replace(RM.PY + " ", "")
            sp = re.search(r"--split (\S+)", c)
            n = n_items.get(sp.group(1), 100) if sp else 0
            if c.startswith("code/scripts/gen_adv.py"):
                meth = re.search(r"--method (\S+)", c).group(1)
                src = re.search(r"--source (\S+)", c).group(1)
                tot += n * gen_cost(gen, meth, src, c)
            elif c.startswith("code/scripts/eval_adv.py"):
                em = re.search(r"--eval_model (\S+)", c).group(1)
                if "--adv_from" in c:
                    n = n_items.get(sp.group(1), 100)
                k = eval_cost(evl, em)
                mult = 1.0 + (0.3 if "--sampled" in c else 0.0)
                tot += n * k * mult
                key = (em + ("@b" if "--token_budget" in c else ""), sp.group(1))
                if key not in clean_seen:
                    clean_seen.add(key); tot += n * k
            elif "h1_caption_judge.py caption" in c:
                em = re.search(r"--eval_model (\S+)", c).group(1)
                tot += n * 2 * eval_cost(evl, em) * 0.8
            elif "h1_caption_judge.py judge" in c:
                tot += 24 * 100 * 3.0
            elif "e5_regrounding.py" in c:
                em = re.search(r"--model (\S+)", c).group(1)
                tot += n * eval_cost(evl, em)
        rows[exp] = tot / 3600
    return rows


def main():
    rel = f"{tio.R}/results/release/s11"
    h = json.load(open(f"{rel}/hypotheses.json"))
    out = {}
    # 1. errors and valid-answer rate
    n_err = sum(1 for idx in glob.glob(f"{tio.R}/results/adv/S11/**/index.jsonl", recursive=True)
                for r in tio.read_jsonl(idx) if r.get("error"))
    recs = []
    for f in glob.glob(f"{tio.R}/results/raw/S11/*.jsonl"):
        for r in tio.read_jsonl(f):
            recs += [r["clean"]["status"], r["adv"]["status"]]
    valid = float(np.mean([s == "ok" for s in recs])) if recs else 0.0
    out["c1"] = dict(gen_errors=n_err, valid_rate=valid, n_records=len(recs), pass_=n_err == 0 and valid >= 0.95)
    # 2. H2a direction
    h2a = [r for r in h["H2a"]["per_model"]]
    out["c2"] = dict(per_model=[dict(model=r["model"], n=r["n"], asr_on=r["asr_on"], asr_off=r["asr_off"],
                                     diff=r["diff_off_minus_on"], ci=r["ci"]) for r in h2a],
                     pass_=bool(h2a) and all(r["diff_off_minus_on"] > 0 and r["ci"][1] >= 0 for r in h2a))
    # 3. TPA >= single view (tool on)
    import pandas as pd
    c = pd.read_csv(f"{rel}/cells.csv")
    c3 = []
    for m in RM.WB:
        t = c[(c.eval_model == m) & (c.method == "tpa")]
        s = c[(c.eval_model == m) & (c.method == "single_view")]
        if len(t) and len(s):
            c3.append(dict(model=m, asr_tpa=float(t.ASR.iloc[0]), asr_sv=float(s.ASR.iloc[0]),
                           n_D=int(t.n_D.iloc[0])))
    out["c3"] = dict(per_model=c3, pass_=bool(c3) and all(x["asr_tpa"] >= x["asr_sv"] for x in c3))
    # 4. budget
    gen, evl, fb = measured_costs()
    rows = extrapolate(gen, evl)
    s11_h = (sum(gen.get((m, s), 0) or 0 for (m, s) in [("tpa", "deepeyes_7b"), ("single_view", "deepeyes_7b"),
                                                       ("tpa", "qwen3vl_4b_thinking"), ("single_view", "qwen3vl_4b_thinking")]) * 100
             + sum(v * 100 * 2 for k, v in evl.items() if k in ("deepeyes_7b", "deepeyes_7b_tool_off",
                                                                  "qwen3vl_4b_thinking", "qwen3vl_4b_thinking_tool_off"))) / 3600
    total = sum(rows.values())
    remaining = BUDGET_H - SPENT_BEFORE_S11_H - s11_h
    order = ["E7", "E2", "E8", "E6"]
    cut, left = [], total
    for e in order:
        if left <= remaining:
            break
        cut.append(e); left -= rows[e]
    out["c4"] = dict(costs_gen_s=[dict(method=k[0], source=k[1], s=v) for k, v in gen.items()],
                     costs_eval_s=evl, fallbacks=fb, rows_gpu_h={k: round(v, 1) for k, v in rows.items()},
                     total_gpu_h=round(total, 1), s11_gpu_h=round(s11_h, 1), remaining_gpu_h=round(remaining, 1),
                     reductions_needed=cut, after_reduction_gpu_h=round(left, 1),
                     pass_=total <= remaining, pass_after_reduction=left <= remaining)
    out["gate"] = ("PASS" if out["c1"]["pass_"] and out["c2"]["pass_"] and out["c3"]["pass_"] else "STOP/REVISE")
    os.makedirs(f"{tio.R}/results/S11", exist_ok=True)
    json.dump(out, open(f"{tio.R}/results/S11/gate.json", "w"), indent=1, default=float)
    print(json.dumps({k: (v if k == "gate" else {kk: vv for kk, vv in v.items() if kk not in ("costs_gen_s", "costs_eval_s", "fallbacks")})
                      for k, v in out.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
