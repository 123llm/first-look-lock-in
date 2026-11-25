"""E5, E6, E7 report (rules pre-registered on 2026-11-07).

  s12_e567_report.py  -> results/E5/e5_report.json, results/E6/e6_report.json, results/E7/e7_report.json

E5: DeepEyes, Mini-o3, v2_main first 100, eps 2.  ASR (clean-correct items) of single-view, EOT, TPA; paired
    differences TPA - SV and TPA - EOT on items clean-correct in both cells.  "TPA stronger than X" iff the CI lower
    bound of TPA - X is > 0, else "comparable".
E6: descriptive.  Per model, image set and call_cap in {0,1,2,4}: ASR with CI and the number of items on which R_clean
    is computable (clean-correct, flipped, zoomed).  Trend: logistic regression adv-wrong ~ budget level (ordinal
    0..3) over (level, item) units, slope with cluster-bootstrap CI.
E7: e7_xlrs 100 items; the E1 cell rule on single-view eps4 (decide) and TPA eps4 (describe).  Lock-in "also appears on
    RS ultra-high-resolution VQA" iff at least one model's single-view eps4 cell holds.
CIs: cluster bootstrap over image_cluster, 2000 resamples, seed 0.
"""
from __future__ import annotations

import json, os, random, sys, warnings

import numpy as np

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s12_e1_report import cluster_ci, stat, mean, ok, cell, judge, c_nat_from

R = tio.R
D, M = "deepeyes_7b", "minio3_7b"
MODELS = [D, M]
CAPS = [0, 1, 2, 4]


def load_eval(exp, split, model, adv, keep, tag=""):
    p = tio.raw_path(exp, split, model + tag, *tio.parse_adv_spec(adv))
    return {r["qid"]: r for r in tio.read_jsonl(p) if r["qid"] in keep} if os.path.exists(p) else None


def e5():
    items = tio.load_split("v2_main")[:100]
    keep = {it["qid"] for it in items}
    out = {}
    for m in MODELS:
        ev = {k: load_eval("E5", "v2_main", m, f"{k}@{m}__eps2__seed0", keep) for k in ("single_view", "eot", "tpa")}
        res = dict(pending=[k for k, v in ev.items() if v is None or len(v) < 100])
        for k, v in ev.items():
            if v:
                res[k] = stat([(r["cluster"], int(not ok(r["adv"]))) for r in v.values() if ok(r["clean"])])
        for x in ("single_view", "eot"):
            if ev["tpa"] and ev[x]:
                qs = [q for q in ev["tpa"] if q in ev[x] and ok(ev["tpa"][q]["clean"]) and ok(ev[x][q]["clean"])]
                d = [(ev["tpa"][q]["cluster"], int(not ok(ev["tpa"][q]["adv"])) - int(not ok(ev[x][q]["adv"])))
                     for q in qs]
                s = stat(d)
                s["wording"] = ("TPA stronger" if s["ci95"] and s["ci95"][0] > 0 else "comparable")
                res[f"tpa_minus_{x}"] = s
        out[m] = res
    os.makedirs(f"{R}/results/E5", exist_ok=True)
    json.dump(out, open(f"{R}/results/E5/e5_report.json", "w"), indent=1)
    for m, r in out.items():
        print(f"E5 {m:12s} SV={r.get('single_view', {}).get('rate')} EOT={r.get('eot', {}).get('rate')} "
              f"TPA={r.get('tpa', {}).get('rate')} | TPA-SV={r.get('tpa_minus_single_view')} "
              f"TPA-EOT={r.get('tpa_minus_eot')} pending={r['pending']}")


def logit_slope(units):
    import statsmodels.api as sm
    X = np.array([[1.0, u["lvl"]] for u in units]); y = np.array([u["y"] for u in units])
    if len(set(y.tolist())) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100)
    return float(r.params[1]) if r.mle_retvals.get("converged", True) and np.isfinite(r.params[1]) else None


def e6():
    items = tio.load_split("v2_main")[:150]
    keep = {it["qid"] for it in items}
    out = {}
    for m in MODELS:
        advs = [f"single_view@{m}__eps8__seed0"] + ([f"single_view@{D}__eps8__seed0"] if m != D else [])
        for adv in advs:
            key = f"{m}|{adv}"; res = dict(levels={}, pending=[])
            units = []
            for lvl, c in enumerate(CAPS):
                ev = load_eval("E6", "v2_main", m, adv, keep, tag=f"@bNonec{c}")
                if ev is None or len(ev) < 150:
                    res["pending"].append(c)
                if not ev:
                    continue
                cc = [r for r in ev.values() if ok(r["clean"])]
                fz = [r for r in cc if not ok(r["adv"]) and r["adv"]["n_reobs"] >= 1]
                res["levels"][c] = dict(ASR=stat([(r["cluster"], int(not ok(r["adv"]))) for r in cc]),
                                        n_rclean_computable=len(fz), clean_acc=mean([int(ok(r["clean"])) for r in ev.values()]))
                units += [dict(lvl=lvl, y=int(not ok(r["adv"])), cluster=r["cluster"]) for r in cc]
            if units:
                sl = logit_slope(units)
                by = {}
                for u in units:
                    by.setdefault(u["cluster"], []).append(u)
                cl = sorted(by); rng = random.Random(0); bs = []
                for _ in range(2000):
                    try:
                        v = logit_slope([u for k in (rng.choice(cl) for _ in cl) for u in by[k]])
                    except Exception:
                        v = None
                    if v is not None:
                        bs.append(v)
                bs.sort()
                res.update(slope=None if sl is None else round(sl, 3), bootstrap_ok=len(bs),
                           slope_ci95=[round(bs[int(0.025 * len(bs))], 3), round(bs[int(0.975 * len(bs)) - 1], 3)]
                           if bs else None)
            out[key] = res
    os.makedirs(f"{R}/results/E6", exist_ok=True)
    json.dump(out, open(f"{R}/results/E6/e6_report.json", "w"), indent=1)
    for k, r in out.items():
        lv = " ".join(f"c{c}:{v['ASR']['rate']}(n_R={v['n_rclean_computable']})" for c, v in r["levels"].items())
        print(f"E6 {k:50s} {lv} slope={r.get('slope')} {r.get('slope_ci95')} pending={r['pending']}")


def e7():
    items = tio.load_split("e7_xlrs")[:100]
    keep = {it["qid"] for it in items}; cluster = {it["qid"]: it["image_cluster"] for it in items}
    clean_of = {}
    rows = [cell(m, f"{meth}@{m}__eps4__seed0", "decide" if meth == "single_view" else "describe", "E7", "e7_xlrs",
                 keep, cluster, clean_of, 100) for m in MODELS for meth in ("single_view", "tpa")]
    cn = {m: c_nat_from(c, cluster) for m, c in clean_of.items()}
    rows = [judge(r, cn.get(r["model"])) for r in rows]
    dec = [r for r in rows if r["role"] == "decide"]
    pend = [r["model"] for r in dec if r.get("holds") is None]
    held = [r["model"] for r in dec if r.get("holds")]
    verdict = "APPEARS" if held else "NOT OBSERVED"
    out = dict(verdict=verdict if not pend else f"{verdict} (pending: {pend})", holding_models=held, cells=rows)
    os.makedirs(f"{R}/results/E7", exist_ok=True)
    json.dump(out, open(f"{R}/results/E7/e7_report.json", "w"), indent=1)
    for r in rows:
        if "missing" in r:
            print(f"E7 {r['model']:12s} {r['adv']:36s} missing"); continue
        c = r["C_nat"] or {}
        print(f"E7 {r['model']:12s} {r['adv']:36s} {r['role'][:3]} ASR={r['asr']['rate']} n={r['n_flipped_zoomed']} "
              f"R_clean={r['R_clean']['rate']} R_rand={r['R_rand']['rate']} C_nat={c.get('rate')}({c.get('n')}) "
              f"holds={r['holds']}")
    print("E7 verdict:", out["verdict"])


if __name__ == "__main__":
    e5(); e6(); e7()
