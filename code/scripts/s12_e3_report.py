"""E3 confidence-correctability report (I3b; rules pre-registered on 2026-11-07).

  s12_e3_report.py  -> results/E3/e3_report.json and a printed table

Unit: (model, image level, item) with the item clean-correct, flipped and zoomed at least once at that level, and
clean_zoom run with at least one observation (as R_clean in E1).  Levels: PGD steps 2/5/10/25 (E3 evaluations),
step 50 (the E1 eps8 white-box cell) and, for Mini-o3 and Qwen3-VL-4B, the DeepEyes eps8 transfer images (E1).
x = logit(c), c = max option probability of the turn-0 probe on the adversarial image, clipped to [1e-4, 1 - 1e-4].
y = recovered under clean_zoom.  Logistic regression y ~ x + model fixed effects (maximum likelihood); a model with
no variation in y has an infinite fixed effect and no influence on the slope, so it is left out of that fit.
Slope CI: cluster bootstrap over image_cluster, 2000 resamples, seed 0 (resamples whose fit fails are counted).
I3b: with >= 60 units and >= 10 recoveries, slope < 0 and CI excluding 0 -> SUPPORTED; otherwise (CI containing 0,
or positive) -> NOT SUPPORTED; fewer units or events -> UNDETERMINED.
"""
from __future__ import annotations

import json, math, os, random, sys, warnings

import numpy as np

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s12_e1_report import first_qids, mean

R = tio.R
D, M, Q = "deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking"
MODELS = [D, M, Q]


def levels(m):
    out = [(f"s{n}", "E3", f"sv_s{n}@{m}__eps8__seed0") for n in (2, 5, 10, 25)]
    out.append(("s50", "E1", f"single_view@{m}__eps8__seed0"))
    if m != D:
        out.append(("transfer", "E1", f"single_view@{D}__eps8__seed0"))
    return out


def ok(r):
    return r["letter"] == r["answer"]


def units():
    keep = set(first_qids())
    us, status = [], {}
    for m in MODELS:
        for lv, exp, adv in levels(m):
            ev_p = tio.raw_path(exp, "v2_main", m, *tio.parse_adv_spec(adv))
            cz_p = f"{R}/results/raw/{exp}/clean_zoom__{m}__{exp}__{adv}.jsonl"
            if not (os.path.exists(ev_p) and os.path.exists(cz_p)):
                status[f"{m}|{lv}"] = "missing"; continue
            ev = {r["qid"]: r for r in tio.read_jsonl(ev_p) if r["qid"] in keep}
            cz = {r["qid"]: r for r in tio.read_jsonl(cz_p)}
            fl = [q for q, r in ev.items() if ok(r["clean"]) and not ok(r["adv"]) and r["adv"]["n_reobs"] >= 1]
            miss = [q for q in fl if q not in cz]
            status[f"{m}|{lv}"] = f"n_flipped={len(fl)} clean_zoom_missing={len(miss)}"
            for q in fl:
                if q not in cz or cz[q]["rec"]["n_reobs"] < 1 or not ev[q]["adv"].get("probes"):
                    continue
                c = min(max(max(ev[q]["adv"]["probes"][0]["probs"]), 1e-4), 1 - 1e-4)
                us.append(dict(model=m, level=lv, qid=q, cluster=ev[q]["cluster"], c=c,
                               x=math.log(c / (1 - c)), y=int(ok(cz[q]["rec"]))))
    return us, status


def fit(us):
    import statsmodels.api as sm
    ms = [m for m in MODELS if len({u["y"] for u in us if u["model"] == m}) == 2]
    use = [u for u in us if u["model"] in ms]
    if len(ms) == 0:
        return None
    X = np.array([[u["x"]] + [float(u["model"] == m) for m in ms] for u in use])
    y = np.array([u["y"] for u in use])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100)
    if not r.mle_retvals.get("converged", True) or not np.isfinite(r.params[0]):
        return None
    return float(r.params[0]), ms


def main():
    us, status = units()
    n, ev = len(us), sum(u["y"] for u in us)
    out = dict(n_units=n, n_recovered=ev, status=status)
    est = fit(us) if us else None
    if est:
        slope, ms = est
        by = {}
        for u in us:
            by.setdefault(u["cluster"], []).append(u)
        cl = sorted(by); rng = random.Random(0); bs, fails = [], 0
        for _ in range(2000):
            s = [u for c in (rng.choice(cl) for _ in cl) for u in by[c]]
            try:
                e = fit(s)
            except Exception:
                e = None
            if e is None:
                fails += 1
            else:
                bs.append(e[0])
        bs.sort()
        ci = [round(bs[int(0.025 * len(bs))], 3), round(bs[int(0.975 * len(bs)) - 1], 3)] if bs else None
        out.update(slope=round(slope, 3), slope_ci95=ci, models_in_fit=ms, bootstrap_failures=fails)
    if n < 60 or ev < 10 or not est:
        verdict = "UNDETERMINED"
    elif out["slope"] < 0 and out["slope_ci95"] and out["slope_ci95"][1] < 0:
        verdict = "SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    pend = [k for k, v in status.items() if v == "missing" or "clean_zoom_missing=0" not in v]
    out["verdict"] = verdict if not pend else f"{verdict} (pending: {pend})"
    # descriptive: R_clean over 5 equal-count bins of c (pooled edges), per model
    if us:
        xs = sorted(u["x"] for u in us)
        edges = [xs[int(len(xs) * k / 5)] for k in range(1, 5)]
        b = lambda x: sum(x >= e for e in edges)
        out["bins_logit_edges"] = [round(e, 3) for e in edges]
        out["curve"] = {m: [dict(bin=k, n=len(g), R_clean=mean([u["y"] for u in g]),
                                 c_mean=mean([u["c"] for u in g]))
                            for k in range(5) for g in [[u for u in us if u["model"] == m and b(u["x"]) == k]]]
                        for m in MODELS}
        out["per_level"] = {f"{m}|{lv}": dict(n=len(g), R_clean=mean([u["y"] for u in g]),
                                              c_mean=mean([u["c"] for u in g]))
                            for m in MODELS for lv, _, _ in levels(m)
                            for g in [[u for u in us if u["model"] == m and u["level"] == lv]]}
    os.makedirs(f"{R}/results/E3", exist_ok=True)
    json.dump(dict(out, units=us), open(f"{R}/results/E3/e3_report.json", "w"), indent=1)
    for k, v in status.items():
        pl = out.get("per_level", {}).get(k, {})
        print(f"{k:32s} {v:40s} n={pl.get('n')} R_clean={pl.get('R_clean')} c={pl.get('c_mean')}")
    print(f"units={n} recovered={ev} slope={out.get('slope')} CI={out.get('slope_ci95')} "
          f"fails={out.get('bootstrap_failures')} -> {out['verdict']}")


if __name__ == "__main__":
    main()
