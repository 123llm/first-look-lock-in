"""S24 derived analyses for the revised paper (post hoc, CPU only; no new model runs).

  s24_derived.py  -> results/release/s18_v1/derived.json

All inputs are the item-level records already used by the S18-verified reports (v2_main, first 150 questions).
1. dynamics   per model: answer of the forced-answer probe along the trajectory, for
              (adv)   white-box eps8 flipped and zoomed questions (the E1 F_z set) and
              (clean) clean first-look errors that zoomed (the C_nat set);
              share whose probe letter is correct / equals the first-look letter at turn 0, after the first returned
              view, after the last returned view, and the final answer.
2. law        the E3 recovery regression refitted with its fixed effects (same units, same estimator), the implied
              evidence scale s = -1/beta, predicted vs observed recovery per confidence bin, per-model slopes
              (exploratory, cluster bootstrap), and an out-of-sample check on clean first-look errors: the fitted law
              evaluated at their first-look confidence versus their observed natural correction rate.
3. views      per model: zoom boxes on the white-box eps8 adversarial images and on clean images -- share of calls on
              an earlier zoomed view (nested), and box area as a fraction of the image for calls on the global view.
"""
from __future__ import annotations

import json, math, os, random, sys, warnings

import numpy as np

ROOT = os.environ.get("TPA_ROOT", ".")
sys.path.insert(0, ROOT + "/code")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpa import io as tio
import s12_e1_report as E1
import s12_e3_report as E3

MODELS = ["deepeyes_7b", "minio3_7b", "pixelreasoner_7b", "qwen3vl_4b_thinking", "qwen3vl_8b_thinking",
          "glm41v_9b_thinking"]
FIT = E3.MODELS
ok = E1.ok


def sig(z):
    return 1 / (1 + math.exp(-z))


def logit_c(c):
    c = min(max(c, 1e-4), 1 - 1e-4)
    return math.log(c / (1 - c))


def load(m, adv):
    keep = set(E1.first_qids())
    p = tio.raw_path("E1", "v2_main", m, *tio.parse_adv_spec(adv))
    return {r["qid"]: r for r in tio.read_jsonl(p) if r["qid"] in keep}


def share(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def dynamics():
    out = {}
    for m in MODELS:
        ev = load(m, f"single_view@{m}__eps8__seed0")
        sets = {
            "adv": [r["adv"] for r in ev.values() if ok(r["clean"]) and not ok(r["adv"]) and r["adv"]["n_reobs"] >= 1],
            "clean": [r["clean"] for r in ev.values() if r["clean"].get("probes")
                      and r["clean"]["probes"][0]["letter"] != r["clean"]["answer"] and r["clean"]["n_reobs"] >= 1],
        }
        d = {}
        for k, rs in sets.items():
            rs = [r for r in rs if r.get("probes")]
            row = dict(n=len(rs))
            for name, pick in (("turn0", lambda p: p[0]), ("view1", lambda p: p[1] if len(p) > 1 else None),
                               ("last", lambda p: p[-1] if len(p) > 1 else None)):
                ps = [(r, pick(r["probes"])) for r in rs]
                ps = [(r, p) for r, p in ps if p is not None]
                row[name] = dict(n=len(ps), correct=share([int(p["letter"] == r["answer"]) for r, p in ps]),
                                 first=share([int(p["letter"] == r["probes"][0]["letter"]) for r, p in ps]),
                                 p_true=share([p["probs"]["ABCDEFGH".index(r["answer"])] for r, p in ps]))
            row["final"] = dict(n=len(rs), correct=share([int(ok(r)) for r in rs]),
                                first=share([int(r["letter"] == r["probes"][0]["letter"]) for r in rs]))
            d[k] = row
        out[m] = d
    return out


def logit_fit(xs, ys, ms, models):
    import statsmodels.api as sm
    X = np.array([[x] + [float(m == mm) for mm in models] for x, m in zip(xs, ms)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.Logit(np.array(ys), X).fit(disp=0, method="newton", maxiter=100)
    if not r.mle_retvals.get("converged", True) or not np.all(np.isfinite(r.params)):
        return None
    return [float(v) for v in r.params]


def boot_slope(us, models, B=2000, seed=0):
    by = {}
    for u in us:
        by.setdefault(u["cluster"], []).append(u)
    cl = sorted(by); rng = random.Random(seed); bs = []
    for _ in range(B):
        s = [u for c in (rng.choice(cl) for _ in cl) for u in by[c]]
        mm = [m for m in models if len({u["y"] for u in s if u["model"] == m}) == 2]
        if not mm:
            continue
        s = [u for u in s if u["model"] in mm]
        p = logit_fit([u["x"] for u in s], [u["y"] for u in s], [u["model"] for u in s], mm)
        if p is not None and abs(p[0]) < 50:
            bs.append(p[0])
    bs.sort()
    return [round(bs[int(0.025 * len(bs))], 3), round(bs[int(0.975 * len(bs)) - 1], 3)], len(bs)


def law():
    us, _ = E3.units()
    models = [m for m in FIT if len({u["y"] for u in us if u["model"] == m}) == 2]
    p = logit_fit([u["x"] for u in us], [u["y"] for u in us], [u["model"] for u in us], models)
    beta, gam = p[0], dict(zip(models, p[1:]))
    rep = json.load(open(f"{ROOT}/results/release/s18_v1/reports/e3_report.json"))
    assert abs(beta - rep["slope"]) < 1e-3, (beta, rep["slope"])
    ci = rep["slope_ci95"]
    edges = rep["bins_logit_edges"]

    def b_of(x):
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)
    bins = {}
    for m in models:
        rows = []
        for b in range(len(edges) + 1):
            s = [u for u in us if u["model"] == m and b_of(u["x"]) == b]
            if s:
                rows.append(dict(bin=b, n=len(s), x_mean=round(sum(u["x"] for u in s) / len(s), 3),
                                 obs=round(sum(u["y"] for u in s) / len(s), 4),
                                 pred=round(sum(sig(beta * u["x"] + gam[m]) for u in s) / len(s), 4)))
        bins[m] = rows
    per_model = {}
    for m in models:
        s = [u for u in us if u["model"] == m]
        pm = logit_fit([u["x"] for u in s], [u["y"] for u in s], [m] * len(s), [m])
        if pm is None:
            per_model[m] = None; continue
        bci, nb = boot_slope(s, [m])
        per_model[m] = dict(n=len(s), events=sum(u["y"] for u in s), slope=round(pm[0], 3), ci95=bci, boot_ok=nb)
    # out-of-sample: clean first-look errors that zoomed (the C_nat set) of the fitted models
    oos = {}
    for m in models:
        ev = load(m, f"single_view@{m}__eps8__seed0")
        rs = [r["clean"] for r in ev.values() if r["clean"].get("probes")
              and r["clean"]["probes"][0]["letter"] != r["clean"]["answer"] and r["clean"]["n_reobs"] >= 1]
        xs = [logit_c(max(r["probes"][0]["probs"])) for r in rs]
        oos[m] = dict(n=len(rs), c_mean=round(sum(max(r["probes"][0]["probs"]) for r in rs) / len(rs), 4),
                      observed=round(sum(int(ok(r)) for r in rs) / len(rs), 4),
                      predicted=round(sum(sig(beta * x + gam[m]) for x in xs) / len(xs), 4))
    return dict(beta=round(beta, 4), beta_ci95=ci, s=round(-1 / beta, 3),
                s_ci95=[round(-1 / ci[0], 3), round(-1 / ci[1], 3)], gamma={m: round(v, 4) for m, v in gam.items()},
                models_in_fit=models, n_units=len(us), bins=bins, per_model=per_model, clean_errors=oos,
                x_clip=round(logit_c(1.0), 3))


def persistence(B=2000, seed=0):
    """Matched-confidence comparison: y ~ beta*logit(c) + alpha*adv + gamma_M over
    (adv)     the E3 units (white-box iterates and transfer images, clean-zoom replay outcome) and
    (natural) the clean first-look errors that zoomed (outcome = their own clean trajectory),
    for the three E3 models.  Both outcomes come from trajectories whose zoomed views are clean; they differ in the
    global image (adversarial vs clean) and the text the model wrote with it.  alpha < 0 means that at equal
    first-look confidence an adversarial first look is corrected less often than a natural one."""
    us, _ = E3.units()
    us = [dict(u, adv=1) for u in us]
    for m in FIT:
        ev = load(m, f"single_view@{m}__eps8__seed0")
        for r0 in ev.values():
            r = r0["clean"]
            if r.get("probes") and r["probes"][0]["letter"] != r["answer"] and r["n_reobs"] >= 1:
                us.append(dict(model=m, cluster=r0["cluster"], x=logit_c(max(r["probes"][0]["probs"])),
                               y=int(ok(r)), adv=0))

    def fit(s):
        import statsmodels.api as sm
        mm = [m for m in FIT if len({u["y"] for u in s if u["model"] == m}) == 2]
        X = np.array([[u["x"], u["adv"]] + [float(u["model"] == m) for m in mm] for u in s if u["model"] in mm])
        y = np.array([u["y"] for u in s if u["model"] in mm])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100)
            except Exception:
                return None
        if not r.mle_retvals.get("converged", True) or not np.all(np.isfinite(r.params)):
            return None
        return float(r.params[0]), float(r.params[1])
    b, a = fit(us)
    by = {}
    for u in us:
        by.setdefault(u["cluster"], []).append(u)
    cl = sorted(by); rng = random.Random(seed); A, Bs = [], []
    for _ in range(B):
        p = fit([u for c in (rng.choice(cl) for _ in cl) for u in by[c]])
        if p is not None and abs(p[1]) < 50:
            Bs.append(p[0]); A.append(p[1])
    A.sort(); Bs.sort()
    q = lambda v: [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]
    nat = [u for u in us if not u["adv"]]
    low = {m: dict(adv=[u for u in us if u["adv"] and u["model"] == m and u["x"] < 3.245],
                   nat=[u for u in nat if u["model"] == m and u["x"] < 3.245]) for m in FIT}
    matched = {m: {k: dict(n=len(v), c_mean=round(sum(sig(u["x"]) for u in v) / len(v), 3) if v else None,
                           rec=round(sum(u["y"] for u in v) / len(v), 3) if v else None) for k, v in d.items()}
               for m, d in low.items()}
    return dict(n_adv=len(us) - len(nat), n_natural=len(nat), beta=round(b, 3), alpha=round(a, 3),
                beta_ci95=q(Bs), alpha_ci95=q(A), boot_ok=len(A), matched_c_below_0p9625=matched)


def views():
    items = {it["qid"]: it for it in tio.load_split("v2_main")[:150]}
    out = {}
    for m in MODELS:
        ev = load(m, f"single_view@{m}__eps8__seed0")
        d = {}
        for which in ("adv", "clean"):
            area, n, nested = [], 0, 0
            for q, r in ev.items():
                it = items[q]
                for b in r[which]["traj"]["boxes"]:
                    if not b:
                        continue
                    n += 1
                    if len(b) > 4 and b[4]:
                        nested += 1; continue
                    area.append((b[2] - b[0]) * (b[3] - b[1]) / (it["w"] * it["h"]))
            area.sort()
            d[which] = dict(n_calls=n, nested=round(nested / n, 4) if n else None,
                            area_median=round(area[len(area) // 2], 4) if area else None,
                            area_in_eot_range=round(sum(0.05 <= a <= 0.30 for a in area) / len(area), 4) if area else None)
        out[m] = d
    return out


if __name__ == "__main__":
    res = dict(note="post hoc analyses on the S18 records; see s24_derived.py", dynamics=dynamics(), law=law(),
               views=views(), persistence=persistence())
    p = f"{ROOT}/results/release/s18_v1/derived.json"
    json.dump(res, open(p, "w"), indent=1)
    print(json.dumps(res["law"], indent=1)[:3000])
    for m, d in res["dynamics"].items():
        print(m, {k: (v["n"], v["turn0"]["correct"], v["view1"]["correct"], v["last"]["correct"], v["final"]["correct"],
                      v["last"]["first"]) for k, v in d.items()})
    print(json.dumps(res["persistence"], indent=1))
