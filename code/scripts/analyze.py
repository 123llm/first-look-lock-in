"""S9 analysis: raw EVAL rows -> per-cell RTAP metrics with image-cluster bootstrap CIs, hypothesis tests H1-H4
(protocol_v2 `hypotheses:`), cost summary and figure data.  Output: results/release/<id>/
  cells.csv           one row per (exp, split, eval_model, method, source, eps, seed); metrics point/lo/hi
  cells_nocontam.csv  same without contamination-flagged items (sensitivity)
  hypotheses.json     H1, H2a, H2b, H3, H4 statistics and support verdicts
  cost.csv            per (exp, split, method, source, eps): mean wall s/img, forward-backward passes, rollouts
  fig*.csv            data behind the result figures (make_figures.py)

  analyze.py --root results --id v1 [--B 2000]
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys, warnings
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio
from tpa.metrics import rtap
from tpa.metrics.stats import cluster_bootstrap, mcnemar_exact, holm, kendall_tau_b, spearman

warnings.filterwarnings("ignore")
NAME = re.compile(r"(?P<split>[^/]+?)__(?P<eval_model>[^/]+?)__(?P<method>[^/@]+)@(?P<source>[^/]+?)__eps(?P<eps>\d+)__seed(?P<seed>\d+)\.jsonl$")
TOOL_OFF = {"deepeyes_7b": "deepeyes_7b_tool_off", "minio3_7b": "minio3_7b_tool_off",
            "qwen3vl_4b_thinking": "qwen3vl_4b_thinking_tool_off"}
REASONING = {"deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "vision_r1_7b", "vlaa_thinker_7b",
             "qwen3vl_4b_thinking_tool_off", "minio3_7b_tool_off"}
MAIN = ["vstar_eval", "hrbench4k_eval", "s11_pilot"]   # s11_pilot only appears under --alias S11=E4


def load_raw(root, alias=None):
    """alias {'S11': 'E4'} lets the pilot be analysed with the same hypothesis code; when given, only the aliased
    experiments are loaded."""
    cells = {}
    for f in sorted(glob.glob(f"{root}/raw/*/*.jsonl")):
        if "/raw/clean/" in f or os.path.basename(f).startswith("veg__"):
            continue
        m = NAME.search(f)
        if not m:
            continue
        exp = f.split("/raw/")[1].split("/")[0]
        if alias:
            if exp not in alias:
                continue
            exp = alias[exp]
        key = (exp, m["split"], m["eval_model"], m["method"], m["source"], int(m["eps"]), int(m["seed"]))
        cells[key] = tio.read_jsonl(f)
    return cells


def success_map(rows):
    """qid -> (in_D, success) using the RTAP definitions."""
    out = {}
    for r in rows:
        if rtap.in_denominator(r):
            out[r["qid"]] = (True, rtap.item_indicators(r)["success"], r["cluster"])
    return out


def paired(rows_a, rows_b):
    """Items in the denominator of both cells: lists of success_a, success_b, cluster."""
    A, Bm = success_map(rows_a), success_map(rows_b)
    q = sorted(set(A) & set(Bm))
    return [A[k][1] for k in q], [Bm[k][1] for k in q], [A[k][2] for k in q]


def diff_ci(sa, sb, cl, B, seed=0):
    rows = list(zip(sa, sb))
    return cluster_bootstrap(rows, lambda rs: float(np.mean([x for x, _ in rs]) - np.mean([y for _, y in rs])) if rs else float("nan"),
                             cl, B=B, seed=seed)


def pool(cells, exp, splits, eval_model, method, source, eps=8):
    rows = []
    for sp in splits:
        rows += cells.get((exp, sp, eval_model, method, source, eps, 0), [])
    return rows


# ------------------------------------------------------------------ hypotheses
def h2a(cells, B):
    out, ps = [], []
    for m, off in TOOL_OFF.items():
        # single-view images of the white-box models are generated once under E4 and reused for E3 (H2a)
        on_r = pool(cells, "E3", MAIN, m, "single_view", m) or pool(cells, "E4", MAIN, m, "single_view", m)
        off_r = pool(cells, "E3", MAIN, off, "single_view", m) or pool(cells, "E4", MAIN, off, "single_view", m)
        if not on_r or not off_r:
            continue
        s_on, s_off, cl = paired(on_r, off_r)
        mc = mcnemar_exact(s_off, s_on)
        p, lo, hi = diff_ci(s_off, s_on, cl, B)
        out.append(dict(model=m, n=len(cl), asr_on=float(np.mean(s_on)), asr_off=float(np.mean(s_off)),
                        diff_off_minus_on=p, ci=[lo, hi], p=mc["p"], n_off_only=mc["n10"], n_on_only=mc["n01"]))
        ps.append(mc["p"])
    for r, ph in zip(out, holm(ps) if ps else []):
        r["p_holm"] = ph
        r["pass"] = r["diff_off_minus_on"] >= 0.10 and ph < 0.01
    n_pass = sum(r["pass"] for r in out)
    verdict = "support" if n_pass >= 2 else ("partial" if n_pass == 1 else "not_supported")
    return dict(per_model=out, n_pass=n_pass, verdict=verdict, s11_direction=[r["diff_off_minus_on"] > 0 for r in out])


def h2b(cells, B):
    gaps, comps = [], []
    for m in ["deepeyes_7b", "qwen3vl_4b_thinking"]:
        on_r, off_r = pool(cells, "E4", MAIN, m, "tpa", m), pool(cells, "E4", MAIN, TOOL_OFF[m], "tpa", m)
        if on_r and off_r:
            s_on, s_off, cl = paired(on_r, off_r)
            p, lo, hi = diff_ci(s_on, s_off, cl, B)
            gaps.append(dict(model=m, n=len(cl), asr_on=float(np.mean(s_on)), asr_off=float(np.mean(s_off)),
                             gap=abs(p), diff_on_minus_off=p, ci=[lo, hi], pass_=abs(p) <= 0.05))
        for eps, exp, splits, bases in [(8, "E4", MAIN, [("single_view", m), ("eot", m), ("mca_style", "clip3")]),
                                        (4, "E4b", ["e4b_eps4"], [("single_view", m), ("eot", m)])]:
            t_r = pool(cells, exp, splits, m, "tpa", m, eps)
            for meth, src in bases:
                b_r = pool(cells, exp, splits, m, meth, src, eps)
                if not t_r or not b_r:
                    continue
                s_t, s_b, cl = paired(t_r, b_r)
                mc = mcnemar_exact(s_t, s_b)
                p, lo, hi = diff_ci(s_t, s_b, cl, B)
                comps.append(dict(model=m, eps=eps, baseline=meth, n=len(cl), asr_tpa=float(np.mean(s_t)),
                                  asr_base=float(np.mean(s_b)), diff=p, ci=[lo, hi], p=mc["p"]))
    for c, ph in zip(comps, holm([c["p"] for c in comps]) if comps else []):
        c["p_holm"] = ph
        c["pass"] = c["diff"] >= 0.10 and ph < 0.01 and c["diff"] > 0
    n_pass = sum(c["pass"] for c in comps)
    all_gap = all(g["pass_"] for g in gaps) if gaps else False
    verdict = ("support" if comps and n_pass == len(comps) and all_gap else
               "partial" if comps and n_pass >= len(comps) / 2 else "not_supported")
    return dict(gap=gaps, comparisons=comps, n_pass=n_pass, n_comp=len(comps), verdict=verdict)


def h1(root, cells, B):
    """24 cells (6 models x 4 attacks): RTAP ASR vs caption-judge success rate; tau_b with image-cluster bootstrap."""
    judged = {}
    for f in glob.glob(f"{root}/raw/E1/h1_judged/*.jsonl"):
        m = NAME.search(f)
        judged[(m["eval_model"], m["method"], m["source"])] = {r["qid"]: r for r in tio.read_jsonl(f)}
    keys = sorted(k for k in judged if ("E1", "e1_subset", k[0], k[1], k[2], 8, 0) in cells)
    if len(keys) < 3:
        return dict(verdict="no_data")
    per_cell = {}
    for k in keys:
        rows = cells[("E1", "e1_subset", k[0], k[1], k[2], 8, 0)]
        per_cell[k] = dict(rtap={r["qid"]: (rtap.in_denominator(r), rtap.item_indicators(r)["success"]) for r in rows},
                           judge=judged[k], cluster={r["qid"]: r["cluster"] for r in rows})
    clusters = sorted({c for v in per_cell.values() for c in v["cluster"].values()})

    def stats(cl_weights, thr="5"):
        xs, ys = [], []
        for k in keys:
            v = per_cell[k]
            num = den = jn = jd = 0.0
            for q, (inD, s) in v["rtap"].items():
                w = cl_weights.get(v["cluster"][q], 0)
                if inD:
                    num += w * s; den += w
                j = v["judge"].get(q)
                if j and j.get("success", {}).get(thr) is not None:
                    jn += w * j["success"][thr]; jd += w
            xs.append(num / den if den else np.nan); ys.append(jn / jd if jd else np.nan)
        return xs, ys

    xs, ys = stats({c: 1 for c in clusters})
    tau = kendall_tau_b(xs, ys)
    rng = np.random.default_rng(0)
    taus = []
    for _ in range(B):
        pick = rng.integers(0, len(clusters), len(clusters))
        w = defaultdict(int)
        for i in pick:
            w[clusters[i]] += 1
        bx, by = stats(w)
        if not (np.isnan(bx).any() or np.isnan(by).any()):
            taus.append(kendall_tau_b(bx, by))
    lo, hi = np.quantile(taus, [0.025, 0.975])
    sens = {t: kendall_tau_b(*stats({c: 1 for c in clusters}, t)) for t in ("3", "7")}
    dbf = []
    for key, rows in cells.items():
        if key[0] in ("E1", "E4") and key[2] in REASONING and key[3] == "tpa":
            r = rtap.compute(rows, B=B, which=["DbF"])
            dbf.append(dict(cell="/".join(map(str, key[:5])), **r["DbF"]))
    dbf_ok = any(d["lo"] > 0 for d in dbf if not np.isnan(d["lo"]))
    support = tau < 0.5 and hi < 0.7 and dbf_ok
    return dict(cells=[dict(model=k[0], attack=f"{k[1]}@{k[2]}", asr_rtap=x, asr_judge=y) for k, x, y in zip(keys, xs, ys)],
                tau_b=tau, ci=[float(lo), float(hi)], tau_sensitivity=sens, dbf=dbf, dbf_lower_gt0=dbf_ok,
                verdict="support" if support else "not_supported")


def h3(root, cells):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    recs = []
    for key, rows in cells.items():
        exp, sp, em, meth, src, eps, seed = key
        if exp != "E5" or "@" in em or em == src:
            continue
        veg = {r["qid"]: r for r in tio.read_jsonl(f"{root}/raw/E5/veg__{sp}__{em}.jsonl")}
        for r in rows:
            if not rtap.in_denominator(r) or r["qid"] not in veg:
                continue
            v = veg[r["qid"]]
            recs.append(dict(model=em, base=src, cluster=r["cluster"], success=int(rtap.item_indicators(r)["success"]),
                             RG=v["rg"], VEG=int(bool(v.get("changed"))), cot=np.log1p(v["cot_tokens"])))
    if len(recs) < 30:
        return dict(verdict="no_data")
    df = pd.DataFrame(recs)
    df["zcot"] = (df["cot"] - df["cot"].mean()) / (df["cot"].std() or 1)
    f = "success ~ RG + VEG + RG:VEG + zcot + C(model)"
    out = dict(n=len(df), n_models=df["model"].nunique())
    try:
        mm = BinomialBayesMixedGLM.from_formula(f, {"img": "0 + C(cluster)"}, df).fit_vb()
        names = list(mm.model.exog_names)
        i3, i4 = names.index("RG:VEG"), names.index("zcot")
        b3, s3 = float(mm.fe_mean[i3]), float(mm.fe_sd[i3])
        b4, s4 = float(mm.fe_mean[i4]), float(mm.fe_sd[i4])
        out["mixed"] = dict(b3=b3, b3_ci=[b3 - 1.96 * s3, b3 + 1.96 * s3], b4=b4, b4_ci=[b4 - 1.96 * s4, b4 + 1.96 * s4],
                            or_per_sd_cot=float(np.exp(b4)), coef={n: float(v) for n, v in zip(names, mm.fe_mean)})
    except Exception as e:
        out["mixed"] = dict(error=str(e)[:300])
    try:
        g = smf.gee(f, groups="cluster", data=df, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = g.conf_int().loc["RG:VEG"].tolist()
        out["gee"] = dict(b3=float(g.params["RG:VEG"]), b3_ci=ci, b4=float(g.params["zcot"]))
    except Exception as e:
        out["gee"] = dict(error=str(e)[:300])
    # intervention: budget forcing (r@b...c0) vs default on the same base attack
    inter = []
    for key, rows in cells.items():
        exp, sp, em, meth, src, eps, seed = key
        if exp == "E5" and "@" in em:
            r0 = cells.get((exp, sp, em.split("@")[0], meth, src, eps, seed))
            if r0:
                s_def, s_bf, cl = paired(r0, rows)
                rg_def = np.mean([x["clean"]["n_reobs"] for x in r0]); rg_bf = np.mean([x["clean"]["n_reobs"] for x in rows])
                mc = mcnemar_exact(s_bf, s_def)
                inter.append(dict(model=em, base=src, split=sp, n=len(cl), asr_default=float(np.mean(s_def)),
                                  asr_forced=float(np.mean(s_bf)), rg_default=float(rg_def), rg_forced=float(rg_bf), p=mc["p"]))
    out["intervention"] = inter
    # descriptive: pair-level spearman between robustness gain and mean RG x VEG
    pair = []
    for (m, b), d in df.groupby(["model", "base"]):
        base_rows = []
        for sp in MAIN:
            base_rows += cells.get(("E5", sp, b, "single_view", b, 8, 0), [])
        if base_rows:
            asr_b = rtap.compute(base_rows, B=0)["ASR"]["point"]
            pair.append(dict(model=m, base=b, d_rob=asr_b - d["success"].mean(), rgveg=float((d["RG"] * d["VEG"]).mean()),
                             cot=float(d["cot"].mean())))
    if len(pair) >= 3:
        out["pair_spearman_rgveg"] = spearman([p["d_rob"] for p in pair], [p["rgveg"] for p in pair])
        out["pair_spearman_cot"] = spearman([p["d_rob"] for p in pair], [p["cot"] for p in pair])
    out["pairs"] = pair
    mx = out.get("mixed", {})
    ok = ("b3_ci" in mx and mx["b3_ci"][1] < 0 and (mx["b4_ci"][0] <= 0 <= mx["b4_ci"][1] or abs(np.log(mx["or_per_sd_cot"])) < np.log(1.2)))
    out["verdict"] = "support" if ok else "not_supported"
    out["data"] = df
    return out


def _slope(df):
    import statsmodels.api as sm
    if df["robust"].nunique() < 2:
        return float("nan")
    X = sm.add_constant(np.log(df["budget"].values.astype(float)))
    try:
        return float(sm.Logit(df["robust"].values, X).fit(disp=0).params[1])
    except Exception:
        return float("nan")


def h4(cells, B):
    rows = []
    for key, rs in cells.items():
        exp, sp, em, meth, src, eps, seed = key
        if exp != "E6" or "@b" not in em:
            continue
        m, b = em.split("@b")
        tok, calls = map(int, b.split("c"))
        kind = "adaptive" if meth.startswith("tpa") else "nonadaptive"
        for r in rs:
            if rtap.in_denominator(r):
                rows.append(dict(model=m, kind=kind, tok=tok, calls=calls, cluster=r["cluster"],
                                 robust=int(not rtap.item_indicators(r)["success"])))
    if not rows:
        return dict(verdict="no_data")
    df = pd.DataFrame(rows)
    out, ok_all = [], []
    for m, d in df.groupby("model"):
        d = d.copy()
        d["budget"] = d["tok"] if d["tok"].nunique() > 1 else d["calls"] + 1
        res = dict(model=m)
        for kind in ("nonadaptive", "adaptive"):
            dk = d[d.kind == kind]
            if len(dk):
                res[f"slope_{kind}"] = _slope(dk)
                res[f"rho_{kind}"] = spearman(*zip(*[(b, g.robust.mean()) for b, g in dk.groupby("budget")])) if dk.budget.nunique() > 2 else float("nan")
                res[f"curve_{kind}"] = [dict(budget=int(b), robustness=float(g.robust.mean()), n=len(g)) for b, g in dk.groupby("budget")]
        cl = sorted(d.cluster.unique())
        rng = np.random.default_rng(0)
        sn, sd = [], []
        groups = {c: g for c, g in d.groupby("cluster")}
        for _ in range(B):
            bd = pd.concat([groups[cl[i]] for i in rng.integers(0, len(cl), len(cl))])
            s_n = _slope(bd[bd.kind == "nonadaptive"])
            sn.append(s_n)
            if (d.kind == "adaptive").any():
                sd.append(_slope(bd[bd.kind == "adaptive"]) - s_n)
        sn = np.array([x for x in sn if not np.isnan(x)])
        res["slope_nonadaptive_ci"] = np.quantile(sn, [0.025, 0.975]).tolist() if len(sn) else None
        if sd:
            sd = np.array([x for x in sd if not np.isnan(x)])
            res["slope_diff"] = res.get("slope_adaptive", np.nan) - res.get("slope_nonadaptive", np.nan)
            res["slope_diff_ci"] = np.quantile(sd, [0.025, 0.975]).tolist()
            ok_all.append(res["slope_nonadaptive_ci"][0] > 0 and res["slope_diff_ci"][1] < 0)
        out.append(res)
    verdict = "support" if ok_all and all(ok_all) else ("partial" if any(ok_all) else "not_supported")
    return dict(per_model=out, verdict=verdict)


def cost(root):
    rows = []
    for idx in glob.glob(f"{root}/adv/*/*/*/index.jsonl"):
        parts = idx.split("/adv/")[1].split("/")
        exp, sp, cell = parts[0], parts[1], parts[2]
        rs = [r for r in tio.read_jsonl(idx) if r.get("error") is None]
        if not rs:
            continue
        rows.append(dict(exp=exp, split=sp, cell=cell, n=len(rs), wall_s=float(np.mean([r["t"] for r in rs])),
                         fwd_bwd=float(np.mean([r["attack"].get("n_fwd_bwd", np.nan) for r in rs])),
                         grad_evals=float(np.mean([r["attack"].get("n_grad_evals", np.nan) for r in rs])),
                         rollouts=float(np.mean([r["attack"].get("n_rollouts", 0) for r in rs]))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{tio.R}/results"); ap.add_argument("--id", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--alias", default="", help="e.g. S11=E4 (analyse the pilot with the E4 hypothesis code)")
    a = ap.parse_args()
    out = f"{a.root}/release/{a.id}"
    os.makedirs(out, exist_ok=True)
    cells = load_raw(a.root, dict(kv.split("=") for kv in a.alias.split(",")) if a.alias else None)
    print(f"{len(cells)} cells", flush=True)
    M = ["ASR", "INV", "PFR", "PFR_clean", "RFR", "SRR", "FSP", "DbF", "DbF_D", "CIR_tokens", "CIR_calls", "RG_clean", "RG_adv"]
    for name, filt in [("cells", None), ("cells_nocontam", True)]:
        recs = []
        for key, rows in sorted(cells.items()):
            rs = [r for r in rows if not (filt and r.get("contaminated"))]
            m = rtap.compute(rs, B=a.B)
            rec = dict(zip(["exp", "split", "eval_model", "method", "source", "eps", "seed"], key),
                       n_rows=m["n_rows"], n_D=m["n_D"], clean_acc=m["clean_acc"])
            for k in M:
                rec[k], rec[k + "_lo"], rec[k + "_hi"] = m[k]["point"], m[k]["lo"], m[k]["hi"]
            recs.append(rec)
        pd.DataFrame(recs).to_csv(f"{out}/{name}.csv", index=False)
        if name == "cells" and not any(r.get("contaminated") for rows in cells.values() for r in rows):
            break
    hyp = dict(H2a=h2a(cells, a.B), H2b=h2b(cells, a.B), H1=h1(a.root, cells, a.B), H4=h4(cells, a.B))
    h = h3(a.root, cells)
    if "data" in h:
        h.pop("data").to_csv(f"{out}/fig5_h3_items.csv", index=False)
    hyp["H3"] = h
    json.dump(hyp, open(f"{out}/hypotheses.json", "w"), indent=1, default=float)
    cost(a.root).to_csv(f"{out}/cost.csv", index=False)
    for k, v in hyp.items():
        print(k, v.get("verdict"), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
