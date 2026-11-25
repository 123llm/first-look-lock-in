"""Transfer probe report (go/no-go for the I2 rewrite, never paper results).
S11 DeepEyes ε=8 images (single_view 100, tpa 84) on the three transfer models.  Paired on items that have both an
SV and a TPA image and are clean-correct for the transfer model (rtap.in_denominator).  Per model: ASR_SV, ASR_TPA,
difference with image-cluster bootstrap CI, exact McNemar; pooled over models with Holm on the three p values.
Also invalid-output rates, so a 'success' driven by broken outputs is visible.
"""
from __future__ import annotations

import json, os, sys

import numpy as np

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa.metrics import rtap
from tpa.metrics.stats import mcnemar_exact, cluster_bootstrap, holm

R = os.environ.get("TPA_ROOT", ".")
MODELS = ["minio3_7b", "vision_r1_7b", "vlaa_thinker_7b"]


def load(m, meth):
    p = f"{R}/results/raw/S11/s11_pilot__{m}__{meth}@deepeyes_7b__eps8__seed0.jsonl"
    return {r["qid"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


out, ps, pooled_sv, pooled_tpa, pooled_cl = {}, [], [], [], []
for m in MODELS:
    sv, tp = load(m, "single_view"), load(m, "tpa")
    q = sorted(set(sv) & set(tp))
    D = [k for k in q if rtap.in_denominator(sv[k])]
    a = np.array([rtap.item_indicators(sv[k])["success"] for k in D], float)
    b = np.array([rtap.item_indicators(tp[k])["success"] for k in D], float)
    cl = [sv[k]["cluster"] for k in D]
    d = b - a
    _, lo, hi = cluster_bootstrap(list(d), lambda rs: float(np.mean(rs)) if rs else float("nan"), cl, B=2000, seed=0)
    mc = mcnemar_exact([bool(x) for x in b], [bool(x) for x in a])
    ps.append(mc["p"])
    pooled_sv += list(a); pooled_tpa += list(b); pooled_cl += [f"{m}|{c}" for c in cl]
    out[m] = dict(n_paired=len(q), n_D=len(D), clean_acc=round(float(np.mean([sv[k]["clean"]["letter"] == sv[k]["clean"]["answer"] for k in q])), 3),
                  asr_sv=round(float(a.mean()), 3), asr_tpa=round(float(b.mean()), 3),
                  diff_tpa_minus_sv=round(float(d.mean()), 3), ci=[round(lo, 3), round(hi, 3)],
                  tpa_only=mc["n10"], sv_only=mc["n01"], p=round(mc["p"], 4),
                  inv_sv=round(float(np.mean([sv[k]["adv"]["status"] != "ok" for k in D])), 3),
                  inv_tpa=round(float(np.mean([tp[k]["adv"]["status"] != "ok" for k in D])), 3))
for m, ph in zip(MODELS, holm(ps)):
    out[m]["p_holm"] = round(ph, 4)
A, B = np.array(pooled_sv), np.array(pooled_tpa)
_, lo, hi = cluster_bootstrap(list(B - A), lambda rs: float(np.mean(rs)) if rs else float("nan"), pooled_cl, B=2000, seed=0)
mc = mcnemar_exact([bool(x) for x in B], [bool(x) for x in A])
out["pooled"] = dict(n=len(A), asr_sv=round(float(A.mean()), 3), asr_tpa=round(float(B.mean()), 3),
                     diff=round(float((B - A).mean()), 3), ci=[round(lo, 3), round(hi, 3)],
                     tpa_only=mc["n10"], sv_only=mc["n01"], p=round(mc["p"], 4))
os.makedirs(f"{R}/results/S11", exist_ok=True)
json.dump(out, open(f"{R}/results/S11/transfer_probe.json", "w"), indent=1)
print(f"{'model':16s} nD  acc   ASR_SV ASR_TPA  TPA-SV  CI              TPA>SV SV>TPA  p      p_holm inv_sv inv_tpa")
for m in MODELS:
    r = out[m]
    print(f"{m:16s} {r['n_D']:>3d} {r['clean_acc']:.2f}  {r['asr_sv']:.3f}  {r['asr_tpa']:.3f}   {r['diff_tpa_minus_sv']:+.3f} "
          f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  {r['tpa_only']:>3d}    {r['sv_only']:>3d}   {r['p']:.4f} {r['p_holm']:.4f} {r['inv_sv']:.2f}   {r['inv_tpa']:.2f}")
r = out["pooled"]
print(f"{'pooled':16s} {r['n']:>3d}       {r['asr_sv']:.3f}  {r['asr_tpa']:.3f}   {r['diff']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  "
      f"{r['tpa_only']:>3d}    {r['sv_only']:>3d}   {r['p']:.4f}")
