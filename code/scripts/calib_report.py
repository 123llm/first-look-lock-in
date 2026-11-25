"""eps calibration report (calib_mme, single-view PGD, protocol v2).  Per (victim, eps): ASR tool-on / tool-off on the
paired denominator D_pair (clean-correct in both arms), the tool-off minus tool-on difference with image-cluster
bootstrap CI and exact McNemar, invalid rate, and the selection rule of the pilot protocol: the white-box eps for
H2a/H2b is the one whose tool-off ASR is in [0.5, 0.8] for every white-box victim (closest to 0.65 otherwise).
Calibration numbers are for choosing eps only; never paper results.

  calib_report.py ROOT_DEEPEYES ROOT_QWEN3   (results roots; DeepEyes ran on the A100, Qwen3 on the RTX 4080)
"""
from __future__ import annotations

import json, os, sys

import numpy as np

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa.metrics import rtap
from tpa.metrics.stats import mcnemar_exact, cluster_bootstrap

ROOTS = {"deepeyes_7b": sys.argv[1], "qwen3vl_4b_thinking": sys.argv[2]}


def rows(root, m, arm, e):
    p = f"{root}/results/raw/CALIB/calib_mme__{arm}__single_view@{m}__eps{e}__seed0.jsonl"
    return {r["qid"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}


out, table = {}, []
for m, root in ROOTS.items():
    for e in (1, 2, 4):
        on, off = rows(root, m, m, e), rows(root, m, f"{m}_tool_off", e)
        q = sorted(set(on) & set(off))
        D = [k for k in q if rtap.in_denominator(on[k]) and rtap.in_denominator(off[k])]
        s_on = np.array([rtap.item_indicators(on[k])["success"] for k in D], float)
        s_off = np.array([rtap.item_indicators(off[k])["success"] for k in D], float)
        inv_on = np.mean([on[k]["adv"]["status"] != "ok" for k in q]) if q else float("nan")
        inv_off = np.mean([off[k]["adv"]["status"] != "ok" for k in q]) if q else float("nan")
        cl = [on[k]["cluster"] for k in D]
        d = s_off - s_on
        _, lo, hi = cluster_bootstrap(list(d), lambda rs: float(np.mean(rs)) if rs else float("nan"), cl,
                                      B=2000, seed=0) if D else (None, float("nan"), float("nan"))
        ci = (lo, hi)
        mc = mcnemar_exact([bool(x) for x in s_off], [bool(x) for x in s_on]) if D else dict(n10=0, n01=0, p=1.0)
        b, c = mc["n10"], mc["n01"]
        rec = dict(model=m, eps=e, n_items=len(q), n_D=len(D),
                   clean_acc_on=round(float(np.mean([on[k]["clean"]["letter"] == on[k]["clean"]["answer"] for k in q])), 3),
                   clean_acc_off=round(float(np.mean([off[k]["clean"]["letter"] == off[k]["clean"]["answer"] for k in q])), 3),
                   asr_on=round(float(s_on.mean()), 3), asr_off=round(float(s_off.mean()), 3),
                   diff_off_minus_on=round(float(d.mean()), 3), ci=[round(float(x), 3) for x in ci],
                   mcnemar_p=round(float(mc["p"]), 4), n_off_only=b, n_on_only=c,
                   inv_on=round(float(inv_on), 3), inv_off=round(float(inv_off), 3))
        out[f"{m}@eps{e}"] = rec
        table.append(rec)

# selection: eps with tool-off ASR in [0.5, 0.8] for both victims; else minimise the max distance to 0.65
cands = []
for e in (1, 2, 4):
    a = [out[f"{m}@eps{e}"]["asr_off"] for m in ROOTS]
    inband = all(0.5 <= x <= 0.8 for x in a)
    cands.append((not inband, max(abs(x - 0.65) for x in a), e))
pick = sorted(cands)[0]
out["selection"] = dict(eps=pick[2], all_in_band=not pick[0], max_dist_to_0_65=round(pick[1], 3),
                        rule="tool-off ASR in [0.5, 0.8] for every white-box victim, else closest to 0.65")
os.makedirs("results/CALIB", exist_ok=True)
json.dump(out, open("results/CALIB/calib_report.json", "w"), indent=1)
print(f"{'model':22s} eps  nD  acc_on acc_off  ASR_on ASR_off  off-on  CI              p      inv_on inv_off")
for r in table:
    print(f"{r['model']:22s} {r['eps']:>3d} {r['n_D']:>3d}  {r['clean_acc_on']:.2f}   {r['clean_acc_off']:.2f}    "
          f"{r['asr_on']:.2f}   {r['asr_off']:.2f}   {r['diff_off_minus_on']:+.2f}  [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]  "
          f"{r['mcnemar_p']:.3f}  {r['inv_on']:.2f}   {r['inv_off']:.2f}")
print("selection:", out["selection"])
