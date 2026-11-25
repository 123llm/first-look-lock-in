"""E2 (two-channel mechanism) and E4-R1/R2 (EIR cost, non-adaptive recovery) report; rules pre-registered on
2026-11-07.

  s12_e2_e4r_report.py  -> results/E2/e2_report.json, results/E4/e4r_report.json and printed tables

E2, per model (E1 eps8 white-box cell, items entering clean_zoom; models with n < 10 do not take part):
  replication A(adv, own) <= 0.30; sufficiency A(gray, neutral) >= 0.50; V, T and the dominant / redundant /
  unidentified thresholds as in PILOTM.  I3a: >= 3 passing models -> SUPPORTED; 1-2 -> PARTIAL; 0 -> NOT SUPPORTED.
E4-R1 (descriptive): dA = A_def(clean) - A_agent(clean), v2_main first 150.
E4-R2: per E1 white-box cell (eps 4, 8), on clean-correct and flipped items, Rec = fraction correct after EIR; cell
  passes iff n >= 10 and Rec >= 0.30; model passes iff at least half of its cells pass.
  I3c: >= 3 passing models -> SUPPORTED; 1-2 -> PARTIAL; 0 -> NOT SUPPORTED.
CIs: cluster bootstrap over image_cluster, 2000 resamples, seed 0.
"""
from __future__ import annotations

import json, os, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s12_e1_report import MODELS, cluster_ci, mean, first_qids
from pilotm_report import CONDS, verdict as mech_verdict

R = tio.R


def three_way(npass, hi=3):
    return "SUPPORTED" if npass >= hi else "PARTIAL" if npass >= 1 else "NOT SUPPORTED"


def e2():
    models = {}
    for m in MODELS:
        f = f"{R}/results/raw/E2/{m}__E1__single_view@{m}__eps8__seed0.jsonl"
        if not os.path.exists(f):
            models[m] = dict(missing=True); continue
        rows = [r for r in tio.read_jsonl(f) if "conds" in r]
        acc = {c: [(r["cluster"], int(r["conds"][c]["letter"] == r["answer"])) for r in rows] for c in CONDS}
        A = {c: mean([v for _, v in xs]) for c, xs in acc.items()}
        dif = lambda c1, c2: [(k, x - y) for (k, x), (_, y) in zip(acc[c1], acc[c2])]
        s = dict(n=len(rows), A=A, A_ci95={c: cluster_ci(xs) for c, xs in acc.items()},
                 V=round(A["gray|own"] - A["adv|own"], 3) if rows else None,
                 T=round(A["adv|neutral"] - A["adv|own"], 3) if rows else None,
                 V_ci95=cluster_ci(dif("gray|own", "adv|own")), T_ci95=cluster_ci(dif("adv|neutral", "adv|own")))
        s.update(mech_verdict(s))
        s["takes_part"] = len(rows) >= 10
        if not s["takes_part"]:
            s["passes"] = False
        models[m] = s
    passing = [m for m, s in models.items() if s.get("passes")]
    out = dict(verdict=three_way(len(passing)), passing_models=passing,
               pending=[m for m, s in models.items() if s.get("missing")], models=models)
    os.makedirs(f"{R}/results/E2", exist_ok=True)
    json.dump(out, open(f"{R}/results/E2/e2_report.json", "w"), indent=1)
    print(f"{'E2 model':22s} {'n':>3s} " + " ".join(f"{c:>13s}" for c in CONDS) + "      V      T  mechanism")
    for m, s in models.items():
        if s.get("missing") or not s["n"]:
            print(f"{m:22s} missing"); continue
        print(f"{m:22s} {s['n']:3d} " + " ".join(f"{s['A'][c]:13.2f}" for c in CONDS)
              + f" {s['V']:6.2f} {s['T']:6.2f}  {s['mechanism']} rep={s['replication']} suf={s['sufficiency']} "
                f"part={s['takes_part']}")
    print("E2 verdict:", out["verdict"], out["pending"] and f"(pending {out['pending']})" or "", passing)


def load(path, keep):
    return {r["qid"]: r for r in tio.read_jsonl(path) if r["qid"] in keep} if os.path.exists(path) else None


def e4r():
    keep = set(first_qids())
    okd = lambda r, agent=False: (r["agent"]["letter"] if agent else r["letter"]) == r["answer"]
    models, cells = {}, {}
    for m in MODELS:
        c = load(f"{R}/results/raw/E4/clean__eir__{m}__v2_main.jsonl", keep)
        if c is None:
            models[m] = dict(missing="R1"); continue
        rs = list(c.values())
        dA = [(r["cluster"], int(okd(r)) - int(okd(r, True))) for r in rs]
        r1 = dict(n=len(rs), A_agent=mean([int(okd(r, True)) for r in rs]), A_def=mean([int(okd(r)) for r in rs]),
                  dA=mean([v for _, v in dA]), dA_ci95=cluster_ci(dA),
                  applied=mean([int(r["eir"].get("applied", False)) for r in rs]),
                  both_correct=sum(okd(r) and okd(r, True) for r in rs),
                  fallback_reanswer=sum(bool(r["eir"].get("applied") and r["eir"].get("fallback")) for r in rs))
        mc = []
        for e in (8, 4):
            adv = f"single_view@{m}__eps{e}__seed0"
            a = load(f"{R}/results/raw/E4/eir__{m}__E1__{adv}.jsonl", keep)
            if a is None:
                cells[f"{m}__{adv}"] = dict(missing=True); mc.append(None); continue
            common = [q for q in a if q in c]
            fl = [q for q in common if okd(c[q], True) and not okd(a[q], True)]
            rec = [(a[q]["cluster"], int(okd(a[q]))) for q in fl]
            dc = [q for q in common if okd(c[q])]
            ac = [q for q in common if okd(c[q], True)]
            x = dict(n_items=len(common), n_flipped=len(fl), Rec=mean([v for _, v in rec]), Rec_ci95=cluster_ci(rec),
                     coverage=mean([int(a[q]["agent"]["n_reobs"] >= 1) for q in fl]),
                     ASR_agent=mean([int(not okd(a[q], True)) for q in ac]),
                     ASR_def=mean([int(not okd(a[q])) for q in dc]))
            x["passes"] = bool(len(fl) >= 10 and x["Rec"] is not None and x["Rec"] >= 0.30)
            cells[f"{m}__{adv}"] = x; mc.append(x["passes"])
        done = [p for p in mc if p is not None]
        models[m] = dict(R1=r1, cells_passing=sum(done), cells_done=len(done), pending=len(done) < len(mc),
                         passes=bool(done and sum(done) * 2 >= len(mc)))
    passing = [m for m, x in models.items() if x.get("passes")]
    pend = [m for m, x in models.items() if x.get("missing") or x.get("pending")]
    out = dict(verdict=three_way(len(passing)), passing_models=passing, pending=pend, models=models, cells=cells)
    os.makedirs(f"{R}/results/E4", exist_ok=True)
    json.dump(out, open(f"{R}/results/E4/e4r_report.json", "w"), indent=1)
    for m, x in models.items():
        if "R1" not in x:
            print(f"R1 {m:22s} missing"); continue
        r = x["R1"]
        print(f"R1 {m:22s} n={r['n']:3d} A_agent={r['A_agent']} A_def={r['A_def']} dA={r['dA']} {r['dA_ci95']} "
              f"applied={r['applied']} both={r['both_correct']} fallback={r['fallback_reanswer']}")
    for k, c in cells.items():
        if c.get("missing"):
            print(f"R2 {k:52s} missing"); continue
        print(f"R2 {k[:52]:52s} n={c['n_flipped']:3d} Rec={c['Rec']} {c['Rec_ci95']} cov={c['coverage']} "
              f"ASR ag={c['ASR_agent']} def={c['ASR_def']} pass={c['passes']}")
    print("I3c verdict:", out["verdict"], f"(pending {pend})" if pend else "", passing)


if __name__ == "__main__":
    e2(); e4r()
