"""Mechanism pilot report (pre-registered on 2026-11-06).

  pilotm_report.py  -> results/PILOTM/pilotm_report.json and a printed table

A(full, text) = accuracy of the forced answer after the last observation.  Per model (cells pooled):
  replication  A(adv, own) <= 0.30          sufficiency  A(gray, neutral) >= 0.50
  V = A(gray, own) - A(adv, own)            T = A(adv, neutral) - A(adv, own)
  dominant channel: effect >= 0.30 and >= other + 0.15; redundant: both < 0.30 and A(gray,neutral) - A(adv,own) >= 0.30
GO iff >= 2 models pass replication and sufficiency with an identified mechanism (dominant or redundant).
"""
from __future__ import annotations

import glob, json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

CONDS = [f"{f}|{t}" for f in ("adv", "gray", "clean") for t in ("own", "neutral")]


def ci(xs, B=2000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed); n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]


def summarize(rows):
    rows = [r for r in rows if "conds" in r]
    acc = {c: [int(r["conds"][c]["letter"] == r["answer"]) for r in rows] for c in CONDS}
    A = {c: (round(sum(v) / len(v), 3) if v else None) for c, v in acc.items()}
    pc = {c: round(sum(r["conds"][c]["p_correct"] for r in rows) / len(rows), 3) if rows else None for c in CONDS}
    diff = lambda c1, c2: [x - y for x, y in zip(acc[c1], acc[c2])]
    return dict(n=len(rows), A=A, A_ci95={c: ci(v) for c, v in acc.items()}, p_correct=pc,
                V=round(A["gray|own"] - A["adv|own"], 3) if rows else None,
                T=round(A["adv|neutral"] - A["adv|own"], 3) if rows else None,
                V_ci95=ci(diff("gray|own", "adv|own")), T_ci95=ci(diff("adv|neutral", "adv|own")))


def verdict(s):
    if not s["n"]:
        return dict(replication=None, sufficiency=None, mechanism=None)
    A, V, T = s["A"], s["V"], s["T"]
    rep, suf = A["adv|own"] <= 0.30, A["gray|neutral"] >= 0.50
    if V >= 0.30 and V >= T + 0.15:
        mech = "visual"
    elif T >= 0.30 and T >= V + 0.15:
        mech = "textual"
    elif V < 0.30 and T < 0.30 and A["gray|neutral"] - A["adv|own"] >= 0.30:
        mech = "redundant"
    else:
        mech = "unidentified"
    return dict(replication=rep, sufficiency=suf, mechanism=mech,
                passes=bool(rep and suf and mech in ("visual", "textual", "redundant")))


def main():
    files = sorted(glob.glob(f"{tio.R}/results/raw/PILOTM/*.jsonl"))
    cells, by_model = {}, {}
    for f in files:
        rows = tio.read_jsonl(f)
        name = os.path.basename(f)[:-6]
        cells[name] = dict(summarize(rows), skipped=sum("skip" in r for r in rows))
        by_model.setdefault(rows[0]["model"] if rows else name.split("__")[0], []).extend(rows)
    models = {m: dict(summarize(rs), **verdict(summarize(rs))) for m, rs in by_model.items()}
    passing = [m for m, v in models.items() if v.get("passes")]
    mechs = [models[m]["mechanism"] for m in passing]
    out = dict(verdict="GO" if len(passing) >= 2 else "ROUTE_B", passing_models=passing,
               majority_mechanism=max(set(mechs), key=mechs.count) if mechs else None, models=models, cells=cells)
    os.makedirs(f"{tio.R}/results/PILOTM", exist_ok=True)
    json.dump(out, open(f"{tio.R}/results/PILOTM/pilotm_report.json", "w"), indent=1)
    hdr = " ".join(f"{c:>13s}" for c in CONDS)
    print(f"{'model/cell':58s} {'n':>3s} {hdr}   V      T")
    for name, s in list(models.items()) + list(cells.items()):
        if not s["n"]:
            continue
        print(f"{name[:58]:58s} {s['n']:3d} " + " ".join(f"{s['A'][c]:13.2f}" for c in CONDS)
              + f" {s['V']:6.2f} {s['T']:6.2f}" + (f"  {s['mechanism']} rep={s['replication']} suf={s['sufficiency']}"
                                                 if "mechanism" in s else ""))
    print("verdict:", out["verdict"], "passing:", passing, "majority:", out["majority_mechanism"])


if __name__ == "__main__":
    main()
