"""H-OP check 1 report (pre-registered on 2026-11-06): TPA-R+J (box jitter on the EIR term) on
DeepEyes, against TPA-R and EOT-R on the same 20 R3 items.

  pilothop_report.py  -> results/PILOTR/hop_check1_report.json and a printed table
Verdict: support H-OP iff ASR_def(TPA-R+J) >= 0.75; not supported iff <= 0.60; otherwise inconclusive.
"""
from __future__ import annotations

import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilotr3_report import cell, ci, R, D

CELLS = {"TPA-R+J": ("PILOTR3", f"tpa_eir_j15@{D}__eps8__seed0"), "TPA-R": ("PILOTR3", f"tpa_eir@{D}__eps8__seed0"),
         "EOT-R": ("PILOTR3", f"eot_eir@{D}__eps8__seed0")}


def main():
    qids = [l.strip() for l in open(f"{R}/code/configs/pilotr3_{D}_qids.txt") if l.strip()]
    cells, wr = {}, {}
    for k, (ex, adv) in CELLS.items():
        cells[k], wr[k] = cell(D, ex, adv, qids)
    a = cells["TPA-R+J"]["ASR_def"]
    verdict = "SUPPORTED" if a >= 0.75 else ("NOT_SUPPORTED" if a <= 0.60 else "INCONCLUSIVE")
    paired = {}
    for ref in ("TPA-R", "EOT-R"):
        common = [q for q in wr["TPA-R+J"] if q in wr[ref]]
        d = [wr["TPA-R+J"][q] - wr[ref][q] for q in common]
        paired[f"J_minus_{ref}"] = dict(diff=round(sum(d) / len(d), 3), ci95=ci(d), n=len(common),
                                        J_only=sum(x > 0 for x in d), ref_only=sum(x < 0 for x in d))
    out = dict(model=D, cells=cells, verdict=verdict, paired=paired)
    json.dump(out, open(f"{R}/results/PILOTR/hop_check1_report.json", "w"), indent=1)
    print("verdict", verdict)
    for k, c in cells.items():
        print(f"  {k:8s} n={c['n']} ASR_def={c['ASR_def']} {c['ASR_def_ci95']} ASR_agent={c['ASR_agent']} "
              f"cov={c['coverage']} how={c['how']}")
    for k, v in paired.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
