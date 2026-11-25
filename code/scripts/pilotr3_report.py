"""Remedy pilot R3 report: adaptive attacks through EIR (pre-registered on 2026-11-06).

  pilotr3_report.py  -> results/PILOTR/pilotr3_report.json and a printed table

Per model, on its 20 R3 items (code/configs/pilotr3_<model>_qids.txt; agent and EIR both correct on the clean image):
  ASR_def(cell)  fraction answered wrong by the defended system (agent rollout + EIR) on the cell's images
  TPA breaks EIR  ASR_def(TPA-R) >= 0.50 and >= best non-adaptive reference + 0.20
  EIR holds       ASR_def(TPA-R) <= 0.30            otherwise: partial
  TPA advantage   ASR_def(TPA-R) - ASR_def(EOT-R) >= 0.15 (paired bootstrap 95% CI reported)
Descriptive: undefended agent ASR, how the defended decision was reached (re-answer on views / no observation ->
agent answer / re-answer unparsable -> agent answer).
"""
from __future__ import annotations

import json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

R = tio.R
D, M = "deepeyes_7b", "minio3_7b"
ADAPT = {m: {"TPA-R": ("PILOTR3", f"tpa_eir@{m}__eps8__seed0"), "EOT-R": ("PILOTR3", f"eot_eir@{m}__eps8__seed0")}
         for m in (D, M)}
REFS = {D: {"SV": ("S11", f"single_view@{D}__eps8__seed0"), "TPA": ("S11", f"tpa@{D}__eps8__seed0")},
        M: {"SV": ("PILOTL", f"single_view@{M}__eps8__seed0"),
            "SV-transfer(DE)": ("S11", f"single_view@{D}__eps8__seed0"),
            "TPA-transfer(DE)": ("S11", f"tpa@{D}__eps8__seed0")}}


def ci(xs, B=2000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed); n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def cell(m, ex, adv, qids):
    d = {r["qid"]: r for r in tio.read_jsonl(f"{R}/results/raw/PILOTR/eir__{m}__{ex}__{adv}.jsonl")}
    rs = [d[q] for q in qids if q in d]
    wrong = {r["qid"]: int(r["letter"] != r["answer"]) for r in rs}
    how = {}
    for r in rs:
        e = r["eir"]
        k = "re-answer" if e.get("applied") and not e.get("fallback") else ("no observation" if not e.get("applied")
                                                                             else "re-answer failed")
        how.setdefault(k, [0, 0]); how[k][0] += 1; how[k][1] += int(r["letter"] != r["answer"])
    return dict(n=len(rs), ASR_def=mean(list(wrong.values())), ASR_def_ci95=ci(list(wrong.values())),
                ASR_agent=mean([int(r["agent"]["letter"] != r["answer"]) for r in rs]),
                coverage=mean([int(r["agent"]["n_reobs"] >= 1) for r in rs]),
                how={k: dict(n=v[0], wrong=v[1]) for k, v in how.items()}), wrong


def main():
    out = {}
    for m in (D, M):
        qids = [l.strip() for l in open(f"{R}/code/configs/pilotr3_{m}_qids.txt") if l.strip()]
        cells, wr = {}, {}
        for name, (ex, adv) in list(ADAPT[m].items()) + list(REFS[m].items()):
            cells[name], wr[name] = cell(m, ex, adv, qids)
        tr, eo = cells["TPA-R"], cells["EOT-R"]
        refs = [cells[k]["ASR_def"] for k in REFS[m] if cells[k]["n"] and k in ("SV", "TPA")]
        best = max(refs) if refs else None
        common = [q for q in wr["TPA-R"] if q in wr["EOT-R"]]
        diff = [wr["TPA-R"][q] - wr["EOT-R"][q] for q in common]
        a = tr["ASR_def"]
        verdict = None
        if tr["n"]:
            if a >= 0.50 and best is not None and a >= best + 0.20:
                verdict = "TPA_BREAKS_EIR"
            elif a <= 0.30:
                verdict = "EIR_HOLDS"
            else:
                verdict = "PARTIAL"
        adv_gap = round(a - eo["ASR_def"], 3) if tr["n"] and eo["n"] else None
        out[m] = dict(cells=cells, best_nonadaptive_ASR_def=best, verdict=verdict, TPA_minus_EOT=adv_gap,
                      TPA_minus_EOT_ci95=ci(diff), n_paired=len(common),
                      TPA_advantage=adv_gap is not None and adv_gap >= 0.15)
    os.makedirs(f"{R}/results/PILOTR", exist_ok=True)
    json.dump(out, open(f"{R}/results/PILOTR/pilotr3_report.json", "w"), indent=1)
    for m, x in out.items():
        print(f"== {m}: verdict {x['verdict']}  best non-adaptive ASR_def {x['best_nonadaptive_ASR_def']}  "
              f"TPA-EOT {x['TPA_minus_EOT']} {x['TPA_minus_EOT_ci95']} (n={x['n_paired']}) advantage={x['TPA_advantage']}")
        for k, c in x["cells"].items():
            print(f"   {k:18s} n={c['n']:2d} ASR_def={c['ASR_def']} {c['ASR_def_ci95']} ASR_agent={c['ASR_agent']} "
                  f"cov={c['coverage']} how={c['how']}")


if __name__ == "__main__":
    main()
