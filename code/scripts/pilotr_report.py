"""Remedy pilot R1/R2 report (pre-registered on 2026-11-06).

  pilotr_report.py  -> results/PILOTR/pilotr_report.json and a printed table

R1 per model and variant: dA = A_def(clean) - A_agent(clean) over all clean items of the model; acceptable iff >= -0.10.
R2 per decide cell: on items clean-correct and flipped under the agent, Rec = fraction correct after the remedy
(no-observation items keep the agent's answer); cell passes iff n >= 10 and Rec >= 0.30.
A model passes iff R1 is acceptable and at least half of its decide cells pass.  Variant for R3: eir unless eirt
passes for more models.  R3 runs iff at least one model passes with the chosen variant.
"""
from __future__ import annotations

import json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

R = tio.R
Q, M, D = "qwen3vl_4b_thinking", "minio3_7b", "deepeyes_7b"
CLEAN = {D: ["s11_pilot", "calib_mme"], Q: ["s11_pilot"], M: ["s11_pilot"]}
QIDS = {l.strip() for l in open(f"{R}/code/configs/pilotl_qids.txt") if l.strip()}
# (model, adv_exp, split, adv, role, restrict to pilotl_qids)
CELLS = [(D, "S11", "s11_pilot", f"single_view@{D}__eps8__seed0", "decide", False),
         (D, "S11", "s11_pilot", f"tpa@{D}__eps8__seed0", "decide", False)] + [
        (D, "CALIB", "calib_mme", f"single_view@{D}__eps{e}__seed0", "decide", False) for e in (1, 2, 4)] + [
        (Q, "PILOTL", "s11_pilot", f"single_view@{Q}__eps4__seed0", "decide", True),
        (Q, "PILOTL", "s11_pilot", f"single_view@{Q}__eps8__seed0", "decide", True),
        (Q, "PILOTL", "s11_pilot", f"tpa@{Q}__eps8__seed0", "decide", True),
        (M, "PILOTL", "s11_pilot", f"single_view@{M}__eps8__seed0", "decide", True),
        (M, "S11", "s11_pilot", f"single_view@{D}__eps8__seed0", "describe", False),
        (M, "S11", "s11_pilot", f"tpa@{D}__eps8__seed0", "describe", False)]


def ci(xs, B=2000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed); n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def rows(path, qids=None):
    d = {r["qid"]: r for r in tio.read_jsonl(path)}
    return {q: r for q, r in d.items() if qids is None or q in qids}


def ok(r, agent=False):
    return (r["agent"]["letter"] if agent else r["letter"]) == r["answer"]


def main():
    out = dict(variants={})
    for v in ("eir", "eirt"):
        clean = {}
        r1 = {}
        for m, splits in CLEAN.items():
            rs = []
            for sp in splits:
                c = rows(f"{R}/results/raw/PILOTR/clean__{v}__{m}__{sp}.jsonl", QIDS if m == Q else None)
                clean[(m, sp)] = c
                rs += list(c.values())
            a_ag, a_df = [int(ok(r, True)) for r in rs], [int(ok(r)) for r in rs]
            dA = round(mean(a_df) - mean(a_ag), 3) if rs else None
            r1[m] = dict(n=len(rs), A_agent=mean(a_ag), A_def=mean(a_df), dA=dA,
                         dA_ci95=ci([x - y for x, y in zip(a_df, a_ag)]), acceptable=dA is not None and dA >= -0.10,
                         applied=mean([int(r["eir"].get("applied", False)) for r in rs]),
                         fallback_reanswer=sum(bool(r["eir"].get("applied") and r["eir"].get("fallback")) for r in rs))
        cells = {}
        for m, ex, sp, adv, role, qf in CELLS:
            name = f"{m}__{ex}__{adv}"
            adv_rows = rows(f"{R}/results/raw/PILOTR/{v}__{m}__{ex}__{adv}.jsonl", QIDS if qf else None)
            c = clean[(m, sp)]
            common = [q for q in adv_rows if q in c]
            fl = [q for q in common if ok(c[q], True) and not ok(adv_rows[q], True)]
            rec = [int(ok(adv_rows[q])) for q in fl]
            zoomed = [q for q in fl if adv_rows[q]["agent"]["n_reobs"] >= 1]
            dc = [q for q in common if ok(c[q])]
            ac = [q for q in common if ok(c[q], True)]
            n = len(fl)
            cells[name] = dict(role=role, n_items=len(common), n_flipped=n, Rec=mean(rec), Rec_ci95=ci(rec),
                               coverage=mean([int(adv_rows[q]["agent"]["n_reobs"] >= 1) for q in fl]),
                               Rec_zoomed=mean([int(ok(adv_rows[q])) for q in zoomed]), n_zoomed=len(zoomed),
                               ASR_agent=mean([int(not ok(adv_rows[q], True)) for q in ac]),
                               ASR_def=mean([int(not ok(adv_rows[q])) for q in dc]),
                               passes=(n >= 10 and mean(rec) >= 0.30) if role == "decide" else None)
        models = {}
        for m in CLEAN:
            dec = [c for k, c in cells.items() if k.startswith(m + "__") and c["role"] == "decide"]
            npass = sum(bool(c["passes"]) for c in dec)
            models[m] = dict(R1_acceptable=r1[m]["acceptable"], decide_cells=len(dec), cells_passing=npass,
                             passes=bool(r1[m]["acceptable"] and dec and npass * 2 >= len(dec)))
        out["variants"][v] = dict(R1=r1, cells=cells, models=models,
                                  passing=[m for m, x in models.items() if x["passes"]])
    pe, pt = out["variants"]["eir"]["passing"], out["variants"]["eirt"]["passing"]
    chosen = "eirt" if len(pt) > len(pe) else "eir"
    out.update(chosen_variant=chosen, r3_models=out["variants"][chosen]["passing"],
               verdict="RUN_R3" if out["variants"][chosen]["passing"] else "REMEDY_FAILS")
    os.makedirs(f"{R}/results/PILOTR", exist_ok=True)
    json.dump(out, open(f"{R}/results/PILOTR/pilotr_report.json", "w"), indent=1)
    for v, d in out["variants"].items():
        print(f"== {v}")
        for m, x in d["R1"].items():
            print(f"  R1 {m:22s} n={x['n']:3d} A_agent={x['A_agent']} A_def={x['A_def']} dA={x['dA']} {x['dA_ci95']} "
                  f"ok={x['acceptable']} applied={x['applied']}")
        for k, c in d["cells"].items():
            print(f"  R2 {k[:62]:62s} {c['role'][:3]} n={c['n_flipped']:3d} Rec={c['Rec']} {c['Rec_ci95']} cov={c['coverage']} "
                  f"Rec|z={c['Rec_zoomed']} ASR ag={c['ASR_agent']} def={c['ASR_def']} pass={c['passes']}")
        print("  models:", {m: x["passes"] for m, x in d["models"].items()})
    print("chosen:", chosen, "R3 models:", out["r3_models"], "verdict:", out["verdict"])


if __name__ == "__main__":
    main()
