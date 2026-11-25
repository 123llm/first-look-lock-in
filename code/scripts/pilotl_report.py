"""First-glance lock-in pilot report (H-LI, pre-registered on 2026-11-05).

  pilotl_report.py   -> results/PILOTL/pilotl_report.json and a printed table

Per cell (victim x attack x eps), on items that are clean-correct, flipped and zoomed at least once in the standard
evaluation:
  R_clean  recovery under clean_zoom          R_rand  recovery under clean_zoom_rand (control)
  C_nat    natural correction rate: clean records of the same model and split whose turn-0 probe is wrong and that
           zoomed at least once, fraction answered correctly at the end
  lock     fraction of the flipped items whose turn-0 probe equals the final answer (descriptive)
Cell holds iff n >= 10, R_clean <= 0.10, R_clean - R_rand <= 0.10 and C_nat - R_clean >= 0.15.
GO iff, excluding DeepEyes eps=8 (the cells the hypothesis came from), cells hold for >= 2 models and at least one
holding cell has eps <= 4.  PARTIAL iff they hold for exactly one model.  NO otherwise.
"""
from __future__ import annotations

import json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

R = tio.R
Q, M, D = "qwen3vl_4b_thinking", "minio3_7b", "deepeyes_7b"
# (model, adv_exp, split, adv spec, role)   role: decide | describe
CELLS = [(D, "CALIB", "calib_mme", f"single_view@{D}__eps{e}__seed0", "decide") for e in (1, 2, 4)] + [
    (Q, "PILOTL", "s11_pilot", f"single_view@{Q}__eps4__seed0", "decide"),
    (Q, "PILOTL", "s11_pilot", f"single_view@{Q}__eps8__seed0", "decide"),
    (Q, "PILOTL", "s11_pilot", f"tpa@{Q}__eps8__seed0", "decide"),
    (M, "PILOTL", "s11_pilot", f"single_view@{M}__eps8__seed0", "decide"),
    (D, "S11", "s11_pilot", f"single_view@{D}__eps8__seed0", "describe"),
    (D, "S11", "s11_pilot", f"tpa@{D}__eps8__seed0", "describe"),
    (M, "S11", "s11_pilot", f"single_view@{D}__eps8__seed0", "describe"),   # transfer from DeepEyes
    (M, "S11", "s11_pilot", f"tpa@{D}__eps8__seed0", "describe"),
    (Q, "CALIB", "calib_mme", f"single_view@{Q}__eps1__seed0", "describe"),
    (Q, "CALIB", "calib_mme", f"single_view@{Q}__eps2__seed0", "describe"),
    (Q, "CALIB", "calib_mme", f"single_view@{Q}__eps4__seed0", "describe"),
]


def ok(rec):
    return rec["letter"] == rec["answer"]


def ci(xs, B=2000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed); n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def pilot_file(cond, model, adv_exp, adv):
    if model == D and adv_exp == "S11":                     # the earlier mechanism pilot (PILOTD naming)
        return f"{R}/results/raw/PILOTD/{cond}__{adv}.jsonl"
    return f"{R}/results/raw/PILOTL/{cond}__{model}__{adv_exp}__{adv}.jsonl"


def c_nat(model, split):
    rs = [r for r in tio.read_jsonl(tio.clean_path(split, model)) if r.get("precision", "nf4") == tio.PRECISION]
    el = [r for r in rs if r.get("probes") and r["probes"][0]["letter"] != r["answer"] and r["n_reobs"] >= 1]
    xs = [int(ok(r)) for r in el]
    return dict(n=len(xs), rate=mean(xs), ci95=ci(xs))


def main():
    rows, cnat = [], {}
    for model, exp, split, adv, role in CELLS:
        ev_path = tio.raw_path(exp, split, model, *tio.parse_adv_spec(adv))
        row = dict(model=model, exp=exp, split=split, adv=adv, role=role, eps=tio.parse_adv_spec(adv)[2])
        if not os.path.exists(ev_path):
            rows.append(dict(row, missing="evaluation")); continue
        ev = {r["qid"]: r for r in tio.read_jsonl(ev_path)}
        if exp == "PILOTL":                                   # pre-registered amendment: even-index subset
            keep = {l.strip() for l in open(f"{R}/code/configs/pilotl_qids.txt") if l.strip()}
            ev = {q: r for q, r in ev.items() if q in keep}
        fl = [q for q, r in ev.items() if ok(r["clean"]) and not ok(r["adv"]) and r["adv"]["n_reobs"] >= 1]
        cc = sum(ok(r["clean"]) for r in ev.values())
        row.update(n_items=len(ev), n_clean_correct=cc,
                   asr=round(sum(not ok(r["adv"]) for r in ev.values() if ok(r["clean"])) / cc, 3) if cc else None,
                   n_flipped_zoomed=len(fl))
        pr = [ev[q]["adv"].get("probes") for q in fl]
        row["lock"] = mean([int(p[0]["letter"] == ev[q]["adv"]["letter"]) for q, p in zip(fl, pr) if p])
        row["turn0_maxprob"] = mean([max(p[0]["probs"]) for p in pr if p])
        key = (model, split)
        if key not in cnat:
            cnat[key] = c_nat(model, split)
        row["C_nat"] = cnat[key]
        rec = {}
        for cond in ("clean_zoom", "clean_zoom_rand"):
            f = pilot_file(cond, model, exp, adv)
            d = {r["qid"]: r for r in tio.read_jsonl(f)} if os.path.exists(f) else {}
            xs = [int(ok(d[q]["rec"])) for q in fl if q in d and d[q]["rec"]["n_reobs"] >= 1]
            rec[cond] = dict(n=len(xs), rate=mean(xs), ci95=ci(xs), n_run=sum(q in d for q in fl))
        row["R_clean"], row["R_rand"] = rec["clean_zoom"], rec["clean_zoom_rand"]
        rc, rr, cn = rec["clean_zoom"]["rate"], rec["clean_zoom_rand"]["rate"], row["C_nat"]["rate"]
        n = rec["clean_zoom"]["n"]
        row["holds"] = (None if rc is None or rr is None or cn is None else
                        bool(n >= 10 and rc <= 0.10 and rc - rr <= 0.10 and cn - rc >= 0.15))
        row["decidable"] = n >= 10
        rows.append(row)

    dec = [r for r in rows if r["role"] == "decide" and r.get("holds") is not None]
    held = [r for r in dec if r["holds"]]
    models = {r["model"] for r in held}
    verdict = ("GO" if len(models) >= 2 and any(r["eps"] <= 4 for r in held) else
               "PARTIAL" if len(models) == 1 else "NO")
    pending = [r["adv"] + "|" + r["model"] for r in rows if r["role"] == "decide" and r.get("holds") is None]
    out = dict(verdict=verdict if not pending else f"{verdict} (pending: {len(pending)} cells)", pending=pending,
               holding_models=sorted(models), cells=rows)
    os.makedirs(f"{R}/results/PILOTL", exist_ok=True)
    json.dump(out, open(f"{R}/results/PILOTL/pilotl_report.json", "w"), indent=1)
    print(f"{'model':20s} {'cell':42s} {'role':8s} {'ASR':>5s} {'n':>3s} {'lock':>5s} {'p0':>5s} "
          f"{'R_cln':>6s} {'R_rnd':>6s} {'C_nat':>6s}(n) holds")
    for r in rows:
        if "missing" in r:
            print(f"{r['model']:20s} {r['exp'] + ':' + r['adv']:42s} {r['role']:8s} missing {r['missing']}"); continue
        f = lambda v: f"{v:6.3f}" if v is not None else "     -"
        print(f"{r['model']:20s} {(r['exp'] + ':' + r['adv'])[:42]:42s} {r['role']:8s} {r['asr'] or 0:5.2f} "
              f"{r['n_flipped_zoomed']:3d} {f(r['lock'])[1:]} {f(r['turn0_maxprob'])[1:]} {f(r['R_clean']['rate'])} "
              f"{f(r['R_rand']['rate'])} {f(r['C_nat']['rate'])}({r['C_nat']['n']}) {r['holds']}")
    print("verdict:", out["verdict"], "holding models:", out["holding_models"])


if __name__ == "__main__":
    main()
