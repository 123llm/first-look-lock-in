"""Mechanism / defense pilot report against the thresholds pre-registered in the pre-registration log (2026-11-05).

  pilotd_report.py            -> results/PILOTD/pilotd_report.json and a printed summary
Reads results/raw/PILOTD/*.jsonl, the S11 DeepEyes records and the clean cache.  Missing conditions are reported
as null (e.g. P4 before its images are generated and evaluated).
"""
from __future__ import annotations

import json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

R = tio.R
SV, TPA = "single_view@deepeyes_7b__eps8__seed0", "tpa@deepeyes_7b__eps8__seed0"
TPA_DRO = "tpa_dro@deepeyes_7b__eps8__seed0"


def load(name):
    p = f"{R}/results/raw/PILOTD/{name}.jsonl"
    return {r["qid"]: r for r in tio.read_jsonl(p)} if os.path.exists(p) else None


def ok(rec):
    return rec["letter"] == rec["answer"]


def ci(xs, B=2000, seed=0):
    """95% percentile bootstrap CI of a mean of 0/1 values."""
    if not xs:
        return None
    rng = random.Random(seed); n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]


def rate(xs):
    return dict(n=len(xs), rate=round(sum(xs) / len(xs), 3) if xs else None, ci95=ci(xs))


def main():
    s11 = {m: {r["qid"]: r for r in tio.read_jsonl(f"{R}/results/raw/S11/s11_pilot__deepeyes_7b__{m}.jsonl")}
           for m in (SV, TPA)}
    clean = {r["qid"]: r for r in tio.read_jsonl(f"{R}/results/raw/clean/s11_pilot__deepeyes_7b.jsonl")}
    out = {}

    # P1: clean-crop substitution on clean-correct, flipped items that zoomed at least once
    p1 = {}
    for m in (SV, TPA):
        base = [q for q, r in s11[m].items() if ok(r["clean"]) and not ok(r["adv"])]
        for cond in ("clean_zoom", "clean_zoom_rand"):
            d = load(f"{cond}__{m}")
            if d is None:
                continue
            elig = [q for q in base if q in d and d[q]["rec"]["n_reobs"] >= 1]
            p1[f"{cond}__{m.split('@')[0]}"] = dict(rate([int(ok(d[q]["rec"])) for q in elig]), n_flipped=len(base))
    sv, rd = p1.get(f"clean_zoom__single_view"), p1.get(f"clean_zoom_rand__single_view")
    if sv and rd:
        # paired difference on items eligible in both arms
        dz, dr = load(f"clean_zoom__{SV}"), load(f"clean_zoom_rand__{SV}")
        both = [q for q, r in s11[SV].items() if ok(r["clean"]) and not ok(r["adv"]) and q in dz and q in dr
                and dz[q]["rec"]["n_reobs"] >= 1 and dr[q]["rec"]["n_reobs"] >= 1]
        diff = [int(ok(dz[q]["rec"])) - int(ok(dr[q]["rec"])) for q in both]
        rng = random.Random(1); B = 2000
        bs = sorted(sum(rng.choice(diff) for _ in diff) / len(diff) for _ in range(B)) if diff else []
        p1["paired_diff_sv"] = dict(n=len(both), diff=round(sum(diff) / len(diff), 3) if diff else None,
                                    ci95=[round(bs[50], 3), round(bs[1949], 3)] if bs else None)
        p1["pass"] = bool(sv["rate"] is not None and sv["rate"] >= 0.30 and sv["rate"] - rd["rate"] >= 0.15)
    out["P1_mechanism"] = p1

    # P2: clean accuracy with DRO vs without (same 100 items)
    dc = load("dro__clean")
    if dc:
        qs = [q for q in dc if q in clean]
        a0 = [int(ok(clean[q])) for q in qs]; a1 = [int(ok(dc[q]["rec"])) for q in qs]
        out["P2_clean_cost"] = dict(n=len(qs), acc_nodef=round(sum(a0) / len(qs), 3),
                                    acc_dro=round(sum(a1) / len(qs), 3),
                                    drop_pp=round(100 * (sum(a0) - sum(a1)) / len(qs), 1),
                                    pass_=bool((sum(a0) - sum(a1)) / len(qs) <= 0.05))

    # P3: non-adaptive ASR with DRO; ASR = wrong on adversarial among clean-correct (no-defense clean, as in S11)
    p3 = {}
    for m in (SV, TPA):
        d = load(f"dro__{m}")
        if d is None:
            continue
        base = [q for q, r in s11[m].items() if ok(r["clean"]) and q in d]
        asr0 = [int(not ok(s11[m][q]["adv"])) for q in base]
        asr1 = [int(not ok(d[q]["rec"])) for q in base]
        # stricter variant: also require the item to be correct on clean with DRO
        base2 = [q for q in base if dc and q in dc and ok(dc[q]["rec"])]
        p3[m.split("@")[0]] = dict(n=len(base), asr_nodef=rate(asr0)["rate"], asr_dro=rate(asr1),
                                   drop_pp=round(100 * (sum(asr0) - sum(asr1)) / len(base), 1),
                                   asr_dro_cleancorrect_under_dro=rate([int(not ok(d[q]["rec"])) for q in base2]))
    if p3:
        p3["pass"] = all(v["drop_pp"] >= 20 for v in p3.values() if isinstance(v, dict))
    out["P3_nonadaptive"] = p3

    # P4: adaptive TPA through DRO (EOT), evaluated with DRO
    d4 = load(f"dro__{TPA_DRO}")
    if d4:
        qs = [q for q in d4 if q in clean and ok(clean[q])]
        x = [int(not ok(d4[q]["rec"])) for q in qs]
        qs2 = [q for q in qs if dc and q in dc and ok(dc[q]["rec"])]
        out["P4_adaptive"] = dict(asr=rate(x), asr_cleancorrect_under_dro=rate([int(not ok(d4[q]["rec"])) for q in qs2]),
                                  pass_=bool(sum(x) / len(x) <= 0.80) if x else None)
    else:
        out["P4_adaptive"] = None

    os.makedirs(f"{R}/results/PILOTD", exist_ok=True)
    json.dump(out, open(f"{R}/results/PILOTD/pilotd_report.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
