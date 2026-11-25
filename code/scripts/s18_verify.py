"""S18 result acceptance: independent recomputation of the S12 headline numbers from item-level raw records.

  s18_verify.py [--adv_index_root DIR ...]  -> results/release/s18_v1/verify.json and verify.md

Written separately from the s12_*_report.py scripts (no shared helpers): each quantity is recomputed from
results/raw/** with its definition from the pre-registration log (2026-11-05 to 2026-11-07) and compared with the value in
results/E*/..._report.json.  Also checks every adversarial index.jsonl (last row per item): generation error, measured
L_inf <= eps, item counts, and collects per-item attack time / peak memory for the cost section.
"""
from __future__ import annotations

import argparse, glob, json, math, os, sys
from collections import Counter, defaultdict

R = os.environ.get("TPA_ROOT", ".")
RAW = f"{R}/results/raw"
MODELS = ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "qwen3vl_8b_thinking", "pixelreasoner_7b",
          "glm41v_9b_thinking"]
D = "deepeyes_7b"


def rows(path):
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line); out[r["qid"]] = r          # last row wins
    return out


def split_items(name, n=None):
    its = json.load(open(f"{R}/code/configs/splits/{name}.json"))["items"]
    return its[:n] if n else its


def right(rec):
    return rec["letter"] == rec["answer"]


def rate(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def lock_cell(exp, split, n, model, adv, clean_pool):
    items = split_items(split, n); qs = [it["qid"] for it in items]
    method, rest = adv.split("@"); src = rest.split("__")[0]
    ev = rows(f"{RAW}/{exp}/{split}__{model}__{adv}.jsonl")
    if ev is None:
        return dict(missing=True)
    ev = {q: ev[q] for q in qs if q in ev}
    for q, r in ev.items():
        clean_pool[model][q] = r["clean"]
    cc = [q for q in ev if right(ev[q]["clean"])]
    flipped = [q for q in cc if not right(ev[q]["adv"])]
    fz = [q for q in flipped if ev[q]["adv"]["n_reobs"] >= 1]
    status = Counter(("wrong_letter" if ev[q]["adv"]["status"] == "ok" and ev[q]["adv"]["letter"] else
                      ev[q]["adv"]["status"]) for q in flipped)
    out = dict(n_eval=len(ev), n_cc=len(cc), asr=rate([int(q in flipped) for q in cc]), n=len(fz),
               flipped_status=dict(status))
    lk = [int(ev[q]["adv"]["probes"][0]["letter"] == ev[q]["adv"]["letter"]) for q in fz if ev[q]["adv"].get("probes")]
    out["lock"] = rate(lk)
    for cond, key in (("clean_zoom", "R_clean"), ("clean_zoom_rand", "R_rand")):
        d = rows(f"{RAW}/{exp}/{cond}__{model}__{exp}__{adv}.jsonl") or {}
        v = [int(right(d[q]["rec"])) for q in fz if q in d and d[q]["rec"]["n_reobs"] >= 1]
        out[key] = rate(v); out[key + "_n"] = len(v); out[key + "_missing"] = sum(q not in d for q in fz)
    return out


def c_nat(pool):
    el = [r for r in pool.values() if r.get("probes") and r["probes"][0]["letter"] != r["answer"] and r["n_reobs"] >= 1]
    return dict(rate=rate([int(right(r)) for r in el]), n=len(el))


def holds(c, cn):
    if c.get("missing") or cn["n"] < 10 or c["n"] < 10 or c["R_clean"] is None:
        return False
    return bool(c["R_clean"] <= 0.10 and c["R_clean"] - c["R_rand"] <= 0.10 and cn["rate"] - c["R_clean"] >= 0.15)


def logit_fit(units):
    """y ~ x + model fixed effects by Newton-Raphson; models without variation in y are dropped."""
    import numpy as np
    by = defaultdict(list)
    for m, x, y in units:
        by[m].append((x, y))
    keep = [m for m, v in by.items() if 0 < sum(y for _, y in v) < len(v)]
    X, Y = [], []
    for m, x, y in units:
        if m in keep:
            X.append([x] + [1.0 if m == k else 0.0 for k in keep]); Y.append(y)
    X, Y = np.array(X), np.array(Y, float); b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        g = X.T @ (Y - p); H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g); b += step
        if np.abs(step).max() < 1e-10:
            break
    return round(float(b[0]), 3), keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adv_index_root", nargs="*", default=[])
    a = ap.parse_args()
    rep = {e: json.load(open(f)) for e, f in [("E1", f"{R}/results/E1/e1_report.json"), ("E3", f"{R}/results/E3/e3_report.json"),
                                               ("E4a", f"{R}/results/E4/e4a_report.json"), ("E5", f"{R}/results/E5/e5_report.json"),
                                               ("E7", f"{R}/results/E7/e7_report.json")]}
    out, checks = {}, []

    def check(name, mine, theirs, tol=0.0015):
        ok = (mine is None and theirs is None) or (mine is not None and theirs is not None and abs(mine - theirs) <= tol)
        checks.append(dict(name=name, recomputed=mine, report=theirs, match=bool(ok)))

    # E1
    pool = defaultdict(dict); e1 = {}
    cells = [(m, f"single_view@{m}__eps{e}__seed0") for m in MODELS for e in (8, 4)] + \
            [(m, f"single_view@{D}__eps8__seed0") for m in MODELS if m != D]
    for m, adv in cells:
        e1[f"{m}|{adv}"] = lock_cell("E1", "v2_main", 150, m, adv, pool)
    cn = {m: c_nat(pool[m]) for m in MODELS}
    for k, c in e1.items():
        m, adv = k.split("|"); c["C_nat"] = cn[m]; c["holds"] = holds(c, cn[m])
        rc = next(x for x in rep["E1"]["cells"] if x["model"] == m and x["adv"] == adv)
        check(f"E1 {k} ASR", c["asr"], rc["asr"]["rate"]); check(f"E1 {k} R_clean", c["R_clean"], rc["R_clean"]["rate"])
        check(f"E1 {k} R_rand", c["R_rand"], rc["R_rand"]["rate"]); check(f"E1 {k} n", c["n"], rc["R_clean"]["n"])
        check(f"E1 {k} C_nat", c["C_nat"]["rate"], rc["C_nat"]["rate"])
        check(f"E1 {k} holds", float(c["holds"]), float(bool(rc.get("holds"))))
    out["E1"] = e1

    # E4 adaptive
    e4 = {}
    for m in ("deepeyes_7b", "minio3_7b", "pixelreasoner_7b"):
        qs = [l.strip() for l in open(f"{R}/code/configs/e4a_{m}_qids.txt") if l.strip()]
        files = {"TPA-R": f"E4/eir__{m}__E4__tpa_eir@{m}__eps8__seed0", "TPA-R+J": f"E4/eir__{m}__E4__tpa_eir_j15@{m}__eps8__seed0",
                 "EOT-R": f"E4/eir__{m}__E4__eot_eir@{m}__eps8__seed0", "non-adaptive": f"E4/eir__{m}__E1__single_view@{m}__eps8__seed0"}
        e4[m] = {}
        for att, f in files.items():
            d = rows(f"{RAW}/{f}.jsonl") or {}
            v = [int(not right(d[q])) for q in qs if q in d]
            e4[m][att] = dict(asr_def=rate(v), n=len(v), missing=sum(q not in d for q in qs),
                              agent_asr=rate([int(d[q]["agent"]["letter"] != d[q]["answer"]) for q in qs if q in d]))
            check(f"E4a {m} {att}", e4[m][att]["asr_def"], rep["E4a"][m]["attacks"][att]["ASR_def"]["rate"])
    out["E4a"] = e4

    # E5
    e5 = {}
    for m in ("deepeyes_7b", "minio3_7b"):
        e5[m] = {}
        for meth in ("single_view", "eot", "tpa"):
            ev = rows(f"{RAW}/E5/v2_main__{m}__{meth}@{m}__eps2__seed0.jsonl") or {}
            qs = [it["qid"] for it in split_items("v2_main", 100)]
            cc = [q for q in qs if q in ev and right(ev[q]["clean"])]
            e5[m][meth] = dict(asr=rate([int(not right(ev[q]["adv"])) for q in cc]), n_cc=len(cc),
                               n_eval=sum(q in ev for q in qs))
    out["E5"] = e5

    # E7
    pool7 = defaultdict(dict); e7 = {}
    for m in ("deepeyes_7b", "minio3_7b"):
        for meth in ("single_view", "tpa"):
            e7[f"{m}|{meth}"] = lock_cell("E7", "e7_xlrs", 100, m, f"{meth}@{m}__eps4__seed0", pool7)
    for k, c in e7.items():
        m = k.split("|")[0]; c["C_nat"] = c_nat(pool7[m]); c["holds"] = holds(c, c["C_nat"])
    out["E7"] = e7

    # E3: units, recoveries, slope
    lv = {"s2": ("E3", "sv_s2"), "s5": ("E3", "sv_s5"), "s10": ("E3", "sv_s10"), "s25": ("E3", "sv_s25"), "s50": ("E1", "single_view")}
    units = []; e3 = {}
    clip = lambda c: min(max(c, 1e-4), 1 - 1e-4)
    for m in ("deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking"):
        levels = [(k, e, f"{p}@{m}__eps8__seed0") for k, (e, p) in lv.items()]
        if m != D:
            levels.append(("transfer", "E1", f"single_view@{D}__eps8__seed0"))
        for k, e, adv in levels:
            ev = rows(f"{RAW}/{e}/v2_main__{m}__{adv}.jsonl") or {}
            qs = [it["qid"] for it in split_items("v2_main", 150)]
            cz = rows(f"{RAW}/{e}/clean_zoom__{m}__{e}__{adv}.jsonl") or {}
            n = rec = 0
            for q in qs:
                r = ev.get(q)
                if not r or not right(r["clean"]) or right(r["adv"]) or r["adv"]["n_reobs"] < 1:
                    continue
                if q not in cz or cz[q]["rec"]["n_reobs"] < 1 or not r["adv"].get("probes"):
                    continue
                c = clip(max(r["adv"]["probes"][0]["probs"])); y = int(right(cz[q]["rec"]))
                units.append((m, math.log(c / (1 - c)), y)); n += 1; rec += y
            e3[f"{m}|{k}"] = dict(n=n, recovered=rec)
    slope, kept = logit_fit(units)
    out["E3"] = dict(levels=e3, units=len(units), recovered=sum(u[2] for u in units), slope=slope, models_in_fit=kept)
    check("E3 units", len(units), rep["E3"]["n_units"], 0)
    check("E3 recovered", sum(u[2] for u in units), rep["E3"]["n_recovered"], 0)
    check("E3 slope", slope, rep["E3"].get("slope"), 0.002)

    # adversarial indices: errors, L_inf, cost
    adv = {}
    roots = a.adv_index_root + [f"{R}/results/adv"]
    for root in roots:
        for f in sorted(glob.glob(f"{root}/E[1-7]/*/*/index.jsonl")):
            key = "/".join(f.split("/")[-4:-1])
            last = rows(f)
            eps = int(key.split("__eps")[1].split("__")[0])
            errs = [q for q, r in last.items() if r.get("error")]
            linf = [r.get("linf_255") for r in last.values() if r.get("error") is None]
            over = [x for x in linf if x is None or x > eps + 1e-3]
            ts = sorted(r["t"] for r in last.values() if r.get("error") is None and isinstance(r.get("t"), (int, float)))
            gb = [r["peak_gb"] for r in last.values() if r.get("error") is None and isinstance(r.get("peak_gb"), (int, float))]
            ent = dict(root=root, n=len(last), n_err=len(errs), linf_max=max([x for x in linf if x is not None], default=None),
                       linf_over=len(over), t_median_s=ts[len(ts) // 2] if ts else None, peak_gb_max=max(gb, default=None))
            if key not in adv or adv[key]["n"] - adv[key]["n_err"] < ent["n"] - ent["n_err"]:
                adv[key] = ent
    out["adv_index"] = adv
    out["checks"] = checks
    out["summary"] = dict(n_checks=len(checks), n_mismatch=sum(not c["match"] for c in checks),
                          adv_cells=len(adv), adv_cells_with_errors=sum(v["n_err"] > 0 for v in adv.values()),
                          adv_cells_linf_over=sum(v["linf_over"] > 0 for v in adv.values()))
    od = f"{R}/results/release/s18_v1"; os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/verify.json", "w"), indent=1)
    print(json.dumps(out["summary"]))
    for c in checks:
        if not c["match"]:
            print("MISMATCH", c)
    for k, v in adv.items():
        if v["n_err"] or v["linf_over"]:
            print("ADV", k, v)


if __name__ == "__main__":
    main()
