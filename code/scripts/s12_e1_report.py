"""E1 lock-in generality report (rules pre-registered on 2026-11-07).

  s12_e1_report.py   -> results/E1/e1_report.json and a printed table

Cells: 6 white-box models x single-view eps in {4, 8} (decide) and DeepEyes eps8 single-view images on the other
five models (describe).  Items: v2_main first 150.  Per cell, on items that are clean-correct, flipped and zoomed at
least once in the standard evaluation:
  R_clean  recovery under clean_zoom          R_rand  recovery under clean_zoom_rand (control)
  C_nat    clean records of the same model (same 150 items, as embedded in its evaluation rows) whose turn-0 probe is wrong and that zoomed at least
           once, fraction answered correctly at the end
Cell holds iff n >= 10, R_clean <= 0.10, R_clean - R_rand <= 0.10 and C_nat - R_clean >= 0.15; C_nat with n < 10
makes the model's cells undecidable, counted as not holding.
Model holds iff at least one of its white-box cells holds.
I2: >= 4 holding models with >= 2 of them holding at eps 4 -> SUPPORTED; 2-3 holding models -> PARTIAL;
<= 1 -> NOT SUPPORTED.  (>= 4 holding models but < 2 at eps 4 does not meet SUPPORTED and is reported as PARTIAL.)
CIs: cluster bootstrap over image_cluster, 2000 resamples, seed 0.
"""
from __future__ import annotations

import json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

R = tio.R
SPLIT, N, EXP = "v2_main", 150, "E1"
MODELS = ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "qwen3vl_8b_thinking", "pixelreasoner_7b",
          "glm41v_9b_thinking"]
D = "deepeyes_7b"
CELLS = [(m, f"single_view@{m}__eps{e}__seed0", "decide") for m in MODELS for e in (8, 4)] + \
        [(m, f"single_view@{D}__eps8__seed0", "describe") for m in MODELS if m != D]


def ok(rec):
    return rec["letter"] == rec["answer"]


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def cluster_ci(pairs, B=2000, seed=0):
    """pairs: [(cluster, value)].  Percentile CI of the pooled mean under cluster resampling."""
    if not pairs:
        return None
    by = {}
    for c, v in pairs:
        by.setdefault(c, []).append(v)
    cl = list(by); rng = random.Random(seed); ms = []
    for _ in range(B):
        s = n = 0
        for c in (rng.choice(cl) for _ in cl):
            s += sum(by[c]); n += len(by[c])
        ms.append(s / n)
    ms.sort()
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)]


def stat(pairs):
    xs = [v for _, v in pairs]
    return dict(n=len(xs), rate=mean(xs), ci95=cluster_ci(pairs))


def first_qids():
    return [it["qid"] for it in tio.load_split(SPLIT)[:N]]


def c_nat_from(clean_recs, cluster):
    """C_nat over the model's clean records (the ones embedded in its evaluation rows, i.e. exactly those used)."""
    rs = list(clean_recs.values())
    el = [r for r in rs if r.get("probes") and r["probes"][0]["letter"] != r["answer"] and r["n_reobs"] >= 1]
    return dict(stat([(cluster[r["qid"]], int(ok(r))) for r in el]), n_clean=len(rs),
                clean_acc=mean([int(ok(r)) for r in rs]))


def cell(model, adv, role, exp, split, keep, cluster, clean_of, n_expect):
    """One lock-in cell (E1 rule).  clean_of[model] accumulates the model's clean records across its cells."""
    method, src, eps, seed = tio.parse_adv_spec(adv)
    row = dict(model=model, adv=adv, role=role, eps=eps, transfer=src != model)
    ev_path = tio.raw_path(exp, split, model, method, src, eps, seed)
    if not os.path.exists(ev_path):
        return dict(row, missing="evaluation")
    ev = {r["qid"]: r for r in tio.read_jsonl(ev_path) if r["qid"] in keep}
    if len(ev) < n_expect:
        row["incomplete_eval"] = len(ev)
    clean_of.setdefault(model, {}).update({q: r["clean"] for q, r in ev.items()})
    cc = [q for q, r in ev.items() if ok(r["clean"])]
    fl = [q for q in cc if not ok(ev[q]["adv"]) and ev[q]["adv"]["n_reobs"] >= 1]
    row.update(n_items=len(ev), n_clean_correct=len(cc),
               asr=stat([(cluster[q], int(not ok(ev[q]["adv"]))) for q in cc]),
               zoom_adv=mean([int(r["adv"]["n_reobs"] >= 1) for r in ev.values()]),
               zoom_clean=mean([int(r["clean"]["n_reobs"] >= 1) for r in ev.values()]),
               n_flipped_zoomed=len(fl))
    pr = [(q, ev[q]["adv"].get("probes")) for q in fl]
    row["lock"] = mean([int(p[0]["letter"] == ev[q]["adv"]["letter"]) for q, p in pr if p])
    row["turn0_maxprob"] = mean([max(p[0]["probs"]) for _, p in pr if p])
    rec = {}
    for cond in ("clean_zoom", "clean_zoom_rand"):
        f = f"{R}/results/raw/{exp}/{cond}__{model}__{exp}__{adv}.jsonl"
        d = {r["qid"]: r for r in tio.read_jsonl(f)} if os.path.exists(f) else {}
        pairs = [(cluster[q], int(ok(d[q]["rec"]))) for q in fl if q in d and d[q]["rec"]["n_reobs"] >= 1]
        rec[cond] = dict(stat(pairs), n_run=sum(q in d for q in fl))
    row["R_clean"], row["R_rand"] = rec["clean_zoom"], rec["clean_zoom_rand"]
    row["_complete"] = (rec["clean_zoom"]["n_run"] == len(fl) and rec["clean_zoom_rand"]["n_run"] == len(fl)
                        and "incomplete_eval" not in row)
    return row


def judge(row, cn):
    """Apply the cell rule once C_nat of the model is known (cn: c_nat_from(...) or None)."""
    if "missing" in row:
        return row
    row["C_nat"] = cn
    rc, rr, n = row["R_clean"]["rate"], row["R_rand"]["rate"], row["R_clean"]["n"]
    if not row.pop("_complete"):
        row["holds"] = None
    elif cn is None or cn["n"] < 10:
        row["holds"] = False; row["undecidable"] = "C_nat n < 10"
    elif n < 10:
        row["holds"] = False; row["undecidable"] = "n < 10"
    else:
        row["holds"] = bool(rc <= 0.10 and rc - rr <= 0.10 and cn["rate"] - rc >= 0.15)
    return row


def main():
    items = tio.load_split(SPLIT)[:N]
    keep = {it["qid"] for it in items}; cluster = {it["qid"]: it["image_cluster"] for it in items}
    clean_of = {}
    rows = [cell(m, adv, role, EXP, SPLIT, keep, cluster, clean_of, N) for m, adv, role in CELLS]
    cnat = {m: c_nat_from(c, cluster) for m, c in clean_of.items()}
    rows = [judge(r, cnat.get(r["model"])) for r in rows]

    dec = [r for r in rows if r["role"] == "decide"]
    pending = [r["adv"] + "|" + r["model"] for r in dec if r.get("holds") is None]
    per_model = {}
    for m in MODELS:
        cs = [r for r in dec if r["model"] == m]
        per_model[m] = dict(holds=any(r.get("holds") for r in cs),
                            eps_holding=sorted(r["eps"] for r in cs if r.get("holds")),
                            pending=any(r.get("holds") is None for r in cs))
    held = [m for m, v in per_model.items() if v["holds"]]
    held4 = [m for m in held if 4 in per_model[m]["eps_holding"]]
    verdict = ("SUPPORTED" if len(held) >= 4 and len(held4) >= 2 else
               "PARTIAL" if len(held) >= 2 else "NOT SUPPORTED")
    out = dict(verdict=verdict if not pending else f"{verdict} (pending: {len(pending)} cells)", pending=pending,
               holding_models=held, holding_models_eps4=held4, per_model=per_model, cells=rows)
    os.makedirs(f"{R}/results/E1", exist_ok=True)
    json.dump(out, open(f"{R}/results/E1/e1_report.json", "w"), indent=1)
    f = lambda v: f"{v:6.3f}" if v is not None else "     -"
    print(f"{'model':20s} {'cell':38s} {'role':8s} {'ASR':>6s} {'n':>3s} {'lock':>6s} {'p0':>6s} "
          f"{'R_cln':>6s} {'R_rnd':>6s} {'C_nat':>6s}(n) holds")
    for r in rows:
        if "missing" in r:
            print(f"{r['model']:20s} {r['adv']:38s} {r['role']:8s} missing {r['missing']}"); continue
        cn = r["C_nat"] or {"rate": None, "n": 0}
        print(f"{r['model']:20s} {r['adv'][:38]:38s} {r['role']:8s} {f(r['asr']['rate'])} {r['n_flipped_zoomed']:3d} "
              f"{f(r['lock'])} {f(r['turn0_maxprob'])} {f(r['R_clean']['rate'])} {f(r['R_rand']['rate'])} "
              f"{f(cn['rate'])}({cn['n']}) {r['holds']}{' [' + r['undecidable'] + ']' if r.get('undecidable') else ''}")
    print("verdict:", out["verdict"], "| holding:", held, "| eps4:", held4)


if __name__ == "__main__":
    main()
