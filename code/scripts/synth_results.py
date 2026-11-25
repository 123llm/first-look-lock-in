"""Synthetic results in the real schema (tpa/io.py), for testing the S9 analysis/table/figure pipeline only.
Items, qids and clusters come from the frozen splits; outcomes are random with planted effects of known direction
(so the analysis can be checked for recovering them).  Written under a separate root (default results_synth/);
never mixed with real results.

  synth_results.py --root ./results_synth --n 60
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

L = "ABCDEFGH"
TOOL = {"deepeyes_7b": "deepeyes_7b_tool_off", "minio3_7b": "minio3_7b_tool_off",
        "qwen3vl_4b_thinking": "qwen3vl_4b_thinking_tool_off"}
REASONING = ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "vision_r1_7b", "vlaa_thinker_7b"]


def rec(rng, it, ok_p, success_p, tool, reasoning, adv=False, budget=None):
    n_opt = len(it["options"])
    gt = L[it["answer_idx"]]
    wrong = [c for c in L[:n_opt] if c != gt]
    n_reobs = int(rng.integers(1, 4)) if tool else 0
    if budget:
        n_reobs = min(n_reobs, budget[1])
    correct = rng.random() < (1 - success_p if adv else ok_p)
    status = "ok" if rng.random() > 0.03 else str(rng.choice(["truncated", "max_turns"]))
    letter = (gt if correct else str(rng.choice(wrong))) if status == "ok" else None
    # probes: after o_0 and each re-observation
    if adv and not correct:
        p0 = str(rng.choice(wrong)) if rng.random() < 0.7 else gt
    elif adv:
        p0 = str(rng.choice(wrong)) if rng.random() < 0.3 else gt     # self-correction
    else:
        p0 = gt if rng.random() < 0.85 else str(rng.choice(wrong))
    probes = [p0] + [(letter or p0) if k == n_reobs - 1 else str(rng.choice([p0, letter or p0]))
                     for k in range(n_reobs)]
    text = "the region shows " + ("some noise and artifacts" if adv and reasoning and rng.random() < 0.15 else "objects")
    ntok = int(rng.integers(150, 900 if reasoning else 30)) if reasoning else int(rng.integers(1, 8))
    if adv:
        ntok = int(ntok * rng.uniform(0.9, 1.4))
    return dict(qid=it["qid"], answer=gt, letter=letter, status=status, n_tokens=ntok, n_calls=n_reobs,
                n_reobs=n_reobs, forced=bool(budget), probes=[dict(turn=i, letter=p) for i, p in enumerate(probes)],
                traj=dict(segments=[text], boxes=[None] * (n_reobs + 1)), t=0.0)


def write_cell(root, exp, split, items, eval_model, method, source, eps, p_succ, rng, clean_cache, tool, reasoning,
               budget=None, seed=0):
    path = f"{root}/raw/{exp}/{split}__{eval_model}__{method}@{source}__eps{eps}__seed{seed}.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = (split, eval_model)
    if key not in clean_cache:
        clean_cache[key] = {it["qid"]: rec(rng, it, 0.7, 0, tool, reasoning, budget=budget) for it in items}
    with open(path, "w") as f:
        for it in items:
            c = clean_cache[key][it["qid"]]
            a = rec(rng, it, 0.7, p_succ, tool, reasoning, adv=True, budget=budget)
            f.write(json.dumps(dict(qid=it["qid"], split=split, cluster=it["image_cluster"], eval_model=eval_model,
                                    method=method, source=source, eps=eps, seed=seed, png_sha256="synthetic",
                                    clean=c, adv=a, target=None, contaminated=False)) + "\n")
    json.dump(dict(protocol_sha256="synthetic", synthetic=True), open(path + ".meta.json", "w"))


def write_adv_index(root, exp, split, items, method, source, eps, rng, t_mean, fb, seed=0):
    d = f"{root}/adv/{exp}/{split}/{method}@{source}__eps{eps}__seed{seed}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/index.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(dict(qid=it["qid"], split=split, method=method, source=source, eps=eps, seed=seed,
                                    png="synthetic", png_sha256="synthetic", clean_sha256="synthetic", linf_255=eps,
                                    attack=dict(n_fwd_bwd=fb, n_grad_evals=fb // 2, n_rollouts=10 if "tpa" in method else 0),
                                    t=float(rng.normal(t_mean, t_mean * 0.1)), error=None)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{tio.R}/results_synth"); ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    R = a.root
    cc = {}
    S = {s: tio.load_split(s)[: a.n] for s in ["vstar_eval", "hrbench4k_eval", "mme_general_eval", "e1_subset",
                                                "e4b_eps4", "e6_budget", "e8_subset", "e7_mme_rs", "e7_xlrs",
                                                "e7_vrsbench_yesno"]}
    # E3 (H2a): single view, tool on vs off -- planted correction effect (on .35 vs off .60)
    for m, off in TOOL.items():
        for sp in ["vstar_eval", "hrbench4k_eval"]:
            write_cell(R, "E3", sp, S[sp], m, "single_view", m, 8, 0.35, rng, cc, True, True)
            write_cell(R, "E3", sp, S[sp], off, "single_view", m, 8, 0.60, rng, cc, False, True)
    # E4 (H2b): white-box main, eps 8; TPA on .75 / off .78
    base = {"tpa": 0.75, "single_view": 0.35, "eot": 0.45}
    for m in ["deepeyes_7b", "qwen3vl_4b_thinking"]:
        for sp in ["vstar_eval", "hrbench4k_eval", "mme_general_eval"]:
            for meth, p in base.items():
                write_cell(R, "E4", sp, S[sp], m, meth, m, 8, p, rng, cc, True, True)
                write_adv_index(R, "E4", sp, S[sp], meth, m, 8, rng, {"tpa": 630, "single_view": 86, "eot": 390}[meth],
                                {"tpa": 250, "single_view": 50, "eot": 150}[meth])
            write_cell(R, "E4", sp, S[sp], TOOL[m], "tpa", m, 8, 0.78, rng, cc, False, True)
            write_cell(R, "E4", sp, S[sp], TOOL[m], "single_view", m, 8, 0.60, rng, cc, False, True)
        for meth, src, p in [("mca_style", "clip3", 0.2), ("reasonbreak", "qwen3vl_4b_instruct", 0.1)]:
            for sp in ["vstar_eval", "hrbench4k_eval"]:
                write_cell(R, "E4", sp, S[sp], m, meth, src, 8, p, rng, cc, True, True)
        # transfer to other models (V*)
        for t, tool in [("minio3_7b", True), ("vision_r1_7b", False), ("vlaa_thinker_7b", False)]:
            for meth, p in [("tpa", 0.3), ("single_view", 0.2), ("eot", 0.25)]:
                write_cell(R, "E4", "vstar_eval", S["vstar_eval"], t, meth, m, 8, p, rng, cc, tool, True)
        # E4b eps 4
        for meth, p in [("tpa", 0.55), ("single_view", 0.25), ("eot", 0.3)]:
            write_cell(R, "E4b", "e4b_eps4", S["e4b_eps4"], m, meth, m, 4, p, rng, cc, True, True)
    for meth, src, p in [("mca_style", "clip3", 0.2), ("reasonbreak", "qwen3vl_4b_instruct", 0.1)]:
        for t, tool in [("minio3_7b", True), ("vision_r1_7b", False), ("vlaa_thinker_7b", False)]:
            write_cell(R, "E4", "vstar_eval", S["vstar_eval"], t, meth, src, 8, p, rng, cc, tool, True)
    # E1 (H1): 6 models x 4 attacks, RTAP rows + judged captions with an unrelated ordering
    e1_models = ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "vision_r1_7b", "vlaa_thinker_7b", "qwen25vl_7b"]
    e1_attacks = [("tpa", "deepeyes_7b", 0.6), ("single_view", "deepeyes_7b", 0.3), ("mca_style", "clip3", 0.2),
                  ("reasonbreak", "qwen3vl_4b_instruct", 0.1)]
    judge_p = {"tpa": 0.3, "single_view": 0.2, "mca_style": 0.6, "reasonbreak": 0.5}
    for m in e1_models:
        for meth, src, p in e1_attacks:
            write_cell(R, "E1", "e1_subset", S["e1_subset"], m, meth, src, 8, p, rng, cc, m in TOOL, m in REASONING)
            path = f"{R}/raw/E1/h1_judged/e1_subset__{m}__{meth}@{src}__eps8__seed0.jsonl"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                for it in S["e1_subset"]:
                    sc = int(rng.integers(1, 6)) if rng.random() < judge_p[meth] else int(rng.integers(6, 11))
                    f.write(json.dumps(dict(qid=it["qid"], cluster=it["image_cluster"], eval_model=m,
                                            adv=f"{meth}@{src}__eps8__seed0", caption_clean="a", caption_adv="b",
                                            score=sc, success={str(t): sc <= t for t in (3, 5, 7)})) + "\n")
            json.dump(dict(synthetic=True), open(path + ".meta.json", "w"))
    # E5 (H3): non-adaptive single view from the base model, evaluated on base and reasoning partner; VEG/RG rows
    pairs = [("qwen25vl_7b", "deepeyes_7b"), ("qwen25vl_7b", "minio3_7b"), ("qwen25vl_7b", "vision_r1_7b"),
             ("qwen25vl_7b", "vlaa_thinker_7b"), ("qwen3vl_4b_instruct", "qwen3vl_4b_thinking"),
             ("deepeyes_7b_tool_off", "deepeyes_7b")]
    for b, r in pairs:
        for sp in ["vstar_eval", "hrbench4k_eval"]:
            write_cell(R, "E5", sp, S[sp], b, "single_view", b, 8, 0.6, rng, cc, False, False)
            path = f"{R}/raw/E5/{sp}__{r}__single_view@{b}__eps8__seed0.jsonl"
            vpath = f"{R}/raw/E5/veg__{sp}__{r}.jsonl"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rows, vrows = [], []
            for it in S[sp]:
                rg = int(rng.integers(0, 4)) if r in TOOL else 0
                veg = int(rng.random() < 0.4)
                cot = float(rng.integers(100, 2000))
                ps = 1 / (1 + np.exp(-(0.4 - 0.5 * rg * veg)))
                c = rec(rng, it, 0.75, 0, r in TOOL, True)
                c.update(n_reobs=rg, n_calls=rg, n_tokens=int(cot))
                a2 = rec(rng, it, 0.75, ps, r in TOOL, True, adv=True)
                rows.append(dict(qid=it["qid"], split=sp, cluster=it["image_cluster"], eval_model=r,
                                 method="single_view", source=b, eps=8, seed=0, png_sha256="synthetic", clean=c,
                                 adv=a2, target=None, contaminated=False))
                vrows.append(dict(qid=it["qid"], cluster=it["image_cluster"], rg=rg,
                                  changed=bool(veg) if rg > 0 else None, cot_tokens=cot))
            for p_, rs in [(path, rows), (vpath, vrows)]:
                with open(p_, "w") as f:
                    for x in rs:
                        f.write(json.dumps(x) + "\n")
                json.dump(dict(synthetic=True), open(p_ + ".meta.json", "w"))
            # budget forcing intervention on the reasoning model (calls capped at 0/1)
            write_cell(R, "E5", sp, S[sp], f"{r}@b512c0", "single_view", b, 8, 0.7, rng, cc, False, True, budget=(512, 0))
    # E6 (H4): budget grid; non-adaptive success falls with budget, adaptive stays
    grid = {"qwen3vl_4b_thinking": [(512, 1), (1536, 3), (4096, 6)], "deepeyes_7b": [(1024, 1), (1024, 3), (1024, 6)],
            "vision_r1_7b": [(256, 0), (768, 0), (2048, 0)]}
    for m, cells in grid.items():
        for k, (tok, calls) in enumerate(cells):
            em = f"{m}@b{tok}c{calls}"
            src = m if m != "vision_r1_7b" else "qwen25vl_7b"
            write_cell(R, "E6", "e6_budget", S["e6_budget"], em, "single_view", src, 8, 0.6 - 0.15 * k, rng, cc,
                       m in TOOL, True, budget=(tok, calls))
            if m != "vision_r1_7b":
                write_cell(R, "E6", "e6_budget", S["e6_budget"], em, f"tpa_b{tok}c{calls}", m, 8, 0.7 - 0.02 * k,
                           rng, cc, True, True, budget=(tok, calls))
    # E7 remote sensing
    for sp in ["e7_mme_rs", "e7_xlrs", "e7_vrsbench_yesno"]:
        for meth, p in [("tpa", 0.7), ("single_view", 0.35)]:
            write_cell(R, "E7", sp, S[sp], "deepeyes_7b", meth, "deepeyes_7b", 8, p, rng, cc, True, True)
            write_cell(R, "E7", sp, S[sp], "qwen3vl_4b_thinking", meth, "deepeyes_7b", 8, p * 0.4, rng, cc, True, True)
    # E8 ablations + eps curve (Qwen3-Think white box)
    m = "qwen3vl_4b_thinking"
    for v, p, t in [("tpa", 0.75, 220), ("tpa_noprobe", 0.6, 160), ("tpa_noresample", 0.65, 120), ("tpa_K10", 0.76, 330),
                    ("tpa_K50", 0.66, 170), ("tpa_w128", 0.72, 190), ("tpa_w512", 0.76, 280)]:
        write_cell(R, "E8", "e8_subset", S["e8_subset"], m, v, m, 8, p, rng, cc, True, True)
        write_adv_index(R, "E8", "e8_subset", S["e8_subset"], v, m, 8, rng, t, 100)
    for eps, p in [(2, 0.2), (4, 0.5), (16, 0.95)]:
        for meth, f in [("tpa", 1.0), ("single_view", 0.5), ("eot", 0.6)]:
            write_cell(R, "E8", "e8_subset", S["e8_subset"], m, meth, m, eps, p * f, rng, cc, True, True)
    for meth, f in [("single_view", 0.5), ("eot", 0.6)]:
        write_cell(R, "E8", "e8_subset", S["e8_subset"], m, meth, m, 8, 0.75 * f, rng, cc, True, True)
    print("synthetic results written to", R)


if __name__ == "__main__":
    main()
