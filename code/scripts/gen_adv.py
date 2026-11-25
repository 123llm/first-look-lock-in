"""Stage 1: generate adversarial PNGs for one (experiment, split, method, source, eps, seed) cell.  Resumable:
items already in index.jsonl are skipped.  Never evaluates (evaluation is eval_adv.py).

  gen_adv.py --exp E4 --split vstar_eval --method tpa --source deepeyes_7b --eps 8
  gen_adv.py --exp E4 --split vstar_eval --method mca_style --source clip3 --eps 8
  gen_adv.py --exp E4 --split vstar_eval --method reasonbreak --source qwen3vl_4b_instruct --eps 8
Options: --limit N, --qids a,b, --steps/--M/--K/--lam/--no_probes/--no_resample (E8), --token_budget/--call_cap (E6),
         --target lowest_clean_prob (E8 targeted).
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile, time, traceback

import torch

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

MK = {"deepeyes_7b": (2, 10), "minio3_7b": (2, 10), "qwen3vl_4b_thinking": (1, 25),   # protocol_v2 attack.tpa
      "pixelreasoner_7b": (2, 10), "qwen3vl_8b_thinking": (1, 25), "glm41v_9b_thinking": (1, 25)}   # v2 victims (2026-11-07)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--method", required=True, choices=["tpa", "single_view", "eot", "mca_style", "reasonbreak"])
    ap.add_argument("--source", required=True); ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qids", default=""); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--M", type=int, default=None); ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--tf_window", type=int, default=256)
    ap.add_argument("--no_probes", action="store_true"); ap.add_argument("--no_resample", action="store_true")
    ap.add_argument("--token_budget", type=int, default=None); ap.add_argument("--call_cap", type=int, default=None)
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--save_steps", default="", help="E3: also save the image after these PGD step counts, "
                    "e.g. 2,5,10,25 (<cell>/steps/<qid>__s<n>.png; the final image is unchanged)")
    ap.add_argument("--target", default=None, choices=[None, "lowest_clean_prob"])
    ap.add_argument("--variant", default="", help="suffix for ablation cells, e.g. _noprobe")
    ap.add_argument("--qids_file", default="", help="restrict to the qids listed in this file (one per line)")
    ap.add_argument("--shard", default="0/1", help="i/n: generate items with index %% n == i (one process per GPU, "
                    "all appending to the same index.jsonl)")
    ap.add_argument("--defense", default="none", choices=["none", "dro"],
                    help="dro: attack the victim equipped with decorrelated re-observation (tpa.defenses.reobs); "
                         "a new random draw per rollout and per gradient forward (EOT)")
    ap.add_argument("--remedy", default="none", choices=["none", "eir", "eirt"],
                    help="attack the agent defended by evidence-isolated re-answering (tpa.defenses.eir); "
                         "method name gets _<remedy>")
    ap.add_argument("--box_jitter", type=float, default=0.0,
                    help="TPA-R+J (H-OP check): jitter every EIR-term zoom box each step; use with --variant")
    a = ap.parse_args()
    assert a.remedy == "none" or a.method in ("tpa", "eot"), "--remedy applies to tpa / eot"
    si, sn = map(int, a.shard.split("/"))

    method_name = a.method + ("" if a.remedy == "none" else f"_{a.remedy}") + a.variant
    out = tio.adv_dir(a.exp, a.split, method_name, a.source, a.eps, a.seed)
    os.makedirs(out, exist_ok=True)
    index = f"{out}/index.jsonl"
    last = {r["qid"]: r for r in tio.read_jsonl(index)}
    done = {q for q, r in last.items() if r.get("error") is None}   # failed items are retried; last row wins
    items = tio.load_split(a.split)
    if a.qids:
        keep = set(a.qids.split(",")); items = [it for it in items if it["qid"] in keep]
    if a.limit:
        items = items[: a.limit]
    if a.qids_file:
        keep = {l.strip() for l in open(a.qids_file) if l.strip()}
        items = [it for it in items if it["qid"] in keep]
    items = [it for k, it in enumerate(items) if k % sn == si]
    todo = [it for it in items if it["qid"] not in done]
    # Scheduling only (the attack RNG is seeded per item, so order and deferral change no image): an item whose last
    # attempt ran out of the per-process cap (OOM while the device still had the memory) at a cap >= the current one
    # fails again at the same step, so it is left for a run with a larger TPA_VRAM_GB; never-tried items go first.
    capped = {it["qid"] for it in todo if tio.cap_oom_gb((last.get(it["qid"]) or {}).get("error")) >= tio.VRAM_GB - 1e-6}
    todo = ([it for it in todo if it["qid"] not in last] +
            [it for it in todo if it["qid"] in last and it["qid"] not in capped])
    print(f"{out}: {len(done)} done, {len(todo)} to do"
          + (f", {len(capped)} deferred (hit the {tio.VRAM_GB:g} GB cap)" if capped else ""), flush=True)
    if not todo:
        if capped:
            sys.exit(3)
        return

    from tpa.models.registry import REGISTRY
    models = {}
    if a.method in ("tpa", "single_view", "eot"):
        from tpa.models.registry import load_victim
        from tpa.attacks.tpa import Attack, AttackConfig
        ag = load_victim(a.source)
        models[a.source] = REGISTRY[a.source]["path"]
        if a.defense == "dro":
            from tpa.defenses.reobs import DecorrelatedReobs
            ag.zoom_transform = DecorrelatedReobs(seed=a.seed)
            ag.zoom_transform.resample_per_forward = True
        if a.remedy != "none":
            from tpa.defenses.eir import make_eir
            ag.eir = make_eir(ag, a.remedy)
        M, K = MK.get(a.source, (2, 10))
        cfg = AttackConfig(method=a.method, eps=a.eps, steps=a.steps, M=a.M or M, K=a.K or K, lam=a.lam,
                           tf_window=a.tf_window or None, seed=a.seed, max_turns=a.max_turns,
                           max_new_tokens=ag.max_new_tokens, token_budget=a.token_budget, call_cap=a.call_cap,
                           use_probes=not a.no_probes, resample=not a.no_resample,
                           remedy=None if a.remedy == "none" else a.remedy, box_jitter=a.box_jitter,
                           save_steps=tuple(int(v) for v in a.save_steps.split(",") if v))
    elif a.method == "mca_style":
        from tpa.attacks.mca import MCAStyle, SURROGATES, MODEL_ROOT
        mca = MCAStyle(eps=a.eps, steps=a.steps, seed=a.seed)
        models.update({k: f"{MODEL_ROOT}/{v}" for k, v in SURROGATES.items()})
    else:
        from tpa.models.registry import load_victim
        from tpa.attacks.reasonbreak import ReasonBreak, make_chain, RB_WEIGHTS
        chain_agent = load_victim(a.source)           # chain generator (attacker side)
        rb = ReasonBreak(eps=a.eps)
        models.update({a.source: REGISTRY[a.source]["path"], "reasonbreak": os.path.dirname(RB_WEIGHTS)})
    tio.write_meta(out, a.split, models, method=method_name, source=a.source, eps=a.eps, seed=a.seed,
                   args=vars(a))

    from tpa.attacks.tpa import save_png
    from tpa.models.qwen_agent import Sample
    n_err = len(capped)
    for it in todo:
        b = tio.image_bytes_of(it)
        x = tio.to01(b)
        s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
        png = f"{out}/{tio.qid_file(it['qid'])}.png"
        t0 = time.time()
        row = dict(qid=it["qid"], split=a.split, method=method_name, source=a.source, eps=a.eps, seed=a.seed,
                   png=os.path.relpath(png, tio.R), clean_sha256=tio.sha256(b))
        try:
            torch.cuda.reset_peak_memory_stats()
            if a.method in ("tpa", "single_view", "eot"):
                if a.target:
                    # protocol: lowest clean probability wrong option from the clean greedy final-answer distribution
                    from tpa.rollout.sampler import record_item
                    from tpa.models.qwen_agent import Trajectory
                    rec = record_item(ag, x, s, a.max_turns, ag.max_new_tokens, probes=False)
                    tr = Trajectory.from_dict(rec["traj"])
                    with torch.no_grad():
                        lg = (ag.final_answer_logits(x.to(ag.device), s, tr) if ag.tool_mode and tr.boxes[-1] is None
                              else ag.probe_logits(x.to(ag.device), s, tr, tr.obs_turns()[-1]))
                    p = torch.softmax(lg.float(), -1)
                    p[s.answer_idx] = 2.0
                    cfg.target_idx = int(p.argmin())
                    row["target"] = "ABCDEFGH"[cfg.target_idx]
                atk = Attack(ag, cfg)
                xa, log = atk.run(x, s)
                if atk.snapshots:
                    os.makedirs(f"{out}/steps", exist_ok=True)
                    row["step_pngs"] = {str(n): save_png(v, f"{out}/steps/{tio.qid_file(it['qid'])}__s{n}.png")
                                        for n, v in sorted(atk.snapshots.items())}
                J = [st["J"] for st in log["steps"]]
                row["attack"] = dict(J_first=J[0], J_last=J[-1], J_max=max(J), J=J, n_fwd_bwd=log["n_fwd_bwd"],
                                     n_grad_evals=log["n_grad_evals"], n_rollouts=log["n_rollouts"],
                                     t_rollout=log["t_rollout"], t_grad=log["t_grad"], peak_gb=log["peak_gb"],
                                     resamples=[r["summary"] for r in log["resamples"]],
                                     trajs=[r["trajs"] for r in log["resamples"]] if a.method == "tpa" else None,
                                     eot_boxes=log["boxes"] or None)
            elif a.method == "mca_style":
                xa, log = mca.run(x, it["qid"])
                row["attack"] = dict(sim_first=log["steps"][0]["sim"], sim_last=log["steps"][-1]["sim"],
                                     n_fwd_bwd=log["n_fwd_bwd"], peak_gb=log["peak_gb"])
            else:
                ch = make_chain(chain_agent, x, it["question"], it["options"])
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    from PIL import Image
                    Image.open(__import__("io").BytesIO(b)).convert("RGB").save(f, format="PNG")
                    tmp = f.name
                try:
                    xa = rb.run(tmp, ch["chain"])
                finally:
                    os.unlink(tmp)
                row["attack"] = dict(chain=ch["chain"], chain_fallback=ch["fallback"], chain_raw=ch["raw"][:3000])
            row["png_sha256"] = save_png(xa, png)
            xr = tio.to01(open(png, "rb").read())
            row["linf_255"] = round(float((xr - x).abs().max() * 255), 3)
            assert row["linf_255"] <= a.eps + 1e-3, f"linf {row['linf_255']} > eps"
            row["error"] = None
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:500]
            traceback.print_exc()
            torch.cuda.empty_cache()
        row["t"] = round(time.time() - t0, 1)
        row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        tio.append_jsonl(index, row)
        print(it["qid"], "linf", row.get("linf_255"), f"{row['t']}s", row["peak_gb"], "GB", row["error"] or "",
              flush=True)
        n_err += row["error"] is not None
    if n_err:
        # non-zero exit so gpu_pool --retries reruns the command; finished items are skipped, failed ones retried
        print(f"{n_err} item(s) failed in this run", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
