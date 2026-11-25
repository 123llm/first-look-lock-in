"""Model-level feasibility check.

For each of N V* samples:
  1. greedy rollout with the official HF processor pixels  (agent.use_diff = False)
  2. greedy rollout with the differentiable processor pixels (agent.use_diff = True)  -> agreement
  3. teacher-forced final-answer logits with grad wrt x01 through the original view AND every zoomed view;
     grad L1 inside the union of executed boxes vs outside; peak VRAM; time
  4. finite-difference sanity check of the gradient along a quantised sign direction
Writes results/S3/model_check_<tag>.json and results/S3/model_check_<tag>_traj.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F

from tpa.data.vstar import load_vstar, load_image01
from tpa.models.qwen_agent import QwenAgent, LETTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--no_tool", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--tf_window", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--fd_eps", type=float, default=2.0, help="FD step in uint8 levels")
    args = ap.parse_args()

    root = "."
    items = load_vstar(f"{root}/datasets/craigwu__vstar_bench")
    # spread over both splits
    sel = items[: args.n // 2] + items[115: 115 + args.n - args.n // 2]
    t0 = time.time()
    agent = QwenAgent(args.model, tool_mode=not args.no_tool, max_pixels=args.max_pixels)
    if args.grad_ckpt:
        agent.enable_grad_ckpt()
    agent.tf_window = args.tf_window
    load_s = time.time() - t0
    mem_load = torch.cuda.max_memory_allocated() / 2**30

    recs = []
    os.makedirs(f"{root}/results/S3", exist_ok=True)
    ftraj = open(f"{root}/results/S3/model_check_{args.tag}_traj.jsonl", "w")
    for s, path, _ in sel:
        x = load_image01(path)
        r = {"id": s.image_id, "hw": list(x.shape[-2:]), "answer": LETTERS[s.answer_idx]}
        torch.cuda.reset_peak_memory_stats()
        agent.use_diff = False
        t = time.time()
        tr_off = agent.rollout(x, s, max_new_tokens=args.max_new_tokens)
        r["t_rollout"] = time.time() - t
        agent.use_diff = True
        tr_dif = agent.rollout(x, s, max_new_tokens=args.max_new_tokens)
        r["mem_rollout_gb"] = torch.cuda.max_memory_allocated() / 2**30
        r.update(letter_off=tr_off.final_letter, letter_dif=tr_dif.final_letter,
                 same_text=tr_off.segments == tr_dif.segments, n_calls=tr_off.n_tool_calls,
                 n_tokens=tr_off.n_tokens, boxes=[b for b in tr_off.boxes if b is not None],
                 correct=tr_off.final_letter == LETTERS[s.answer_idx])
        ftraj.write(json.dumps({"id": s.image_id, "segments": tr_off.segments, "boxes": tr_off.boxes,
                                "final": tr_off.final_letter}) + "\n")
        ftraj.flush()
        # ---- gradient through all views of the official trajectory
        torch.cuda.reset_peak_memory_stats()
        xg = x.to(agent.device).clone().requires_grad_(True)
        t = time.time()
        try:
            logits = agent.final_answer_logits(xg, s, tr_off)
            loss = F.cross_entropy(logits[None], torch.tensor([s.answer_idx], device=logits.device))
            loss.backward()
            g = xg.grad.detach().abs().sum(0)
            mask = torch.zeros_like(g, dtype=torch.bool)
            for b in r["boxes"]:
                mask[b[1]:b[3], b[0]:b[2]] = True
            r.update(t_grad=time.time() - t, mem_grad_gb=torch.cuda.max_memory_allocated() / 2**30,
                     tf_letter=LETTERS[int(logits.detach().argmax())], loss=float(loss.detach()),
                     grad_l1=float(g.sum()), grad_l1_in_boxes=float(g[mask].sum()),
                     box_area_frac=float(mask.float().mean()), grad_finite=bool(torch.isfinite(g).all()))
            # ---- finite difference along a quantised sign direction
            d = (xg.grad.sign() * args.fd_eps / 255.0)
            x2 = ((xg.detach() + d) * 255).round().clamp(0, 255) / 255
            with torch.no_grad():
                l2 = F.cross_entropy(agent.final_answer_logits(x2, s, tr_off)[None],
                                     torch.tensor([s.answer_idx], device=logits.device))
            r.update(fd_actual=float(l2 - loss), fd_linear=float((xg.grad * (x2 - xg.detach())).sum()))
            # ---- gradient through the re-observations only (original view detached)
            if r["boxes"]:
                agent.zoom_grad_only = True
                xz = x.to(agent.device).clone().requires_grad_(True)
                lz = F.cross_entropy(agent.final_answer_logits(xz, s, tr_off)[None],
                                     torch.tensor([s.answer_idx], device=logits.device))
                lz.backward()
                gz = xz.grad.detach().abs().sum(0)
                r.update(zoom_grad_l1=float(gz.sum()), zoom_grad_l1_outside=float(gz[~mask].sum()))
                agent.zoom_grad_only = False
                del xz
        except torch.OutOfMemoryError as e:
            r.update(grad_error="OOM", mem_grad_gb=torch.cuda.max_memory_allocated() / 2**30)
            torch.cuda.empty_cache()
        except RuntimeError as e:
            r.update(grad_error=str(e)[:200])
            torch.cuda.empty_cache()
        del xg
        torch.cuda.empty_cache()
        recs.append(r)
        print(json.dumps(r), flush=True)

    ok = [r for r in recs if "grad_error" not in r]
    summ = {
        "model": args.model, "tag": args.tag, "n": len(recs), "max_pixels": args.max_pixels,
        "grad_ckpt": args.grad_ckpt, "tf_window": args.tf_window, "load_s": load_s, "mem_load_gb": mem_load,
        "agree_letter": sum(r["letter_off"] == r["letter_dif"] for r in recs) / len(recs),
        "agree_text": sum(r["same_text"] for r in recs) / len(recs),
        "acc_clean": sum(r["correct"] for r in recs) / len(recs),
        "mean_calls": sum(r["n_calls"] for r in recs) / len(recs),
        "tf_matches_rollout": (lambda u: (sum(r["tf_letter"] == r["letter_off"] for r in u) / len(u)) if u else None)(
            [r for r in ok if r["letter_off"] is not None]),
        "n_no_answer": sum(1 for r in recs if r["letter_off"] is None),
        "grad_errors": [r.get("grad_error") for r in recs if "grad_error" in r],
        "max_mem_rollout_gb": max(r["mem_rollout_gb"] for r in recs),
        "max_mem_grad_gb": max(r.get("mem_grad_gb", 0) for r in recs),
        "mean_t_rollout": sum(r["t_rollout"] for r in recs) / len(recs),
        "mean_t_grad": (sum(r["t_grad"] for r in ok) / len(ok)) if ok else None,
        "zoom_grad_nonzero": (sum(r.get("zoom_grad_l1", 0) > 0 for r in ok if r["boxes"]) / max(1, sum(1 for r in ok if r["boxes"]))) if ok else None,
        "fd_sign_agree": (lambda u: (sum((r["fd_actual"] > 0) == (r["fd_linear"] > 0) for r in u) / len(u)) if u else None)([r for r in ok if r["loss"] > 1e-6]),
        "fd_n_unsaturated": sum(1 for r in ok if r["loss"] > 1e-6),
        "records": recs,
    }
    with open(f"{root}/results/S3/model_check_{args.tag}.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "records"}, indent=1))


if __name__ == "__main__":
    main()
