"""S6 smoke test for one victim (protocol_v2): load -> greedy rollout on clean items -> (tool white-box models)
gradient of the final-answer CE w.r.t. the input through every re-observation.

Records per item: status, parsed letter, correctness, tokens, tool calls, wall time, peak VRAM; per grad item:
grad norm > 0, teacher-forced argmax == rollout letter, time, peak VRAM.  Raw outputs go to
results/S6/smoke_<key>.jsonl (they also feed the 200-item parser audit).

usage: s6_smoke.py --model minio3_7b --split s11_pilot:15 --split mme_general_eval:15 --grad 3
"""
from __future__ import annotations

import argparse, io, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa.data.benchmarks import image_bytes
from tpa.models.qwen_agent import Sample, LETTERS
from tpa.models.registry import load_victim

R = os.environ.get("TPA_ROOT", ".")
Image.MAX_IMAGE_PIXELS = None


def to01(b: bytes, max_side: int | None = None) -> torch.Tensor:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0


def items_from(spec: str):
    name, n = spec.split(":")
    d = json.load(open(f"{R}/code/configs/splits/{name}.json"))
    return [dict(it, split=name) for it in d["items"][: int(n)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--grad", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--qids", default="")
    a = ap.parse_args()
    os.makedirs(f"{R}/results/S6", exist_ok=True)
    out_path = f"{R}/results/S6/smoke_{a.model}{a.tag}.jsonl"
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    ag = load_victim(a.model)
    load = {"load_s": round(time.time() - t0, 1), "mem_load_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "answer_open": ag.ANSWER_OPEN, "answer_prefix_text": ag.answer_prefix_text}
    print("loaded", load, flush=True)
    items = [it for s in a.split for it in items_from(s)]
    if a.qids:
        keep = set(a.qids.split(",")); items = [it for it in items if it["qid"] in keep]
    rows = []
    with open(out_path, "w") as f:
        for i, it in enumerate(items):
            x = to01(image_bytes(it["dataset"], it["image_key"]))
            s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                tr = ag.rollout(x, s, max_turns=a.max_turns, max_new_tokens=ag.max_new_tokens)
                err = None
            except Exception as e:      # recorded, not hidden: an OOM here is a smoke failure
                tr, err = None, f"{type(e).__name__}: {e}"[:300]
                torch.cuda.empty_cache()
            r = {"qid": it["qid"], "split": it["split"], "hw": list(x.shape[-2:]), "options": it["options"],
                 "answer": LETTERS[it["answer_idx"]], "error": err, "t": round(time.time() - t, 1),
                 "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            if tr is not None:
                r.update(status=tr.status, letter=tr.final_letter, correct=tr.final_letter == LETTERS[it["answer_idx"]],
                         n_tokens=tr.n_tokens, n_calls=tr.n_tool_calls, segments=tr.segments,
                         boxes=[list(b) if b else None for b in tr.boxes])
            if tr is not None and i < a.grad and ag.tool_mode and tr.status == "ok":
                ag.enable_grad_ckpt(); ag.tf_window = 256
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                try:
                    xg = x.to(ag.device).requires_grad_(True)
                    lg = ag.final_answer_logits(xg, s, tr)
                    loss = F.cross_entropy(lg[None], torch.tensor([it["answer_idx"]], device=lg.device))
                    loss.backward()
                    r["grad"] = {"norm": float(xg.grad.norm()), "nonzero": bool(xg.grad.abs().sum() > 0),
                                 "tf_argmax": LETTERS[int(lg.argmax())], "t": round(time.time() - t, 1),
                                 "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                except Exception as e:
                    r["grad"] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    torch.cuda.empty_cache()
                ag.tf_window = None
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(i, r["qid"], r.get("status"), r.get("letter"), r["answer"], r["t"], "s", r["peak_gb"], "GB",
                  "calls", r.get("n_calls"), "grad", r.get("grad"), r["error"] or "", flush=True)
    ok = [r for r in rows if r.get("status")]
    summ = dict(model=a.model, n=len(rows), **load,
                n_error=sum(r["error"] is not None for r in rows),
                status={k: sum(r.get("status") == k for r in rows) for k in ["ok", "truncated", "max_turns", "unparsable"]},
                acc=round(np.mean([r["correct"] for r in ok]), 3) if ok else None,
                mean_t=round(np.mean([r["t"] for r in rows]), 1), max_peak_gb=max(r["peak_gb"] for r in rows),
                mean_calls=round(np.mean([r["n_calls"] for r in ok]), 2) if ok else None,
                mean_tokens=round(np.mean([r["n_tokens"] for r in ok]), 0) if ok else None,
                n_malformed_calls=getattr(ag, "n_malformed", None),
                grad=[r["grad"] for r in rows if "grad" in r])
    json.dump(summ, open(f"{R}/results/S6/smoke_{a.model}{a.tag}_summary.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
