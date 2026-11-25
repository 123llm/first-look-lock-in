"""S7 numerical check: clean record -> single_view / EOT / TPA at eps 8 -> PNG round trip -> adversarial record ->
RTAP metrics; plus VEG intervention and one budget-forced rollout per item.  Uses calib_mme (the only split
allowed for tuning/inspection, protocol_v2 isolation).  Numbers here are code checks, never paper results.

usage: s7_check.py --model deepeyes_7b --n 5 --steps 50 [--K 10 --M 2]
"""
from __future__ import annotations

import argparse, io, json, os, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa.data.benchmarks import image_bytes
from tpa.models.qwen_agent import Sample, Trajectory, LETTERS
from tpa.models.registry import load_victim
from tpa.attacks.tpa import Attack, AttackConfig, save_png
from tpa.rollout.sampler import record_item
from tpa.eval.regrounding import veg_item, rollout_budget
from tpa.metrics import rtap

R = __import__("os").environ.get("TPA_ROOT", ".")
Image.MAX_IMAGE_PIXELS = None
MK = {"deepeyes_7b": (2, 10), "minio3_7b": (2, 10), "qwen3vl_4b_thinking": (1, 25)}


def load01(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--methods", default="single_view,eot,tpa")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = f"{R}/results/S7/{a.model}{a.tag}"
    os.makedirs(f"{out}/adv", exist_ok=True)
    ag = load_victim(a.model)
    M, K = MK[a.model]
    M, K = a.M or M, a.K or K
    mt, mnt = 6, ag.max_new_tokens
    items = json.load(open(f"{R}/code/configs/splits/calib_mme.json"))["items"]
    rows = {m: [] for m in a.methods.split(",")}
    fo = open(f"{out}/items.jsonl", "w")
    n_done = 0
    for it in items:
        if n_done >= a.n:
            break
        x = load01(image_bytes(it["dataset"], it["image_key"]))
        s = Sample(it["qid"], it["question"], it["options"], it["answer_idx"])
        clean = record_item(ag, x, s, mt, mnt)
        if clean["status"] != "ok" or clean["letter"] != clean["answer"]:
            print(it["qid"], "clean", clean["letter"], clean["answer"], clean["status"], "-> skipped (not in D)", flush=True)
            continue
        n_done += 1
        tr = Trajectory.from_dict(clean["traj"])
        v = veg_item(ag, x, s, tr, mt, mnt)
        bf = rollout_budget(ag, x, s, token_budget=256, call_cap=1, max_new_tokens=mnt)
        print(it["qid"], "clean", clean["letter"], clean["answer"], clean["status"], "reobs", clean["n_reobs"],
              "probes", [p["letter"] for p in clean["probes"]], f"{clean['t']}s",
              "| VEG", None if v is None else (v["letter"], v["changed"]),
              "| budget(256,1)", bf.final_letter, bf.status, bf.forced, bf.n_tokens, flush=True)
        rec = dict(qid=it["qid"], clean=clean, veg=v, budget=bf.to_dict(), attacks={})
        for m in rows:
            cfg = AttackConfig(method=m, eps=8, steps=a.steps, M=M, K=K, max_turns=mt, max_new_tokens=mnt)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            xa, log = Attack(ag, cfg).run(x, s)
            ta = time.time() - t0
            path = f"{out}/adv/{it['qid'].replace('/', '_')}__{m}.png"
            sha = save_png(xa, path)
            xr = load01(open(path, "rb").read())
            assert torch.equal(xr, xa), "PNG round trip changed the image"
            linf = float((xr - x).abs().max() * 255)
            adv = record_item(ag, xr, s, mt, mnt)
            J = [st["J"] for st in log["steps"]]
            print(f"  {m:11s} J {J[0]:.3f}->{J[-1]:.3f} (max {max(J):.3f}) linf {linf:.0f} "
                  f"adv {adv['letter']} {adv['status']} reobs {adv['n_reobs']} probes {[p['letter'] for p in adv['probes']]} "
                  f"| {ta:.0f}s (rollout {log['t_rollout']}s, grad {log['t_grad']}s, fwd/bwd {log['n_fwd_bwd']}, "
                  f"rollouts {log['n_rollouts']}) peak {log['peak_gb']} GB", flush=True)
            rec["attacks"][m] = dict(adv=adv, J=J, linf=linf, sha=sha, t=round(ta, 1),
                                     log={k: log[k] for k in ("n_fwd_bwd", "n_rollouts", "t_rollout", "t_grad", "peak_gb")},
                                     resamples=[r["summary"] for r in log["resamples"]])
            rows[m].append(dict(qid=it["qid"], cluster=it["image_cluster"], clean=clean, adv=adv, target=None))
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
    summ = {}
    for m, rs in rows.items():
        r = rtap.compute(rs, B=200)
        summ[m] = r
        print(m, "D", r["n_D"], {k: (round(v["point"], 3) if isinstance(v, dict) else v) for k, v in r.items()
                                 if isinstance(v, dict)}, flush=True)
    json.dump(summ, open(f"{out}/rtap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
