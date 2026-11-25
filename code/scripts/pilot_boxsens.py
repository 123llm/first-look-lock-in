"""H-OP check 2 (descriptive, pre-registered on 2026-11-06): zoom-box shift sensitivity of the EIR
re-answer on TPA-R images.  Forward passes only.

  pilot_boxsens.py --model deepeyes_7b|minio3_7b [--shard i/n]

For every R3 item, the trajectories of the last resample of the TPA-R attack (the boxes the perturbation was last
optimised for) that fed back at least one observation (Mini-o3: only trajectories whose boxes all come from the
original image) are re-used; every zoom box is shifted by s x box size along +x, -x, +y, -y (s in 0, 5, 10, 20 %;
shifted boxes are moved back inside the image) and the EIR direct-answer logits are computed on the adversarial image.
Output: results/raw/PILOTR3/boxsens__<model>.jsonl (one row per item x trajectory with all shifts).
"""
from __future__ import annotations

import argparse, os, sys

import torch

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

SHIFTS = (0.0, 0.05, 0.10, 0.20)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def shift_box(b, s, d, h, w):
    x1, y1, x2, y2 = b[:4]
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(round(d[0] * s * bw)), int(round(d[1] * s * bh))
    nx1 = min(max(x1 + dx, 0), w - bw)
    ny1 = min(max(y1 + dy, 0), h - bh)
    return (nx1, ny1, nx1 + bw, ny1 + bh) + tuple(b[4:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    cell = f"tpa_eir@{a.model}__eps8__seed0"
    idx = {r["qid"]: r for r in tio.read_jsonl(f"{tio.R}/results/adv/PILOTR3/s11_pilot/{cell}/index.jsonl")
           if r.get("error") is None}
    qids = [l.strip() for l in open(f"{tio.R}/code/configs/pilotr3_{a.model}_qids.txt") if l.strip()]
    items = {it["qid"]: it for it in tio.load_split("s11_pilot")}
    todo_q = [q for k, q in enumerate(qids) if k % sn == si and q in idx]
    out = f"{tio.R}/results/raw/PILOTR3/boxsens__{a.model}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["qid"], r["traj_i"]) for r in tio.read_jsonl(out)}
    from tpa.models.registry import load_victim, REGISTRY
    from tpa.models.qwen_agent import Sample, Trajectory, LETTERS
    from tpa.defenses.eir import make_eir
    ag = load_victim(a.model)
    eir = make_eir(ag, "eir")
    tio.write_meta(out, "s11_pilot", {a.model: REGISTRY[a.model]["path"]}, stage="pilot_boxsens", shifts=SHIFTS)
    for q in todo_q:
        r = idx[q]; it = items[q]
        pb = open(f"{tio.R}/{r['png']}", "rb").read()
        assert tio.sha256(pb) == r["png_sha256"], "PNG changed since generation"
        x = tio.to01(pb).to(ag.device)
        h, w = x.shape[-2:]
        s = Sample(q, it["question"], it["options"], it["answer_idx"])
        last = r["attack"]["trajs"][-1] or []
        for ti, td in enumerate(last):
            if (q, ti) in done:
                continue
            tr = Trajectory.from_dict(td)
            L = tr.obs_turns()[-1]
            row = dict(qid=q, traj_i=ti, model=a.model, answer=LETTERS[s.answer_idx], n_reobs=tr.n_reobs)
            if tr.n_reobs < 1 or any(b is not None and len(b) > 4 and b[4] != 0 for b in tr.boxes[:L]):
                row["skip"] = "no observation" if tr.n_reobs < 1 else "box from a zoomed observation"
                tio.append_jsonl(out, row); continue
            res = {}
            with torch.no_grad():
                for sh in SHIFTS:
                    for d in (DIRS[:1] if sh == 0 else DIRS):
                        t2 = Trajectory(segments=list(tr.segments), status=tr.status,
                                        boxes=[None if b is None else shift_box(b, sh, d, h, w) for b in tr.boxes])
                        lg = eir.answer_logits(x, s, t2)
                        p = torch.softmax(lg.float(), -1)
                        res[f"{sh}|{d[0]},{d[1]}"] = dict(letter=LETTERS[int(lg.argmax())],
                                                          p_correct=round(float(p[s.answer_idx]), 4))
            row["res"] = res
            tio.append_jsonl(out, row)
            print(q, ti, " ".join(f"{k}:{v['letter']}" for k, v in res.items()), "ans", row["answer"], flush=True)


if __name__ == "__main__":
    main()
