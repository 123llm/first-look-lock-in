"""Mechanism pilot (pre-registered on 2026-11-06): visual vs textual lock-in channel.

  pilot_mech.py --model qwen3vl_4b_thinking --adv_exp PILOTL --adv single_view@qwen3vl_4b_thinking__eps8__seed0 \
                --qids_file code/configs/pilotl_qids.txt [--shard i/n]

Input: the clean_zoom trajectories of the lock-in pilot (zoomed views cropped from the clean image) on items that are
clean-correct, flipped and zoomed at least once.  For each item the answer is forced right after the last fed-back
observation (protocol_v2 probe) under 3 x 2 conditions, forward passes only:
  full view  adv | gray (0.5, same size) | clean          (zoomed views always cropped from the clean image)
  text       own (the trajectory's segments) | neutral (each zoom turn reduced to the model's own zoom action)
Output: results/raw/PILOTM/<model>__<adv_exp>__<adv>.jsonl, one row per item with all six conditions.
"""
from __future__ import annotations

import argparse, os, sys

import torch

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

FULLS = ("adv", "gray", "clean")
TEXTS = ("own", "neutral")
NEUTRAL_THINK = "Let me zoom in on this region to check the details."


def traj_file(model, adv_exp, adv, traj_exp="PILOTL"):
    if traj_exp != "PILOTL":                                   # formal runs: pilot_defense.py --out_exp <traj_exp>
        return f"{tio.R}/results/raw/{traj_exp}/clean_zoom__{model}__{adv_exp}__{adv}.jsonl"
    if model == "deepeyes_7b" and adv_exp == "S11":
        return f"{tio.R}/results/raw/PILOTD/clean_zoom__{adv}.jsonl"
    return f"{tio.R}/results/raw/PILOTL/clean_zoom__{model}__{adv_exp}__{adv}.jsonl"


def neutral_segment(ag, tr, k, x):
    """Zoom action of turn k without reasoning, in the victim's own format (same box, same source)."""
    b = tr.boxes[k]
    seg = tr.segments[k]
    cls = type(ag).__name__
    # v2 victims (2026-11-07): keep the model's own tool-call text verbatim, replace the reasoning
    if cls == "PixelReasonerAgent" and "<tool_call>" in seg:
        return NEUTRAL_THINK + "<tool_call>" + seg.split("<tool_call>")[-1]
    if cls == "GLMAgent" and "</think>" in seg:
        return "<think>" + NEUTRAL_THINK + "</think>" + seg.split("</think>")[-1]
    if getattr(ag, "BOX_HAS_SRC", False):                      # Mini-o3: relative coords of the source observation
        w, h = ag._obs_sizes(x, tr, k)[b[4]]
        bb = [round(b[0] / w, 3), round(b[1] / h, 3), round(b[2] / w, 3), round(b[3] / h, 3)]
        src = "original_image" if b[4] == 0 else f"observation_{b[4]}"
        return ('<think> %s </think> <grounding>{"bbox_2d": %s, "source": "%s"}</grounding>'
                % (NEUTRAL_THINK, bb, src))
    return ag.action_segment(b[:4], x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adv_exp", required=True)
    ap.add_argument("--adv", required=True); ap.add_argument("--split", default=None)
    ap.add_argument("--qids_file", default=""); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--traj_exp", default="PILOTL", help="experiment dir of the clean_zoom trajectories")
    ap.add_argument("--out_exp", default="PILOTM", help="results/raw/<out_exp>/<model>__<adv_exp>__<adv>.jsonl")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    split = a.split or ("calib_mme" if a.adv_exp == "CALIB" else "s11_pilot")
    ev = {r["qid"]: r for r in tio.read_jsonl(tio.raw_path(a.adv_exp, split, a.model, *tio.parse_adv_spec(a.adv)))}
    keep = {q for q, r in ev.items() if r["clean"]["letter"] == r["clean"]["answer"]
            and r["adv"]["letter"] != r["adv"]["answer"] and r["adv"]["n_reobs"] >= 1}
    if a.qids_file:
        keep &= {l.strip() for l in open(a.qids_file) if l.strip()}
    cz = {r["qid"]: r for r in tio.read_jsonl(traj_file(a.model, a.adv_exp, a.adv, a.traj_exp))}
    adv = tio.load_adv_index(a.adv, a.adv_exp, split, "")
    items = [it for it in tio.load_split(split) if it["qid"] in keep and it["qid"] in cz]
    items = [it for k, it in enumerate(items) if k % sn == si]
    out = f"{tio.R}/results/raw/{a.out_exp}/{a.model}__{a.adv_exp}__{a.adv}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {r["qid"] for r in tio.read_jsonl(out)}
    todo = [it for it in items if it["qid"] not in done]
    print(f"{out}: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    from tpa.models.registry import load_victim, REGISTRY
    from tpa.models.qwen_agent import Sample, Trajectory, LETTERS
    ag = load_victim(a.model)
    ag.tf_window = None
    tio.write_meta(out, split, {a.model: REGISTRY[a.model]["path"]}, stage="pilot_mech", adv=a.adv,
                   adv_exp=a.adv_exp, model=a.model, fulls=FULLS, texts=TEXTS, neutral_think=NEUTRAL_THINK)
    for it in todo:
        q = it["qid"]
        xc = tio.to01(tio.image_bytes_of(it))
        pb = open(f"{tio.R}/{adv[q]['png']}", "rb").read()
        assert tio.sha256(pb) == adv[q]["png_sha256"], "PNG changed since generation"
        xa = tio.to01(pb)
        tr = Trajectory.from_dict(cz[q]["rec"]["traj"])
        L = tr.obs_turns()[-1]
        row = dict(qid=q, cluster=it["image_cluster"], model=a.model, adv=a.adv, adv_exp=a.adv_exp,
                   answer=LETTERS[it["answer_idx"]], L=L, n_seg=len(tr.segments), status=cz[q]["rec"]["status"])
        if L == 0 or any(tr.boxes[k] is None for k in range(L)):
            row["skip"] = "no fed-back observation" if L == 0 else "turn without zoom before the last observation"
            tio.append_jsonl(out, row); print(q, row["skip"], flush=True); continue
        trn = Trajectory(segments=[neutral_segment(ag, tr, k, xc) for k in range(L)] + tr.segments[L:],
                         boxes=list(tr.boxes))
        s = Sample(q, it["question"], it["options"], it["answer_idx"])
        fulls = dict(adv=xa, gray=torch.full_like(xc, 0.5), clean=xc)
        conds = {}
        ag.zoom_source, ag.zoom_box_map, ag.zoom_transform = xc, None, None
        with torch.no_grad():
            for fn in FULLS:
                for tn in TEXTS:
                    lg = ag.probe_logits(fulls[fn].to(ag.device), s, tr if tn == "own" else trn, L)
                    p = torch.softmax(lg, -1)
                    conds[f"{fn}|{tn}"] = dict(letter=LETTERS[int(lg.argmax())], p_correct=round(float(p[it["answer_idx"]]), 4),
                                               probs=[round(float(v), 4) for v in p])
        ag.zoom_source = None
        row["conds"] = conds
        tio.append_jsonl(out, row)
        print(q, row["answer"], " ".join(f"{k}:{v['letter']}" for k, v in conds.items()), flush=True)


if __name__ == "__main__":
    main()
