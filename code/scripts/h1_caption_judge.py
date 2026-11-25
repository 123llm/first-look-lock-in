"""H1 reference protocol (captioning + open judge), two resumable stages so that the victim and the judge are
never on the GPU together.

  h1_caption_judge.py caption --exp E1 --split e1_subset --adv tpa@deepeyes_7b__eps8__seed0 --eval_model deepeyes_7b
      -> results/raw/<E>/h1_captions/<split>__<eval_model>__<adv>.jsonl  (clean captions cached per model)
  h1_caption_judge.py judge --exp E1 --judge qwen3vl_8b_instruct
      -> adds results/raw/<E>/h1_judged/<same name>.jsonl with score and success at thresholds 3/5/7
"""
from __future__ import annotations

import argparse, glob, os, sys

sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

JUDGES = {"qwen3vl_8b_instruct": __import__("os").environ.get("TPA_ROOT", ".") + "/models/Qwen__Qwen3-VL-8B-Instruct",
          "qwen3vl_4b_instruct": __import__("os").environ.get("TPA_ROOT", ".") + "/models/Qwen__Qwen3-VL-4B-Instruct"}


def caption_stage(a):
    from tpa.models.registry import load_victim, REGISTRY
    from tpa.eval.caption_judge import caption
    items = tio.load_split(a.split)
    adv = tio.load_adv_index(a.adv, a.exp, a.split, a.adv_from)
    items = [it for it in items if it["qid"] in adv and adv[it["qid"]].get("error") is None]
    cdir = f"{tio.R}/results/raw/{a.exp}/h1_captions"
    cclean = f"{cdir}/clean__{a.split}__{a.eval_model}.jsonl"
    out = f"{cdir}/{a.split}__{a.eval_model}__{a.adv}.jsonl"
    have_c = {r["qid"]: r for r in tio.read_jsonl(cclean)}
    done = {r["qid"] for r in tio.read_jsonl(out)}
    todo = [it for it in items if it["qid"] not in done]
    if not todo:
        return
    ag = load_victim(a.eval_model)
    tio.write_meta(out, a.split, {a.eval_model: REGISTRY[a.eval_model]["path"]}, stage="caption", adv=a.adv)
    for it in todo:
        if it["qid"] not in have_c:
            c = caption(ag, tio.to01(tio.image_bytes_of(it)))
            have_c[it["qid"]] = dict(qid=it["qid"], **c)
            tio.append_jsonl(cclean, have_c[it["qid"]])
        pb = open(f"{tio.R}/{adv[it['qid']]['png']}", "rb").read()
        c = caption(ag, tio.to01(pb))
        tio.append_jsonl(out, dict(qid=it["qid"], cluster=it["image_cluster"], eval_model=a.eval_model, adv=a.adv,
                                   caption_clean=have_c[it["qid"]]["caption"], caption_adv=c["caption"],
                                   truncated=c["truncated"] or have_c[it["qid"]]["truncated"]))
        print(it["qid"], len(c["caption"]), flush=True)


def judge_stage(a):
    from tpa.models.qwen_agent import QwenAgent
    from tpa.eval.caption_judge import judge, judge_success
    files = sorted(glob.glob(f"{tio.R}/results/raw/{a.exp}/h1_captions/*.jsonl"))
    files = [f for f in files if not os.path.basename(f).startswith("clean__")]
    jag = None
    for f in files:
        out = f.replace("/h1_captions/", "/h1_judged/")
        done = {r["qid"] for r in tio.read_jsonl(out)}
        rows = [r for r in tio.read_jsonl(f) if r["qid"] not in done]
        if not rows:
            continue
        if jag is None:
            jag = QwenAgent(JUDGES[a.judge], tool_mode=False, answer_mode="instruct")
            jag.key, jag.max_new_tokens = a.judge, 8
        tio.write_meta(out, a.split or "n/a", {a.judge: JUDGES[a.judge]}, stage="judge", judge=a.judge)
        for r in rows:
            j = judge(jag, r["caption_clean"], r["caption_adv"])
            r.update(judge=a.judge, score=j["score"], judge_raw=j["raw"],
                     success={str(t): judge_success(j["score"], t) for t in (3, 5, 7)})
            tio.append_jsonl(out, r)
        print(out, len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["caption", "judge"])
    ap.add_argument("--exp", required=True); ap.add_argument("--split", default=None)
    ap.add_argument("--adv", default=None); ap.add_argument("--eval_model", default=None)
    ap.add_argument("--adv_from", default="")
    ap.add_argument("--judge", default="qwen3vl_8b_instruct")
    a = ap.parse_args()
    caption_stage(a) if a.stage == "caption" else judge_stage(a)


if __name__ == "__main__":
    main()
