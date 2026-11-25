"""Remedy pilot R1/R2 (pre-registered on 2026-11-06): evidence-isolated re-answering on top of the
agent trajectories already recorded by eval_adv.py (no new agent rollouts; one re-answer per record).

  pilot_remedy.py --model deepeyes_7b --variant eir --split s11_pilot                      (clean records, R1)
  pilot_remedy.py --model deepeyes_7b --variant eir --adv_exp S11 --split s11_pilot \
                  --adv single_view@deepeyes_7b__eps8__seed0                                (adversarial records, R2)
  [--qids_file code/configs/pilotl_qids.txt] [--shard i/n]

Output: results/raw/<out_exp>/clean__<variant>__<model>__<split>.jsonl  or
        results/raw/<out_exp>/<variant>__<model>__<adv_exp>__<adv>.jsonl
row: qid, answer, agent (letter, status, n_reobs), defended letter and the re-answer details.
"""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--variant", default="eir", choices=["eir", "eirt"])
    ap.add_argument("--split", required=True); ap.add_argument("--adv_exp", default="")
    ap.add_argument("--adv", default=""); ap.add_argument("--qids_file", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out_exp", default="PILOTR", help="output dir results/raw/<out_exp>/")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    if a.adv:
        recs = {r["qid"]: r for r in tio.read_jsonl(tio.raw_path(a.adv_exp, a.split, a.model, *tio.parse_adv_spec(a.adv)))}
        adv = tio.load_adv_index(a.adv, a.adv_exp, a.split, "")
        out = f"{tio.R}/results/raw/{a.out_exp}/{a.variant}__{a.model}__{a.adv_exp}__{a.adv}.jsonl"
    else:
        recs = {}
        for r in tio.read_jsonl(tio.clean_path(a.split, a.model)):
            if r.get("precision", "nf4") == tio.PRECISION:
                recs.setdefault(r["qid"], r)
        out = f"{tio.R}/results/raw/{a.out_exp}/clean__{a.variant}__{a.model}__{a.split}.jsonl"
    keep = {l.strip() for l in open(a.qids_file) if l.strip()} if a.qids_file else None
    items = [it for it in tio.load_split(a.split) if it["qid"] in recs and (keep is None or it["qid"] in keep)
             and (not a.adv or it["qid"] in adv)]          # records whose PNG is no longer in the index are skipped
    items = [it for k, it in enumerate(items) if k % sn == si]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {r["qid"] for r in tio.read_jsonl(out)}
    todo = [it for it in items if it["qid"] not in done]
    print(f"{out}: {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    from tpa.models.registry import load_victim, REGISTRY
    from tpa.models.qwen_agent import Sample, Trajectory
    from tpa.defenses.eir import make_eir
    ag = load_victim(a.model)
    eir = make_eir(ag, a.variant)
    tio.write_meta(out, a.split, {a.model: REGISTRY[a.model]["path"]}, stage="pilot_remedy", variant=a.variant,
                   adv=a.adv, adv_exp=a.adv_exp, model=a.model)
    import time
    for it in todo:
        q = it["qid"]
        if a.adv:
            pb = open(f"{tio.R}/{adv[q]['png']}", "rb").read()
            assert tio.sha256(pb) == adv[q]["png_sha256"], "PNG changed since generation"
            rec = recs[q]["adv"]
        else:
            pb = tio.image_bytes_of(it)
            rec = recs[q]
            assert rec.get("image_sha256") in (None, tio.sha256(pb)), "clean image changed"
        x = tio.to01(pb)
        tr = Trajectory.from_dict(rec["traj"])
        assert tr.final_letter == rec["letter"], "stored trajectory does not match the record"
        s = Sample(q, it["question"], it["options"], it["answer_idx"])
        t0 = time.time()
        d = eir.answer(x, s, tr)
        row = dict(qid=q, cluster=it["image_cluster"], model=a.model, variant=a.variant, adv=a.adv or None,
                   answer=rec["answer"], agent=dict(letter=rec["letter"], status=rec["status"], n_reobs=tr.n_reobs),
                   letter=d.pop("letter"), eir=d, t=round(time.time() - t0, 2))
        tio.append_jsonl(out, row)
        print(q, row["answer"], "agent", rec["letter"], "def", row["letter"], d.get("eir_status"), d.get("fallback"),
              f"{row['t']}s", flush=True)


if __name__ == "__main__":
    main()
