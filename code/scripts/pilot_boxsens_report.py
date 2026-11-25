"""Summary of H-OP check 2 (descriptive): EIR error rate on TPA-R images vs zoom-box shift.

  pilot_boxsens_report.py  -> results/PILOTR/boxsens_report.json and a printed table
Error rate = fraction of (item, trajectory, direction) whose EIR direct answer is wrong; trajectories pooled over items;
CI = bootstrap over items.
"""
from __future__ import annotations

import json, os, random, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

SHIFTS = ("0.0", "0.05", "0.1", "0.2")


def main():
    out = {}
    for m in ("deepeyes_7b", "minio3_7b"):
        rows = [r for r in tio.read_jsonl(f"{tio.R}/results/raw/PILOTR3/boxsens__{m}.jsonl") if "res" in r]
        by_item = {}
        for r in rows:
            for k, v in r["res"].items():
                by_item.setdefault(r["qid"], {}).setdefault(k.split("|")[0], []).append(
                    (int(v["letter"] != r["answer"]), v["p_correct"]))
        res = {}
        for sh in SHIFTS:
            per = [sum(e for e, _ in d[sh]) / len(d[sh]) for d in by_item.values() if sh in d]
            pc = [sum(p for _, p in d[sh]) / len(d[sh]) for d in by_item.values() if sh in d]
            rng = random.Random(0)
            bs = sorted(sum(rng.choice(per) for _ in per) / len(per) for _ in range(2000)) if per else []
            res[sh] = dict(err=round(sum(per) / len(per), 3) if per else None,
                           err_ci95=[round(bs[50], 3), round(bs[1949], 3)] if bs else None,
                           p_correct=round(sum(pc) / len(pc), 3) if pc else None, n_items=len(per))
        out[m] = dict(n_traj=len(rows), skipped=sum(1 for r in tio.read_jsonl(
            f"{tio.R}/results/raw/PILOTR3/boxsens__{m}.jsonl") if "skip" in r), by_shift=res)
        print(m, f"trajectories {len(rows)}",
              " | ".join(f"shift {sh}: err {v['err']} {v['err_ci95']} p_corr {v['p_correct']}" for sh, v in res.items()))
    os.makedirs(f"{tio.R}/results/PILOTR", exist_ok=True)
    json.dump(out, open(f"{tio.R}/results/PILOTR/boxsens_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
