"""E4 adaptive items: the first 100 items of v2_main (file order) that both the agent and EIR answer correctly on the
clean image (pre-registered on 2026-11-07).  Needs the clean EIR records
results/raw/E4/clean__eir__<model>__v2_main.jsonl over all 200 items.

  make_e4_qids.py --model deepeyes_7b [--n 100]   -> code/configs/e4a_<model>_qids.txt
"""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
d = {r["qid"]: r for r in tio.read_jsonl(f"{tio.R}/results/raw/E4/clean__eir__{a.model}__v2_main.jsonl")}
items = tio.load_split("v2_main")
miss = [it["qid"] for it in items if it["qid"] not in d]
assert not miss, f"{len(miss)} items lack clean EIR records"
pick = [it["qid"] for it in items if d[it["qid"]]["letter"] == d[it["qid"]]["answer"]
        and d[it["qid"]]["agent"]["letter"] == d[it["qid"]]["answer"]][: a.n]
open(f"{tio.R}/code/configs/e4a_{a.model}_qids.txt", "w").write("\n".join(pick) + "\n")
last = max(i for i, it in enumerate(items) if it["qid"] in set(pick))
print(f"{a.model}: {len(pick)} items (both correct), last index {last}, beyond first 150: "
      f"{sum(1 for i, it in enumerate(items) if i >= 150 and it['qid'] in set(pick))}")
