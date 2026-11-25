"""Formal split of the v2 matrix: V* 100 + HR-Bench-4K 100, disjoint from every pilot
item at the image-cluster level (s11_pilot contains pilotl_qids and the R3 / H-OP items; calib_mme is another dataset).

  make_v2_split.py  -> code/configs/splits/v2_main.json
Items keep the stratified random order of vstar_eval / hrbench4k_eval, so prefixes are nested subsets (n = 150 uses the
first 75 + 75, interleaved by dataset).
"""
from __future__ import annotations

import json, os

R = os.environ.get("TPA_ROOT", ".")
S = f"{R}/code/configs/splits"


def load(n):
    return json.load(open(f"{S}/{n}.json"))["items"]


def main():
    pilot = load("s11_pilot")
    pq, pc = {it["qid"] for it in pilot}, {it["image_cluster"] for it in pilot}
    pick = {}
    for name in ("vstar_eval", "hrbench4k_eval"):
        pick[name] = [it for it in load(name) if it["qid"] not in pq and it["image_cluster"] not in pc][:100]
        assert len(pick[name]) == 100, (name, len(pick[name]))
    items = [it for pair in zip(pick["vstar_eval"], pick["hrbench4k_eval"]) for it in pair]   # interleaved
    assert len({it["qid"] for it in items}) == 200 and not ({it["image_cluster"] for it in items} & pc)
    out = dict(name="v2_main", protocol="v2", seed=0,
               note="V* 100 + HR-Bench-4K 100 (interleaved), image clusters disjoint from s11_pilot; source order kept",
               n=len(items), n_images=len({it["image_sha256"] for it in items}),
               n_clusters=len({it["image_cluster"] for it in items}), items=items)
    json.dump(out, open(f"{S}/v2_main.json", "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in out.items() if k != "items"})


if __name__ == "__main__":
    main()
