"""E3 inputs: expose the intermediate PGD iterates saved by gen_adv.py --save_steps as ordinary adversarial sets.

  make_step_index.py --source deepeyes_7b [--exp E1] [--eps 8] [--out_exp E3]

For every step n in the E1 index rows' `step_pngs`, writes results/adv/<out_exp>/v2_main/sv_s<n>@<source>__eps<eps>__seed0/
index.jsonl whose rows point at the PNGs under the E1 directory (no copy; sha256 checked here and again at evaluation).
"""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--exp", default="E1")
    ap.add_argument("--eps", type=int, default=8); ap.add_argument("--out_exp", default="E3")
    ap.add_argument("--split", default="v2_main")
    a = ap.parse_args()
    src = tio.adv_dir(a.exp, a.split, "single_view", a.source, a.eps, 0)
    rows = tio.read_jsonl(f"{src}/index.jsonl")
    steps = sorted({int(n) for r in rows for n in (r.get("step_pngs") or {})})
    assert steps, f"no step_pngs in {src}"
    for n in steps:
        d = tio.adv_dir(a.out_exp, a.split, f"sv_s{n}", a.source, a.eps, 0)
        os.makedirs(d, exist_ok=True)
        out = []
        for r in rows:
            sha = (r.get("step_pngs") or {}).get(str(n))
            if r.get("error") is not None or sha is None:
                continue
            png = f"{os.path.relpath(src, tio.R)}/steps/{tio.qid_file(r['qid'])}__s{n}.png"
            assert tio.sha256(open(f"{tio.R}/{png}", "rb").read()) == sha, png
            out.append(dict(qid=r["qid"], split=a.split, method=f"sv_s{n}", source=a.source, eps=a.eps, seed=0,
                            png=png, png_sha256=sha, step=n, from_index=f"{os.path.relpath(src, tio.R)}/index.jsonl",
                            error=None))
        with open(f"{d}/index.jsonl", "w") as f:
            for r in out:
                f.write(__import__("json").dumps(r) + "\n")
        print(d, len(out))


if __name__ == "__main__":
    main()
