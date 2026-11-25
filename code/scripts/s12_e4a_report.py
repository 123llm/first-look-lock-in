"""E4 adaptive evaluation through EIR (rules pre-registered on 2026-11-07).

  s12_e4a_report.py  -> results/E4/e4a_report.json and a printed table

Per model (DeepEyes, Mini-o3, PixelReasoner) on its items code/configs/e4a_<model>_qids.txt (first 100 of v2_main with
agent and EIR both correct on the clean image), eps 8:
  ASR_def(attack) = fraction answered wrong by the defended system (agent rollout + EIR) on that attack's images
  attacks: TPA-R (primary), TPA-R+J (box jitter 0.15), EOT-R; non-adaptive reference = E1 single-view eps8 images
  paired differences (same items): TPA-R - EOT-R, TPA-R+J - TPA-R, TPA-R - non-adaptive; cluster bootstrap CIs
Wording: "the adaptive attack gets through EIR" iff ASR_def(TPA-R) >= 0.50 and CI_low(TPA-R - non-adaptive) > 0;
"knowing where the model looks helps" iff CI_low(TPA-R - EOT-R) > 0, otherwise "comparable to random boxes".
Reported whatever the outcome; descriptive: undefended agent ASR per attack.
"""
from __future__ import annotations

import json, os, sys

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s12_e1_report import stat

R = tio.R
MODELS = ["deepeyes_7b", "minio3_7b", "pixelreasoner_7b"]


def files(m):
    return {"TPA-R": f"eir__{m}__E4__tpa_eir@{m}__eps8__seed0", "TPA-R+J": f"eir__{m}__E4__tpa_eir_j15@{m}__eps8__seed0",
            "EOT-R": f"eir__{m}__E4__eot_eir@{m}__eps8__seed0", "non-adaptive": f"eir__{m}__E1__single_view@{m}__eps8__seed0"}


def main():
    out = {}
    for m in MODELS:
        qf = f"{R}/code/configs/e4a_{m}_qids.txt"
        if not os.path.exists(qf):
            out[m] = dict(missing="item list"); continue
        qids = [l.strip() for l in open(qf) if l.strip()]
        res = dict(n_items=len(qids), attacks={}, pending=[])
        d = {}
        for k, f in files(m).items():
            p = f"{R}/results/raw/E4/{f}.jsonl"
            rows = {r["qid"]: r for r in tio.read_jsonl(p)} if os.path.exists(p) else {}
            d[k] = {q: rows[q] for q in qids if q in rows}
            if len(d[k]) < len(qids):
                res["pending"].append(f"{k} ({len(d[k])}/{len(qids)})")
            if d[k]:
                rs = list(d[k].values())
                res["attacks"][k] = dict(ASR_def=stat([(r["cluster"], int(r["letter"] != r["answer"])) for r in rs]),
                                         ASR_agent=stat([(r["cluster"], int(r["agent"]["letter"] != r["answer"]))
                                                         for r in rs]),
                                         eir_applied=sum(bool(r["eir"].get("applied")) for r in rs),
                                         eir_fallback=sum(bool(r["eir"].get("applied") and r["eir"].get("fallback"))
                                                          for r in rs))
        for a, b in (("TPA-R", "EOT-R"), ("TPA-R+J", "TPA-R"), ("TPA-R", "non-adaptive")):
            qs = [q for q in qids if q in d[a] and q in d[b]]
            if qs:
                res[f"{a} - {b}"] = stat([(d[a][q]["cluster"], int(d[a][q]["letter"] != d[a][q]["answer"])
                                           - int(d[b][q]["letter"] != d[b][q]["answer"])) for q in qs])
        t = res["attacks"].get("TPA-R", {}).get("ASR_def", {}).get("rate")
        dn = res.get("TPA-R - non-adaptive", {}).get("ci95"); de = res.get("TPA-R - EOT-R", {}).get("ci95")
        res["gets_through_EIR"] = bool(t is not None and t >= 0.50 and dn and dn[0] > 0)
        res["knowing_where_helps"] = ("yes" if de and de[0] > 0 else "comparable to random boxes") if de else None
        out[m] = res
    os.makedirs(f"{R}/results/E4", exist_ok=True)
    json.dump(out, open(f"{R}/results/E4/e4a_report.json", "w"), indent=1)
    for m, r in out.items():
        if "missing" in r:
            print(f"E4a {m:18s} missing {r['missing']}"); continue
        at = " ".join(f"{k}={v['ASR_def']['rate']}{v['ASR_def']['ci95']}" for k, v in r["attacks"].items())
        print(f"E4a {m:18s} n={r['n_items']} {at}")
        for k in ("TPA-R - EOT-R", "TPA-R+J - TPA-R", "TPA-R - non-adaptive"):
            if k in r:
                print(f"    {k:22s} {r[k]['rate']} {r[k]['ci95']} (n={r[k]['n']})")
        print(f"    through EIR: {r['gets_through_EIR']} | knowing where helps: {r['knowing_where_helps']} "
              f"| pending {r['pending']}")


if __name__ == "__main__":
    main()
