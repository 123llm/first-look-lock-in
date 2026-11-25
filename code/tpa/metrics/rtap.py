"""RTAP trajectory-level metrics (protocol_v2 `metrics:`).

Input: one row per item with the clean and the adversarial greedy records produced by
tpa.rollout.sampler.record_item on the saved PNGs (free generation; never teacher-forced logits):
    {"qid", "cluster", "clean": rec, "adv": rec, "target": letter | None}
rec = {"answer", "letter", "status", "n_tokens", "n_calls", "n_reobs", "probes": [{"turn", "letter"}, ...],
       "traj": {"segments": [...]}}

Denominator D (untargeted): clean status ok and clean letter == answer.  Targeted: same D, success = adv letter
== target.  Invalid adversarial outputs (truncated / max_turns / unparsable) are never successes; their rate is
reported separately.  "Flipped" for a letter L means L != answer (untargeted) or L == target (targeted).

  ASR   P(adv valid and final flipped)
  INV   P(adv invalid)
  PFR   P(probe after o_0 flipped) on adv; PFR_clean is the same read-out on the clean image (baseline)
  RFR   P(probe after o_0 not flipped and final flipped)        -- flips that arise during reasoning
  SRR   P(final correct and valid | some probe flipped)         -- self-correction given a perception-level flip
  FSP   mean over successes of (first flipped position) / (L-1), positions = probes after each shown observation
        followed by the final answer (0 = flipped at the original view, 1 = only the final answer flipped)
  DbF   P(adv trace mentions the frozen lexicon and clean trace does not | success); DbF_D: same over D
  CIR   sum adv / sum clean over D, for generated tokens and tool calls
  RG    mean shown re-observations per trajectory (clean, adv)
"""
from __future__ import annotations

import re

import numpy as np

from tpa.metrics.stats import cluster_bootstrap

DBF_LEXICON = ["noise", "noisy", "artifact", "artifacts", "artefact", "distorted", "distortion", "perturbed",
               "perturbation", "adversarial", "corrupted", "corruption", "grainy", "pixelated", "pixelation",
               "glitch", "static", "speckle", "jpeg"]
_DBF = re.compile(r"\b(" + "|".join(DBF_LEXICON) + r")\b", re.I)


def trace_text(rec: dict) -> str:
    return "\n".join(rec.get("traj", {}).get("segments", []))


def dbf_hit(rec: dict) -> bool:
    return bool(_DBF.search(trace_text(rec)))


def in_denominator(row: dict) -> bool:
    c = row["clean"]
    return c["status"] == "ok" and c["letter"] == c["answer"]


def item_indicators(row: dict) -> dict:
    a, c, gt, tgt = row["adv"], row["clean"], row["clean"]["answer"], row.get("target")
    flip = (lambda L: L is not None and L != gt) if tgt is None else (lambda L: L == tgt)
    valid = a["status"] == "ok"
    success = valid and flip(a["letter"])
    probes = [p["letter"] for p in a.get("probes", [])]
    seq = probes + [a["letter"] if valid else None]
    first = next((i for i, L in enumerate(seq) if flip(L)), None)
    any_probe_flip = any(flip(L) for L in probes)
    p0 = probes[0] if probes else None
    c0 = c["probes"][0]["letter"] if c.get("probes") else None
    adv_dbf, clean_dbf = dbf_hit(a), dbf_hit(c)
    return dict(
        success=success, invalid=not valid,
        pfr=flip(p0), pfr_clean=flip(c0),
        rfr=(not flip(p0)) and success,
        any_probe_flip=any_probe_flip,
        recovered=any_probe_flip and valid and a["letter"] == gt,
        fsp=(first / (len(seq) - 1) if success and first is not None and len(seq) > 1 else
             (0.0 if success and len(seq) == 1 else None)),
        dbf=success and adv_dbf and not clean_dbf,
        tok_adv=a["n_tokens"], tok_clean=c["n_tokens"], call_adv=a["n_calls"], call_clean=c["n_calls"],
        rg_adv=a.get("n_reobs", a["n_calls"]), rg_clean=c.get("n_reobs", c["n_calls"]),
    )


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


STATS = {
    "ASR": lambda I: _mean([i["success"] for i in I]),
    "INV": lambda I: _mean([i["invalid"] for i in I]),
    "PFR": lambda I: _mean([i["pfr"] for i in I]),
    "PFR_clean": lambda I: _mean([i["pfr_clean"] for i in I]),
    "RFR": lambda I: _mean([i["rfr"] for i in I]),
    "SRR": lambda I: _mean([i["recovered"] for i in I if i["any_probe_flip"]]),
    "FSP": lambda I: _mean([i["fsp"] for i in I if i["success"]]),
    "DbF": lambda I: _mean([i["dbf"] for i in I if i["success"]]),
    "DbF_D": lambda I: _mean([i["dbf"] for i in I]),
    "CIR_tokens": lambda I: (sum(i["tok_adv"] for i in I) / max(1, sum(i["tok_clean"] for i in I))) if I else float("nan"),
    "CIR_calls": lambda I: (sum(i["call_adv"] for i in I) / sum(i["call_clean"] for i in I))
    if I and sum(i["call_clean"] for i in I) > 0 else float("nan"),
    "RG_clean": lambda I: _mean([i["rg_clean"] for i in I]),
    "RG_adv": lambda I: _mean([i["rg_adv"] for i in I]),
}


def compute(rows: list[dict], B: int = 2000, seed: int = 0, which=None) -> dict:
    """Point estimates and image-cluster bootstrap 95% CIs over the denominator D."""
    D = [r for r in rows if in_denominator(r)]
    I = [item_indicators(r) for r in D]
    cl = [r["cluster"] for r in D]
    out = dict(n_rows=len(rows), n_D=len(D), clean_acc=_mean([in_denominator(r) for r in rows]))
    for name, f in STATS.items():
        if which and name not in which:
            continue
        if not I:
            out[name] = dict(point=float("nan"), lo=float("nan"), hi=float("nan"))
            continue
        p, lo, hi = cluster_bootstrap(I, f, cl, B=B, seed=seed) if B else (f(I), float("nan"), float("nan"))
        out[name] = dict(point=p, lo=lo, hi=hi)
    return out
