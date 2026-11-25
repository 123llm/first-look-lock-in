"""Trajectory sampling, caching and evaluation records (protocol_v2 attack.tpa, metrics).

sample_trajectories : M self-sampled trajectories of the victim at the current image (EoSO resampling step);
                      sampling parameters are the victim's own (registry.SAMPLING); one torch seed per resample.
TrajCache           : append-only JSONL store keyed by (model, qid, tag) for resumable runs.
record_item         : greedy free generation on an (adversarial) image + probe letters after every observation
                      that was shown to the model; this record is what tpa.metrics consumes.  The probes read the
                      letter distribution under the protocol probe prefix (no gradient).
"""
from __future__ import annotations

import json
import os
import time

import torch

from tpa.models.qwen_agent import Trajectory, Sample, LETTERS


def sample_trajectories(agent, x01: torch.Tensor, s: Sample, M: int, seed: int, max_turns: int,
                        max_new_tokens: int, **rollout_kw) -> list[Trajectory]:
    torch.manual_seed(seed)
    return [agent.rollout(x01, s, max_turns=max_turns, max_new_tokens=max_new_tokens,
                          gen_kwargs=agent.sampling, seed=None, **rollout_kw) for _ in range(M)]


class TrajCache:
    def __init__(self, path: str):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                self.d[r["key"]] = r
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def get(self, key: str):
        r = self.d.get(key)
        return None if r is None else Trajectory.from_dict(r["traj"])

    def put(self, key: str, traj: Trajectory, **extra):
        r = dict(key=key, traj=traj.to_dict(), **extra)
        self.d[key] = r
        with open(self.path, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.no_grad()
def probe_letters(agent, x01: torch.Tensor, s: Sample, traj: Trajectory) -> list[dict]:
    """Probe after the original view and after each observation that was fed back."""
    out = []
    tfw, agent.tf_window = agent.tf_window, None     # probes on the full (untruncated) trajectory
    try:
        for t in traj.obs_turns():
            lg = agent.probe_logits(x01.to(agent.device), s, traj, t)
            p = torch.softmax(lg, -1)
            out.append(dict(turn=t, letter=LETTERS[int(lg.argmax())], probs=[round(float(v), 4) for v in p]))
    finally:
        agent.tf_window = tfw
    return out


def record_item(agent, x01: torch.Tensor, s: Sample, max_turns: int, max_new_tokens: int, probes: bool = True,
                **rollout_kw) -> dict:
    t0 = time.time()
    tr = agent.rollout(x01, s, max_turns=max_turns, max_new_tokens=max_new_tokens, **rollout_kw)
    r = dict(qid=s.image_id, answer=LETTERS[s.answer_idx], letter=tr.final_letter, status=tr.status,
             n_tokens=tr.n_tokens, n_calls=tr.n_tool_calls, n_reobs=tr.n_reobs, forced=tr.forced,
             traj=tr.to_dict(), precision=getattr(agent, "quant", "nf4"))
    if probes:
        r["probes"] = probe_letters(agent, x01, s, tr)
    r["t"] = round(time.time() - t0, 2)
    return r
