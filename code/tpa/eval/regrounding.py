"""Re-grounding analysis (I3) and budget forcing (I3 intervention, I4 grid).

RG  : re-observations actually shown to the model per trajectory (Trajectory.n_reobs).
VEG : visual-evidence gain of re-grounding, measured by intervention (protocol_v2 metrics.veg, after 2608.06270):
      on the clean greedy trajectory, the first executed re-observation is replaced by a random box of the same
      size (seed crc32(question_id)); the model keeps its own call text, sees the random crop instead, and
      continues greedily.  Outcome = the final answer changes (letter or validity).  VEG = P(changed) over items
      whose clean trajectory has >= 1 re-observation.  A model whose answers do not depend on what it zooms into
      has VEG ~ 0.
Budget forcing: free generation under a total generated-token budget and a tool-call cap; when either runs out
      the reasoning is closed and an answer is forced (QwenAgent.rollout token_budget / call_cap).
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from tpa.models.qwen_agent import Trajectory, Sample


def random_box_same_size(box, h: int, w: int, qid: str):
    """Uniformly placed box with the same width/height as `box` inside an h x w image (seed crc32(qid)).
    Keeps a trailing source index (Mini-o3) unchanged."""
    x1, y1, x2, y2 = box[:4]
    bw, bh = min(x2 - x1, w), min(y2 - y1, h)
    rng = np.random.default_rng(zlib.crc32(qid.encode()))
    nx = int(rng.integers(0, w - bw + 1))
    ny = int(rng.integers(0, h - bh + 1))
    return (nx, ny, nx + bw, ny + bh) + tuple(box[4:])


def _source_size(agent, x01, traj: Trajectory, k: int):
    """(h, w) of the image the k-th box refers to (original image; Mini-o3 may point at an earlier observation)."""
    h, w = x01.shape[-2:]
    b = traj.boxes[k]
    if getattr(agent, "BOX_HAS_SRC", False) and b[4] != 0:
        sizes = agent._obs_sizes(x01, traj, k)
        w, h = sizes[b[4]]
    return h, w


@torch.no_grad()
def veg_item(agent, x01: torch.Tensor, s: Sample, clean: Trajectory, max_turns: int, max_new_tokens: int) -> dict | None:
    """Returns None when the clean trajectory has no shown re-observation (item not in the VEG denominator)."""
    turns = clean.obs_turns()
    if len(turns) < 2:
        return None
    k = turns[1] - 1                                   # index of the first executed call
    h, w = _source_size(agent, x01, clean, k)
    rb = random_box_same_size(clean.boxes[k], h, w, s.image_id)
    prefix = Trajectory(segments=list(clean.segments[:k + 1]), boxes=list(clean.boxes[:k]) + [rb],
                        n_tool_calls=sum(b is not None for b in clean.boxes[:k + 1]))
    tr = agent.rollout(x01, s, max_turns=max_turns, max_new_tokens=max_new_tokens, start=prefix)
    changed = (tr.status != clean.status) or (tr.final_letter != clean.final_letter)
    return dict(qid=s.image_id, box=list(clean.boxes[k]), random_box=list(rb), letter=tr.final_letter,
                status=tr.status, clean_letter=clean.final_letter, changed=changed, traj=tr.to_dict())


def veg(results: list[dict | None]) -> float:
    r = [x["changed"] for x in results if x is not None]
    return float(np.mean(r)) if r else float("nan")


def rg(trajs: list[Trajectory]) -> float:
    return float(np.mean([t.n_reobs for t in trajs])) if trajs else float("nan")


@torch.no_grad()
def rollout_budget(agent, x01: torch.Tensor, s: Sample, token_budget: int, call_cap: int, max_new_tokens: int | None = None):
    """One cell of the H4 grid / the I3 budget-forcing intervention (greedy)."""
    return agent.rollout(x01, s, max_turns=call_cap + 1, max_new_tokens=max_new_tokens or token_budget,
                         token_budget=token_budget, call_cap=call_cap)
