"""TPA (Expectation over Self-Observation) and the two baselines that share its PGD engine and zoom operator.

One implementation, three objectives (protocol_v2 `attack:`):
  tpa          J(δ) = E_{τ~M(·|x+δ_r)} [ l_final(τ) + λ · mean_{t∈obs(τ)} l_probe(τ, t) ]
               τ: M trajectories self-sampled with the victim's own sampling parameters at the current
               (quantised) image, resampled every K steps (seed = attack seed + resample index); boxes are
               constants; every observation is recomputed from x+δ by the differentiable operators, so the
               gradient flows through the original view and every re-observation; reasoning text is truncated to
               the last `tf_window` tokens per segment (truncated teacher forcing).
  single_view  J(δ) = l(letter | o_0)  with no tool context (original view only), read where the deployed prompt
               writes the answer.  This is TPA with the trajectory and re-observation terms removed.
  eot          Same J as TPA, but each step conditions on M random boxes (area fraction U[0.05, 0.30] of the image,
               aspect log-U[0.5, 2]) executed with the victim's own zoom operator after a canonical tool call with
               no reasoning (agent.action_segment), instead of self-sampled trajectories.
l = cross-entropy over the item's option letters (untargeted: maximise CE(gt); targeted: minimise CE(target)).
PGD: ℓ∞, T steps, α = ε/8, sign gradient, uniform random start; the returned image is on the uint8 grid.
Identical probe/final positions are computed once per step (their weights are summed).

This module must not import tpa.metrics (protocol isolation rule).
"""
from __future__ import annotations

import io
import hashlib
import math
import zlib
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

from tpa.models.qwen_agent import Trajectory, Sample, LETTERS
from tpa.rollout.sampler import sample_trajectories


@dataclass
class AttackConfig:
    method: str = "tpa"                 # tpa | single_view | eot
    eps: int = 8                        # /255
    steps: int = 50
    alpha_div: int = 8                  # alpha = eps / alpha_div
    M: int = 2
    K: int = 10
    lam: float = 1.0
    tf_window: int | None = 256
    seed: int = 0
    target_idx: int | None = None       # None -> untargeted
    max_turns: int = 6
    max_new_tokens: int = 1024
    token_budget: int | None = None     # H4 adaptive attack at a forced budget
    call_cap: int | None = None
    use_probes: bool = True             # E8 ablation: drop the re-observation (probe) terms
    resample: bool = True               # E8 ablation: sample trajectories only once (step 0)
    eot_area: tuple = (0.05, 0.30)
    eot_aspect: tuple = (0.5, 2.0)
    keep_trajs: bool = True             # store sampled trajectories in the log
    remedy: str | None = None           # "eir" | "eirt": attack the agent defended by evidence-isolated
                                        # re-answering (agent.eir); the final term of every trajectory with a
                                        # fed-back observation becomes the re-answer logits
    box_jitter: float = 0.0             # >0: TPA-R+J, every step each EIR-term box is jittered (centre shift
    box_scale: tuple = (0.8, 1.25)      # U(-j, j) x box size, width/height x log-U[box_scale]); H-OP check
    save_steps: tuple = ()              # E3: keep the canonical uint8 image after these PGD step counts
                                        # (no extra gradients; the final image is unchanged)


def quantize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round() / 255


def objective(logits: torch.Tensor, s: Sample, target_idx: int | None) -> torch.Tensor:
    """Quantity to maximise.  logits are already restricted to the item's option letters."""
    if target_idx is None:
        return F.cross_entropy(logits[None], torch.tensor([s.answer_idx], device=logits.device))
    return -F.cross_entropy(logits[None], torch.tensor([target_idx], device=logits.device))


def jitter_box(box, h: int, w: int, rng: np.random.Generator, shift: float, scale=(0.8, 1.25)):
    """Random perturbation of a zoom box in original-image pixels (TPA-R+J): centre shifted by U(-shift, shift) x
    box size, width and height multiplied by log-U[scale], clipped into the image, at least 28 px per side."""
    assert len(box) == 4, "box jitter is defined for boxes in original-image pixels"
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 + rng.uniform(-shift, shift) * bw
    cy = (y1 + y2) / 2 + rng.uniform(-shift, shift) * bh
    lo, hi = math.log(scale[0]), math.log(scale[1])
    nw = min(w, max(28, bw * math.exp(rng.uniform(lo, hi))))
    nh = min(h, max(28, bh * math.exp(rng.uniform(lo, hi))))
    nx1 = int(round(min(max(cx - nw / 2, 0), w - nw)))
    ny1 = int(round(min(max(cy - nh / 2, 0), h - nh)))
    return (nx1, ny1, min(w, nx1 + int(round(nw))), min(h, ny1 + int(round(nh))))


def jitter_boxes(boxes, h: int, w: int, rng: np.random.Generator, shift: float, scale=(0.8, 1.25), obs_hw=None):
    """Jitter every zoom box of a trajectory (TPA-R+J).  4-tuples are in original-image pixels (DeepEyes) and are
    jittered by jitter_box exactly as before.  5-tuples (x1, y1, x2, y2, src) are in pixels of observation `src`
    (Mini-o3, PixelReasoner): the box is carried into the jittered chain by rescaling it to the size of its jittered
    source observation, then jittered inside that observation; obs_hw(box) gives the (h, w) of the observation a
    box produces (agent-specific: raw crop for Mini-o3, resized crop for PixelReasoner)."""
    out, old_hw, new_hw = [], [(h, w)], [(h, w)]
    for b in boxes:
        if b is None:
            out.append(None)
        elif len(b) == 4:
            out.append(jitter_box(b, h, w, rng, shift, scale))
        else:
            s = b[4]
            (oh, ow), (nh, nw) = old_hw[s], new_hw[s]
            fx, fy = nw / ow, nh / oh
            j = jitter_box((b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy), nh, nw, rng, shift, scale)
            out.append(j + (s,))
            old_hw.append(obs_hw(b)); new_hw.append(obs_hw(j))
    return out


def sample_eot_box(h: int, w: int, rng: np.random.Generator, area=(0.05, 0.30), aspect=(0.5, 2.0)):
    """Random box (x1, y1, x2, y2) in pixels: area fraction ~ U[area], aspect (w/h) ~ log-U[aspect]."""
    a = rng.uniform(*area) * h * w
    r = math.exp(rng.uniform(math.log(aspect[0]), math.log(aspect[1])))
    bw = int(round(min(w, max(28, math.sqrt(a * r)))))
    bh = int(round(min(h, max(28, math.sqrt(a / r)))))
    x1 = int(rng.integers(0, w - bw + 1))
    y1 = int(rng.integers(0, h - bh + 1))
    return (x1, y1, x1 + bw, y1 + bh)


def _key(traj: Trajectory | None, upto: int, kind: str):
    if traj is None:
        return (kind, upto)
    return (kind, upto, tuple(traj.segments[:upto]) if kind == "probe" else tuple(traj.segments),
            tuple(traj.boxes[:upto]) if kind == "probe" else tuple(traj.boxes))


class Attack:
    def __init__(self, agent, cfg: AttackConfig):
        self.agent, self.cfg = agent, cfg
        assert cfg.method in ("tpa", "single_view", "eot"), cfg.method
        agent.enable_grad_ckpt()

    # ---------------------------------------------------------------- terms
    def _traj_terms(self, trajs: list[Trajectory]):
        """[(key, fn(x)->logits, weight)] for the EoSO objective over the given trajectories."""
        ag, cfg, s = self.agent, self.cfg, self.s
        terms = {}

        def add(key, fn, w):
            if key in terms:
                terms[key][1] += w
            else:
                terms[key] = [fn, w]

        M = len(trajs)
        eir = getattr(ag, "eir", None) if cfg.remedy else None
        assert not cfg.remedy or eir is not None, "remedy set but agent.eir missing"
        for tr in trajs:
            if eir is not None and eir.applies(tr):     # defended decision = re-answer on the fed-back views
                last = tr.obs_turns()[-1]
                te = tr
                if cfg.box_jitter > 0:                   # fresh jitter per step and per trajectory (EOT over boxes)
                    h, w = self.hw
                    te = Trajectory(segments=list(tr.segments), status=tr.status,
                                    boxes=jitter_boxes(tr.boxes, h, w, self.rng, cfg.box_jitter, cfg.box_scale,
                                                       getattr(ag, "obs_hw", None)))
                add(("eir", last, tuple(te.boxes[:last])), (lambda x, te=te: eir.answer_logits(x, s, te)), 1.0 / M)
            elif tr.segments and tr.boxes[-1] is None and tr.status != "synthetic":
                add(_key(tr, len(tr.segments), "final"), (lambda x, tr=tr: ag.final_answer_logits(x, s, tr)), 1.0 / M)
            else:   # synthetic EOT trajectory or pending call: the "final" read-out is the probe after the last obs
                last = tr.obs_turns()[-1]
                add(_key(tr, last, "probe"), (lambda x, tr=tr, t=last: ag.probe_logits(x, s, tr, t)), 1.0 / M)
            if cfg.use_probes and cfg.lam > 0:
                obs = tr.obs_turns()
                for t in obs:
                    add(_key(tr, t, "probe"), (lambda x, tr=tr, t=t: ag.probe_logits(x, s, tr, t)),
                        cfg.lam / (M * len(obs)))
        return [(k, f, w) for k, (f, w) in terms.items()]

    def _terms(self, k: int, x_q: torch.Tensor):
        cfg, ag, s = self.cfg, self.agent, self.s
        if cfg.method == "single_view":
            return [(("direct", 0), (lambda x: ag.probe_logits(x, s, None, 0, direct=True)), 1.0)]
        if cfg.method == "eot":
            h, w = x_q.shape[-2:]
            trajs = []
            for _ in range(cfg.M):
                b = sample_eot_box(h, w, self.rng, cfg.eot_area, cfg.eot_aspect)
                tr = Trajectory(segments=[ag.action_segment(b, x_q)], boxes=[b + ((0,) if ag.BOX_HAS_SRC else ())],
                                n_tool_calls=1, status="synthetic")
                trajs.append(tr)
            self.log["boxes"].append([list(t.boxes[0]) for t in trajs])
            return self._traj_terms(trajs)
        # tpa
        if k == 0 or (cfg.resample and k % cfg.K == 0):
            r = k // cfg.K
            tfw, ag.tf_window = ag.tf_window, None
            t0 = time.time()
            self.trajs = sample_trajectories(ag, x_q, s, cfg.M, cfg.seed + r, cfg.max_turns, cfg.max_new_tokens,
                                             token_budget=cfg.token_budget, call_cap=cfg.call_cap)
            ag.tf_window = tfw
            self.log["n_rollouts"] += cfg.M
            self.log["t_rollout"] += time.time() - t0
            self.log["resamples"].append(dict(
                step=k, seed=cfg.seed + r,
                summary=[dict(status=t.status, letter=t.final_letter, n_calls=t.n_tool_calls, n_tokens=t.n_tokens)
                         for t in self.trajs],
                trajs=[t.to_dict() for t in self.trajs] if cfg.keep_trajs else None))
        return self._traj_terms(self.trajs)

    # ---------------------------------------------------------------- PGD
    def run(self, x0: torch.Tensor, s: Sample):
        cfg, ag = self.cfg, self.agent
        self.s = s
        self.rng = np.random.default_rng([cfg.seed, zlib.crc32(s.image_id.encode())])
        dev = ag.device
        x0 = x0.to(dev).float()
        self.hw = tuple(x0.shape[-2:])
        eps, alpha = cfg.eps / 255.0, cfg.eps / 255.0 / cfg.alpha_div
        g = torch.Generator().manual_seed(cfg.seed)
        delta = ((torch.rand(x0.shape, generator=g) * 2 - 1) * eps).to(dev)
        delta = (x0 + delta).clamp(0, 1) - x0
        self.log = dict(cfg=asdict(cfg), steps=[], resamples=[], boxes=[], n_rollouts=0, n_fwd_bwd=0, n_grad_evals=0,
                        t_rollout=0.0, t_grad=0.0)
        self.snapshots = {}
        old_tfw = ag.tf_window
        torch.cuda.reset_peak_memory_stats()
        try:
            for k in range(cfg.steps):
                x_q = quantize(x0 + delta)
                terms = self._terms(k, x_q)
                ag.tf_window = cfg.tf_window
                t0 = time.time()
                grad = torch.zeros_like(delta)
                J, parts = 0.0, []
                for key, fn, w in terms:
                    d = delta.clone().requires_grad_(True)
                    lg = fn(x0 + d)
                    j = objective(lg, s, cfg.target_idx)
                    (w * j).backward()
                    grad += d.grad
                    J += w * float(j.detach())
                    parts.append(dict(kind=key[0], upto=key[1], w=round(w, 4), j=round(float(j.detach()), 4),
                                      argmax=LETTERS[int(lg.argmax())]))
                    del d, lg, j
                self.log["n_fwd_bwd"] += len(terms)       # actual model forward+backward passes
                # gradient evaluations of the objective: one per trajectory / random box (T*M for TPA and EOT,
                # protocol "equal backward count"), one per step for single view; per-term passes are accumulated
                # into that single gradient (identical to back-propagating the summed per-trajectory loss)
                self.log["n_grad_evals"] += 1 if cfg.method == "single_view" else cfg.M
                self.log["t_grad"] += time.time() - t0
                delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
                delta = (x0 + delta).clamp(0, 1) - x0
                self.log["steps"].append(dict(k=k, J=round(J, 4), terms=parts, gnorm=float(grad.abs().mean())))
                if k + 1 in cfg.save_steps:
                    self.snapshots[k + 1] = ((x0 + delta).clamp(0, 1) * 255).round().to(torch.uint8).cpu().float() / 255
        finally:
            ag.tf_window = old_tfw
        # canonical uint8 image, divided on the CPU (CUDA division can differ by 1 ulp from the PNG reload)
        x_adv = ((x0 + delta).clamp(0, 1) * 255).round().to(torch.uint8).cpu().float() / 255
        x0 = x0.cpu()
        linf = float((x_adv - x0).abs().max()) * 255
        assert linf <= cfg.eps + 1e-3, f"linf {linf} > eps {cfg.eps}"
        self.log.update(linf_255=round(linf, 3), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        t_rollout=round(self.log["t_rollout"], 1), t_grad=round(self.log["t_grad"], 1))
        return x_adv.cpu(), self.log


def to_png_bytes(x01: torch.Tensor) -> bytes:
    from PIL import Image
    arr = (x01.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def save_png(x01: torch.Tensor, path: str) -> str:
    b = to_png_bytes(x01)
    with open(path, "wb") as f:
        f.write(b)
    return hashlib.sha256(b).hexdigest()
