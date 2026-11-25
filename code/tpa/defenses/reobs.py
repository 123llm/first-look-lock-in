"""Decorrelated re-observation (DRO): a training-free defense on the zoom tool of a thinking-with-images model.

Each re-observation (zoomed view) is passed through a fresh random transform before the model sees it, so that
successive views no longer share one fixed adversarial perturbation:
    random down-scale s ~ U[0.5, 0.9] (bilinear, antialias) -> bicubic up-scale back -> random cyclic shift of
    up to +-4 px -> Gaussian noise sigma = 0.02 -> uint8 rounding (straight-through).
The original full view is left unchanged.  All operations are differentiable, so an adaptive attacker can take
gradients through them; a new random draw is made on every call (EOT when called inside an attack loop).
Parameters were fixed before any result (2026-11-05) and are not tuned.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tpa.processors.qwen_diff import ste_round_uint8


class DecorrelatedReobs:
    """The random draw for the k-th zoomed view depends only on (epoch, k): a view the tool has returned stays the
    same while the conversation is re-encoded turn after turn, as with a real tool.  `epoch` is set per item for
    evaluation (crc32(qid)) and advanced by `new_draw()` for every rollout / gradient forward inside an attack (EOT)."""

    def __init__(self, scale=(0.5, 0.9), shift: int = 4, sigma: float = 0.02, seed: int = 0):
        self.scale, self.shift, self.sigma = scale, shift, sigma
        self.epoch = seed
        self.resample_per_forward = False      # True inside an adaptive attack

    def reseed(self, seed: int):
        self.epoch = seed

    def new_draw(self):
        self.epoch += 1

    def params(self):
        return dict(scale=list(self.scale), shift=self.shift, sigma=self.sigma)

    def __call__(self, v: torch.Tensor, k: int = 0) -> torch.Tensor:
        """v: float (3, H, W) in [0, 1] on the uint8 grid.  Returns the same shape on the uint8 grid."""
        g = torch.Generator().manual_seed((self.epoch * 1_000_003 + k) % (2**63 - 1))
        _, H, W = v.shape
        lo, hi = self.scale
        s = lo + (hi - lo) * float(torch.rand(1, generator=g))
        h2, w2 = max(28, int(round(H * s))), max(28, int(round(W * s)))
        y = F.interpolate(v[None], size=(h2, w2), mode="bilinear", antialias=True, align_corners=False)
        y = F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)[0]
        dy, dx = (int(t) for t in torch.randint(-self.shift, self.shift + 1, (2,), generator=g))
        y = torch.roll(y, shifts=(dy, dx), dims=(1, 2))
        noise = torch.randn(y.shape, generator=g).to(device=y.device, dtype=y.dtype)
        y = (y + self.sigma * noise).clamp(0, 1)
        return ste_round_uint8(y * 255.0) / 255.0
