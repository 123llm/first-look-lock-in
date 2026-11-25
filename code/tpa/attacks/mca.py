"""MCA-style (reimpl.) transfer baseline: local-crop alignment on a CLIP surrogate ensemble, after M-Attack /
M-Attack-V2 (VILA-Lab/M-Attack-V2 @ 8faca6d, attack/base.py PGDAttackMutiPass + generate_ad_samples.py, MIT).

Kept from the official attack: an ensemble of CLIP image encoders; at every step `passes` local crops of the
adversarial image (RandomResizedCrop, area scale [0.5, 1.0], ratio [3/4, 4/3], resized to 224) whose gradients
are accumulated before one update (multi-pass, K = 10 in the official config); cosine similarity of the
encoders' image embeddings.
Changed for this paper (protocol_v2 attack.mca_style_reimpl, untargeted feature-push variant): there is no
target image; the loss pushes the embedding of each adversarial crop away from the embedding of the same crop of
the clean image (minimise cosine similarity).  The update is the protocol's PGD (sign, alpha = eps/8, T = 50,
uniform random start, l_inf projection), not Adam with eps 16, so all gradient baselines share one optimiser.
The perturbation lives at the full image resolution; crops are resized with antialiased bilinear interpolation
(differentiable).  The surrogates never see the victim.
"""
from __future__ import annotations

import math
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SURROGATES = {"clip-vit-b16": "openai__clip-vit-base-patch16", "clip-vit-b32": "openai__clip-vit-base-patch32",
              "clip-vit-l14": "openai__clip-vit-large-patch14"}
MODEL_ROOT = __import__("os").environ.get("TPA_ROOT", ".") + "/models"


def rrc_params(h: int, w: int, rng: np.random.Generator, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
    """torchvision RandomResizedCrop.get_params with a numpy generator (10 tries, then centre crop)."""
    area = h * w
    for _ in range(10):
        a = area * rng.uniform(*scale)
        r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if 0 < cw <= w and 0 < ch <= h:
            return int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1)), ch, cw
    in_r = w / h
    if in_r < ratio[0]:
        cw, ch = w, int(round(w / ratio[0]))
    elif in_r > ratio[1]:
        ch, cw = h, int(round(h * ratio[1]))
    else:
        cw, ch = w, h
    return (h - ch) // 2, (w - cw) // 2, ch, cw


class MCAStyle:
    def __init__(self, surrogates=("clip-vit-b16", "clip-vit-b32", "clip-vit-l14"), device="cuda", eps: int = 8,
                 steps: int = 50, alpha_div: int = 8, scale=(0.5, 1.0), passes: int = 10, seed: int = 0,
                 res: int = 224):
        from transformers import CLIPVisionModelWithProjection
        self.models = []
        for name in surrogates:
            m = CLIPVisionModelWithProjection.from_pretrained(f"{MODEL_ROOT}/{SURROGATES[name]}",
                                                              dtype=torch.float32).to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.models.append(m)
        self.names = list(surrogates)
        self.device, self.eps, self.steps, self.alpha_div = device, eps, steps, alpha_div
        self.scale, self.passes, self.seed, self.res = scale, passes, seed, res
        self.mean = torch.tensor(CLIP_MEAN, device=device)[:, None, None]
        self.std = torch.tensor(CLIP_STD, device=device)[:, None, None]

    def _embed(self, m, x224):
        out = m(pixel_values=((x224 - self.mean) / self.std)[None])
        return F.normalize(out.image_embeds, dim=-1)

    def _crop(self, x, box):
        t, l, h, w = box
        return F.interpolate(x[None, :, t:t + h, l:l + w], size=(self.res, self.res), mode="bilinear",
                             align_corners=False, antialias=True)[0]

    def run(self, x0: torch.Tensor, qid: str):
        dev = self.device
        x0 = x0.to(dev).float()
        H, W = x0.shape[-2:]
        eps, alpha = self.eps / 255.0, self.eps / 255.0 / self.alpha_div
        rng = np.random.default_rng([self.seed, zlib.crc32(qid.encode())])
        g = torch.Generator().manual_seed(self.seed)
        delta = ((torch.rand(x0.shape, generator=g) * 2 - 1) * eps).to(dev)
        delta = (x0 + delta).clamp(0, 1) - x0
        log = dict(method="mca_style", surrogates=self.names, passes=self.passes, steps=[], n_fwd_bwd=0)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        for k in range(self.steps):
            d = delta.clone().requires_grad_(True)
            sim_tot = 0.0
            for _ in range(self.passes):
                box = rrc_params(H, W, rng, self.scale)
                xa = self._crop((x0 + d).clamp(0, 1), box)
                with torch.no_grad():
                    xc = self._crop(x0, box)
                sim = 0.0
                for m in self.models:
                    with torch.no_grad():
                        ec = self._embed(m, xc)
                    sim = sim + (self._embed(m, xa) * ec).sum() / len(self.models)
                (sim / self.passes).backward()          # accumulate over passes, one update (multi-pass)
                sim_tot += float(sim.detach()) / self.passes
                log["n_fwd_bwd"] += len(self.models)
            # minimise similarity -> descend
            delta = (delta - alpha * d.grad.sign()).clamp(-eps, eps)
            delta = (x0 + delta).clamp(0, 1) - x0
            log["steps"].append(dict(k=k, sim=round(sim_tot, 4)))
        x_adv = ((x0 + delta).clamp(0, 1) * 255).round().to(torch.uint8).cpu().float() / 255
        linf = float((x_adv - x0.cpu()).abs().max()) * 255
        assert linf <= self.eps + 1e-3
        log.update(linf_255=round(linf, 3), t=round(time.time() - t0, 1),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        return x_adv, log
