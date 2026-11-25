"""V* Bench loader (craigwu/vstar_bench): 191 questions = direct_attributes (115) + relative_position (76).

The official annotation stores the correct option at options[0] (DeepEyes eval_vstar.py: answer = options[0]).
We shuffle options with a fixed per-question seed (2026-10-31):
    rng = numpy.random.default_rng([0, zlib.crc32(question_id)])
so the correct letter is uniformly spread and targeted-attack targets are not position-biased.
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np
import torch
from PIL import Image

from tpa.models.qwen_agent import Sample

SPLITS = ("direct_attributes", "relative_position")


def shuffle_options(qid: str, options: list[str], correct: int = 0):
    rng = np.random.default_rng([0, zlib.crc32(qid.encode())])
    perm = rng.permutation(len(options))
    new = [options[i] for i in perm]
    return new, int(np.where(perm == correct)[0][0])


def load_vstar(root: str, splits=SPLITS, limit: int | None = None):
    items = []
    for sp in splits:
        d = os.path.join(root, sp)
        for fn in sorted(f for f in os.listdir(d) if f.endswith(".json")):
            with open(os.path.join(d, fn)) as f:
                anno = json.load(f)
            stem = fn[:-5]
            img = next(os.path.join(d, stem + e) for e in (".jpg", ".JPG", ".png", ".jpeg")
                       if os.path.exists(os.path.join(d, stem + e)))
            qid = f"vstar/{sp}/{stem}"
            opts, ans = shuffle_options(qid, anno["options"], 0)
            items.append((Sample(image_id=qid, question=anno["question"], options=opts, answer_idx=ans), img,
                          anno.get("bbox")))
    return items[:limit] if limit else items


def load_image01(path: str) -> torch.Tensor:
    """RGB image as float [0,1] tensor (3,H,W) on the uint8 grid."""
    arr = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
