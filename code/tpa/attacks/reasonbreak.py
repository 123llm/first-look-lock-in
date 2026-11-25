"""ReasonBreak baseline (Zhang et al., ICLR 2026; jiamingzhang94/reasonbreak @ 7b46b6f, MIT; weights
hf jiamingzz/reason_break weights.pth; embedding bank data/embedding_bank.pth; CLIP ViT-B/32 sha256 40d36571...).

Official part (unchanged code, imported from third_party/reasonbreak/generate.py): the image is tiled into
224x224 blocks (dynamic aspect-ratio tiling, max 64 blocks); each block gets the key concepts of the reasoning
steps whose square box overlaps it; the concept embedding farthest from them in the embedding bank drives the
trained decoder, whose noise is clamped to eps, stitched and resized to the image.
Changed (partial reproduction): the official pipeline gets the reasoning chain (steps with key_concept and
square_bbox [cx, cy, size]) from Gemini for geolocation.  We have no API, and our task is multiple-choice VQA, so
the chain is produced by an open model (Qwen3-VL-4B-Instruct, greedy) with a question-conditioned prompt that asks
for the same JSON fields.  eps follows our protocol (8/255) instead of the official 16/255.  The adversarial image
is taken as the official generator returns it (PIL conversion of the clamped float image) and then saved as PNG.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch

RB_DIR = __import__("os").environ.get("TPA_ROOT", ".") + "/third_party/reasonbreak"
RB_WEIGHTS = __import__("os").environ.get("TPA_ROOT", ".") + "/models/reasonbreak/weights.pth"
RB_BANK = f"{RB_DIR}/data/embedding_bank.pth"

CHAIN_PROMPT = (
    "Question: {question}\nOptions:\n{options}\n"
    "Before answering, plan how you would reason about this image to answer the question. Give 2 to 4 reasoning "
    "steps. Each step names one key visual concept that the step relies on and the square region of the image "
    "where that concept is visible. Keep each reasoning to one short sentence. Output only JSON in this format:\n"
    '{{"reasoning_chain": [{{"step_id": 1, "reasoning": "...", "key_concept": "short noun phrase", '
    '"square_bbox": [center_x, center_y, size]}}]}}\n'
    "center_x and center_y are relative to the image width and height (0 to 1); size is the side of the square "
    "relative to the shorter image side (0 to 1).")


def chain_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return CHAIN_PROMPT.format(question=question, options=opts)


_STEP = re.compile(r'"key_concept"\s*:\s*"([^"]+)"\s*,\s*"square_bbox"\s*:\s*\[([^\]]*)\]', re.S)


def _square(b):
    """3 numbers -> as given; 4 numbers -> official conversion of corners [x1,y1,x2,y2] to [cx, cy, max(w,h)]
    (generate.py parse_reasoning_chain); when the 4 numbers are not valid corners (x2<=x1 or y2<=y1) they are
    read as [cx, cy, w, h]."""
    try:
        b = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if b and max(b) > 1.0 and max(b) <= 1000.0:     # Qwen3-VL grounds natively on a 0-1000 relative scale
        b = [v / 1000.0 for v in b]
    if len(b) == 4:
        x1, y1, x2, y2 = b
        b = [(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)] if (x2 > x1 and y2 > y1) else [x1, y1, max(x2, y2)]
    if len(b) != 3 or b[2] <= 0:
        return None
    return [min(1.0, max(0.0, v)) for v in b]


def parse_chain(text: str) -> list[dict]:
    """Official parsing rule (JSON between the first '{' and the last '}'); if that fails (e.g. cut-off output),
    the steps that are complete are recovered with a regex.  Only steps with a key concept and a valid box are kept."""
    i, j = text.find("{"), text.rfind("}") + 1
    try:
        steps = [(st.get("key_concept"), st.get("square_bbox", st.get("bbox")), st.get("reasoning", ""))
                 for st in json.loads(text[i:j])["reasoning_chain"]]
    except Exception:
        steps = [(m.group(1), m.group(2).split(","), "") for m in _STEP.finditer(text)]
    out = []
    for k, (concept, box, reason) in enumerate(steps):
        sq = _square(box) if isinstance(box, (list, tuple)) else None
        if isinstance(concept, str) and concept.strip() and sq:
            out.append(dict(step_id=k + 1, key_concept=concept.strip(), square_bbox=sq, reasoning=reason))
    return out


@torch.no_grad()
def make_chain(agent, x01: torch.Tensor, question: str, options: list[str], max_new_tokens: int = 768) -> dict:
    from tpa.eval.caption_judge import _generate_text
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": chain_prompt(question, options)}]}]
    text, _ = _generate_text(agent, msgs, [x01.to(agent.device)], max_new_tokens)
    chain = parse_chain(text)
    if not chain:   # official fallback box when a step has no box; here: one global step with the question
        chain = [dict(step_id=1, key_concept=question[:60], square_bbox=[0.5, 0.5, 1.0], reasoning="fallback")]
    return dict(chain=chain, raw=text, fallback=chain[0].get("reasoning") == "fallback")


class ReasonBreak:
    def __init__(self, eps: int = 8, max_num_blocks: int = 64, device: str = "cuda"):
        if RB_DIR not in sys.path:
            sys.path.insert(0, RB_DIR)
        from generate import AdversarialGenerator          # official code
        self.gen = AdversarialGenerator(eps=eps / 255.0, max_num_blocks=max_num_blocks, decoder_path=RB_WEIGHTS,
                                        embedding_bank_path=RB_BANK, device=device, save_format="png")
        self.eps = eps

    def run(self, image_path: str, chain: list[dict]):
        # the official function swallows exceptions and returns the clean image; detect that explicitly
        arr = self.gen.generate_adversarial_sample(image_path, chain)
        from PIL import Image
        clean = np.asarray(Image.open(image_path).convert("RGB"))
        if arr is None or np.array_equal(np.asarray(arr), clean):
            raise RuntimeError("ReasonBreak generator returned the clean image (internal exception)")
        # official float->PIL conversion truncates (mul(255).byte()), which can exceed eps by one grey level;
        # project back onto the protocol's l_inf ball in uint8 space (affects only those +-1 pixels)
        c = clean.astype(np.int16)
        arr = np.clip(np.asarray(arr).astype(np.int16), c - self.eps, c + self.eps).clip(0, 255).astype(np.uint8)
        x_adv = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        return x_adv
