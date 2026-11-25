"""PixelReasoner crop operator: the differentiable path must reproduce the official PIL code bit for bit
(crop_image_normalized + resize_cropped, third_party/Pixel-Reasoner experience_maker.py), and pass a gradient."""
import os, sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa.models.pixelreasoner_agent import pr_crop_box, pr_observe, pr_resize_size, PR_MIN, PR_MAX
from tpa.processors.qwen_diff import smart_resize


def official(img: Image.Image, bb, padding=0.1):
    img_x, img_y = img.size
    if bb[0] < 1 and bb[1] < 1 and bb[2] < 1 and bb[3] < 1:
        n = (float(bb[0]) - padding, float(bb[1]) - padding, float(bb[2]) + padding, float(bb[3]) + padding)
    else:
        n = (float(bb[0]) / img_x - padding, float(bb[1]) / img_y - padding, float(bb[2]) / img_x + padding,
             float(bb[3]) / img_y + padding)
    x1, y1, x2, y2 = [min(max(0, v), 1) for v in n]
    c = img.crop((x1 * img_x, y1 * img_y, x2 * img_x, y2 * img_y))
    w, h = c.size
    assert w > 28 and h > 28
    rh, rw = smart_resize(h, w, 28, PR_MIN, PR_MAX)
    return c.resize((rw, rh))


def test_bit_exact_and_grad():
    rng = np.random.default_rng(0)
    n_ok = 0
    for t in range(40):
        H, W = int(rng.integers(300, 1600)), int(rng.integers(300, 1600))
        arr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        a, b = sorted(rng.uniform(0, 1, 2)); c, d = sorted(rng.uniform(0, 1, 2))
        bb = [round(a, 3), round(c, 3), round(b, 3), round(d, 3)] if t % 4 else \
             [float(int(a * W)), float(int(c * H)), float(int(b * W) + 1), float(int(d * H) + 1)]
        try:
            ref = official(img, bb)
        except AssertionError:
            continue
        box = pr_crop_box(W, H, bb)
        x = torch.from_numpy(arr).permute(2, 0, 1).float().div(255).requires_grad_(True)
        z = pr_observe(x, box)
        assert (z.shape[-1], z.shape[-2]) == ref.size == pr_resize_size(box[2] - box[0], box[3] - box[1])
        got = (z.detach() * 255).round().byte().permute(1, 2, 0).numpy()
        assert np.array_equal(got, np.asarray(ref)), f"case {t}: max diff {np.abs(got.astype(int) - np.asarray(ref)).max()}"
        z.sum().backward()
        assert x.grad[:, box[1]:box[3], box[0]:box[2]].abs().sum() > 0
        n_ok += 1
    assert n_ok >= 20, n_ok
    print("bit-exact cases:", n_ok)


if __name__ == "__main__":
    test_bit_exact_and_grad()
