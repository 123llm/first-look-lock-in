"""S3 item 1 (partial): alignment of the differentiable Qwen preprocessing / zoom with the official paths.
Run: venv/bin/python -m pytest -q code/tests/test_qwen_diff.py  (or as a script to print the report)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__import__("os").environ.get("TPA_ROOT", "."))
sys.path.insert(0, str(ROOT / "code"))
from tpa.processors.qwen_diff import (QwenVisionSpec, qwen_preprocess, deepeyes_zoom, minio3_zoom,  # noqa: E402
                                      smart_resize)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor  # noqa: E402

VSTAR = ROOT / "datasets" / "craigwu__vstar_bench"


def load_images(n=20):
    files = sorted(p for p in VSTAR.rglob("*.jpg"))[:n]
    return [(p.name, Image.open(p).convert("RGB")) for p in files]


def to01(pil):
    return torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0


def report():
    proc = Qwen2VLImageProcessor()
    out = {"preprocess": [], "deepeyes_zoom": [], "minio3_zoom": [], "grad": None}
    for max_pixels in (1280 * 28 * 28, 16384 * 28 * 28):
        spec = QwenVisionSpec(max_pixels=max_pixels)
        for name, pil in load_images():
            ref = proc(images=[pil], return_tensors="pt", size={"shortest_edge": spec.min_pixels,
                                                               "longest_edge": max_pixels})
            pv, grid = qwen_preprocess(to01(pil), spec)
            assert torch.equal(grid, ref["image_grid_thw"]), (grid, ref["image_grid_thw"])
            d = (pv - ref["pixel_values"]).abs()
            out["preprocess"].append({"img": name, "max_pixels": max_pixels, "max_abs": d.max().item(),
                                      "mean_abs": d.mean().item()})
    # zoom vs PIL reference (DeepEyes: BICUBIC; Mini-o3: LANCZOS), 3 boxes per image
    rng = np.random.default_rng(0)
    for name, pil in load_images():
        w, h = pil.size
        for _ in range(3):
            cw, ch = int(rng.integers(64, max(65, w // 3))), int(rng.integers(64, max(65, h // 3)))
            x1, y1 = int(rng.integers(0, w - cw)), int(rng.integers(0, h - ch))
            box = (x1, y1, x1 + cw, y1 + ch)
            nw, nh = smart_resize(cw, ch, 28, 4 * 28 * 28, 16384 * 28 * 28)
            ref = np.asarray(pil.crop(box).resize((nw, nh), resample=Image.BICUBIC)).astype(np.float32)
            ours = deepeyes_zoom(to01(pil), box).permute(1, 2, 0).numpy() * 255
            out["deepeyes_zoom"].append({"img": name, "max_abs_255": float(np.abs(ours - ref).max()),
                                         "mean_abs_255": float(np.abs(ours - ref).mean())})
            r = min(4, min(w / cw, h / ch))
            tw, th = max(28, int(cw * r)), max(28, int(ch * r))
            ref2 = np.asarray(pil.crop(box).resize((tw, th), resample=Image.LANCZOS)).astype(np.float32)
            ours2 = minio3_zoom(to01(pil), box, (w, h)).permute(1, 2, 0).numpy() * 255
            out["minio3_zoom"].append({"img": name, "max_abs_255": float(np.abs(ours2 - ref2).max()),
                                       "mean_abs_255": float(np.abs(ours2 - ref2).mean())})
    # gradient through zoom + preprocess
    name, pil = load_images(1)[0]
    x = to01(pil).requires_grad_(True)
    w, h = pil.size
    obs = deepeyes_zoom(x, (w // 4, h // 4, w // 2, h // 2))
    pv, _ = qwen_preprocess(obs, QwenVisionSpec())
    pv.square().mean().backward()
    g = x.grad
    inside = g[:, h // 4:h // 2, w // 4:w // 2].abs().sum().item()
    outside = g.abs().sum().item() - inside
    out["grad"] = {"finite": bool(torch.isfinite(g).all()), "inside_box_l1": inside, "outside_box_l1": outside}
    return out


def summarize(out):
    s = {}
    for k in ("preprocess", "deepeyes_zoom", "minio3_zoom"):
        key = "max_abs" if k == "preprocess" else "max_abs_255"
        vals = [r[key] for r in out[k]]
        means = [r["mean_abs" if k == "preprocess" else "mean_abs_255"] for r in out[k]]
        s[k] = {"n": len(vals), "worst_max_abs": max(vals), "avg_mean_abs": float(np.mean(means))}
    s["grad"] = out["grad"]
    return s


def test_alignment():
    s = summarize(report())
    assert s["grad"]["finite"] and s["grad"]["inside_box_l1"] > 0 and s["grad"]["outside_box_l1"] == 0
    # normalised pixel units: one uint8 step ~ 1/255/0.27 ~ 0.0146
    assert s["preprocess"]["worst_max_abs"] <= 0.03


if __name__ == "__main__":
    full = report()
    s = summarize(full)
    dst = ROOT / "results" / "S3"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "processor_alignment.json").write_text(json.dumps({"summary": s, "detail": full}, indent=1))
    print(json.dumps(s, indent=1))
