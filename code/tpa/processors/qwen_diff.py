"""Differentiable re-implementation of the Qwen2-VL / Qwen2.5-VL / Qwen3-VL image pipeline
and of the victims' crop-zoom observation operators (DeepEyes: bicubic, Mini-o3: Lanczos).

Design:
- Input image x is float in [0, 1], shape (3, H, W), and may require grad.
- The official HF (transformers 5.x) torchvision backend resizes a uint8 tensor (output rounded to uint8),
  then rescales by 1/255 and normalises.  We resize the float image directly; the rounding of the
  official path is reproduced with a straight-through estimator (value = rounded, gradient = identity),
  so forward values match the official processor while gradients flow.
- Crop boxes are integer pixel boxes and are treated as constants (no gradient w.r.t. the box).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Copy of transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize (transformers 5.14.1)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resize_glm4v(height: int, width: int, factor: int, min_pixels: int, max_pixels: int,
                       temporal: int = 2) -> tuple[int, int]:
    """Copy of transformers.models.glm4v.image_processing_glm4v.smart_resize (transformers 5.14.1) for a still image
    (num_frames = temporal_factor = temporal_patch_size): the pixel budget counts the duplicated frame."""
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = temporal
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((temporal * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (temporal * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def ste_round_uint8(x255: torch.Tensor) -> torch.Tensor:
    """Round to the uint8 grid in the forward pass, identity gradient in the backward pass."""
    q = x255.clamp(0, 255).round()
    return x255 + (q - x255).detach()


@dataclass
class QwenVisionSpec:
    patch_size: int = 14
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_pixels: int = 56 * 56
    max_pixels: int = 14 * 14 * 4 * 1280
    image_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple = (0.26862954, 0.26130258, 0.27577711)
    interpolation: InterpolationMode = InterpolationMode.BICUBIC
    glm4v: bool = False            # GLM-4.1V smart_resize (duplicated frame counted in the pixel budget)

    @classmethod
    def from_hf_processor(cls, image_processor, max_pixels: int | None = None, min_pixels: int | None = None):
        size = getattr(image_processor, "size", None) or {}
        get = (lambda k: size.get(k)) if isinstance(size, dict) else (lambda k: getattr(size, k, None))
        return cls(
            patch_size=image_processor.patch_size,
            merge_size=image_processor.merge_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            min_pixels=min_pixels or get("shortest_edge") or cls.min_pixels,
            max_pixels=max_pixels or get("longest_edge") or cls.max_pixels,
            image_mean=tuple(image_processor.image_mean),
            image_std=tuple(image_processor.image_std),
            glm4v=type(image_processor).__name__.startswith("Glm4v"),
        )

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


def _to_uint8(x01: torch.Tensor) -> torch.Tensor:
    return (x01.detach() * 255.0).round().clamp(0, 255).to(torch.uint8)


def resize_like_hf(x01: torch.Tensor, out_hw: tuple[int, int], interpolation: InterpolationMode,
                   round_uint8: bool = True) -> torch.Tensor:
    """Resize a float [0,1] image (3,H,W) the way the HF torchvision backend resizes a uint8 image.
    Forward value: exactly the official uint8 torchvision result (x is assumed to lie on the uint8 grid).
    Backward: gradient of the float resize (surrogate).  Returns float in [0, 255]."""
    y = tvF.resize((x01 * 255.0).unsqueeze(0), list(out_hw), interpolation=interpolation,
                   antialias=True).squeeze(0)
    if not round_uint8:
        return y.clamp(0, 255)
    # exact value computed on CPU: the HF processor runs on CPU, and torchvision's uint8 antialiased resize
    # uses a fixed-point kernel on CPU but float-then-round on CUDA (differs by a few grey levels)
    exact = tvF.resize(_to_uint8(x01).cpu().unsqueeze(0), list(out_hw), interpolation=interpolation,
                       antialias=True).squeeze(0).to(device=y.device, dtype=y.dtype)
    return y + (exact - y).detach()


def resize_like_pil(x01: torch.Tensor, out_hw: tuple[int, int], pil_resample) -> torch.Tensor:
    """Forward value: exactly PIL.Image.resize on the uint8 image (as the victims' official tool code does).
    Backward: gradient of the torchvision float resize with the matching kernel.  Returns float in [0, 255]."""
    import numpy as np
    from PIL import Image
    interp = {Image.BICUBIC: InterpolationMode.BICUBIC, Image.LANCZOS: InterpolationMode.LANCZOS,
              Image.BILINEAR: InterpolationMode.BILINEAR}[pil_resample]
    y = tvF.resize((x01 * 255.0).unsqueeze(0), list(out_hw), interpolation=interp,
                   antialias=True).squeeze(0)
    arr = _to_uint8(x01).permute(1, 2, 0).cpu().numpy()
    ref = Image.fromarray(arr).resize((out_hw[1], out_hw[0]), resample=pil_resample)
    exact = torch.from_numpy(np.array(ref)).permute(2, 0, 1).to(device=y.device, dtype=y.dtype)
    return y + (exact - y).detach()


def qwen_preprocess(x01: torch.Tensor, spec: QwenVisionSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable equivalent of Qwen2VLImageProcessor._preprocess for one image.
    Returns (pixel_values [grid_h*grid_w, C*T*P*P], image_grid_thw [1,3])."""
    _, h, w = x01.shape
    rs = smart_resize_glm4v if spec.glm4v else smart_resize
    rh, rw = rs(h, w, spec.factor, spec.min_pixels, spec.max_pixels)
    if (rh, rw) != (h, w):
        img = resize_like_hf(x01, (rh, rw), spec.interpolation)
    else:
        img = ste_round_uint8(x01 * 255.0)
    img = img / 255.0
    mean = torch.tensor(spec.image_mean, device=img.device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor(spec.image_std, device=img.device, dtype=img.dtype).view(3, 1, 1)
    img = (img - mean) / std
    p, m, t = spec.patch_size, spec.merge_size, spec.temporal_patch_size
    gh, gw = rh // p, rw // p
    patches = img.unsqueeze(0).reshape(1, 3, gh // m, m, p, gw // m, m, p)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flat = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1).reshape(1, gh * gw, 3 * t * p * p)
    grid = torch.tensor([[1, gh, gw]], dtype=torch.long, device=x01.device)
    return flat.squeeze(0), grid


# ---------------------------------------------------------------- crop-zoom observation operators
def _clip_box(box, h, w):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError(f"degenerate box {box} for image {w}x{h}")
    return x1, y1, x2, y2


def deepeyes_zoom(x01: torch.Tensor, box, factor: int = 28, min_pixels: int = 4 * 28 * 28,
                  max_pixels: int = 16384 * 28 * 28) -> torch.Tensor:
    """DeepEyes eval_vstar.py: pil.crop(box) -> smart_resize -> PIL BICUBIC -> PNG (lossless).
    Note: the official code calls smart_resize((right-left), (bottom-top)) i.e. passes (w, h) into
    (height, width); we reproduce the call exactly, then PIL .resize((new_w, new_h)).
    Returns the zoomed observation as float [0,1] on the uint8 grid (STE)."""
    _, h, w = x01.shape
    x1, y1, x2, y2 = _clip_box(box, h, w)
    crop = x01[:, y1:y2, x1:x2]
    new_w, new_h = smart_resize(x2 - x1, y2 - y1, factor, min_pixels, max_pixels)
    from PIL import Image
    out = resize_like_pil(crop, (new_h, new_w), Image.BICUBIC)
    return out / 255.0


def minio3_zoom(obs01: torch.Tensor, box, image_size_used: tuple[int, int], resize: int = 4) -> torch.Tensor:
    """Mini-o3 function_tools.crop_image: crop, upscale by min(resize, w/crop_w, h/crop_h), min side 28,
    PIL LANCZOS.  `obs01` is the source observation (original image or an earlier observation)."""
    _, h, w = obs01.shape
    x1, y1, x2, y2 = _clip_box(box, h, w)
    crop = obs01[:, y1:y2, x1:x2]
    cw, ch = x2 - x1, y2 - y1
    if resize > 1:
        uw, uh = image_size_used
        r = min(resize, min(uw / cw, uh / ch))
        tw, th = max(28, int(cw * r)), max(28, int(ch * r))
        from PIL import Image
        return resize_like_pil(crop, (th, tw), Image.LANCZOS) / 255.0
    return ste_round_uint8(crop * 255.0) / 255.0


def _pil_nearest_index(n_in: int, n_out: int) -> torch.Tensor:
    """Source indices of PIL NEAREST resize.  Pillow (12.x, Geometry.c affine_transform, float path) starts at
    s/2 and adds s = n_in/n_out in double precision once per output pixel, truncating to int; the accumulated
    rounding error matters, so we reproduce the sequential sum (np.cumsum is sequential).  Verified bit-exact
    against Pillow 12.3 on 1500 random size pairs (code/tests/test_minio3_ops.py)."""
    import numpy as np
    s = n_in / n_out
    steps = np.full(n_out, s, dtype=np.float64)
    steps[0] = s * 0.5
    idx = np.minimum(np.cumsum(steps).astype(np.int64), n_in - 1)
    return torch.from_numpy(idx)


def resize_nearest_like_pil(x01: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """PIL Image.resize(..., NEAREST) as an index gather (bit-exact, differentiable w.r.t. the selected pixels)."""
    _, h, w = x01.shape
    oh, ow = out_hw
    iy = _pil_nearest_index(h, oh).to(x01.device)
    ix = _pil_nearest_index(w, ow).to(x01.device)
    return x01[:, iy][:, :, ix]


def minio3_process_size(w: int, h: int, max_pixels: int = 2_000_000, min_pixels: int = 40_000):
    """Size sequence of Mini-o3 `process_image` (vllm_async_engine.py:189); returns the list of (w, h) resizes."""
    import math
    steps = []
    if w * h > max_pixels:
        f = math.sqrt(max_pixels / (w * h)); w, h = int(w * f), int(h * f); steps.append((w, h))
    if w * h < min_pixels:
        f = math.sqrt(min_pixels / (w * h)); w, h = int(w * f), int(h * f); steps.append((w, h))
    if w < 28 or h < 28:
        f = 28 / min(w, h); w, h = int(w * f + 1), int(h * f + 1); steps.append((w, h))
    if w / h >= 200:
        w, h = w, int(w / 190 + 1); steps.append((w, h))
    return steps


def minio3_process(x01: torch.Tensor, max_pixels: int = 2_000_000, min_pixels: int = 40_000) -> torch.Tensor:
    """Differentiable Mini-o3 `process_image`: chained PIL NEAREST resizes into [min_pixels, max_pixels]."""
    _, h, w = x01.shape
    for (nw, nh) in minio3_process_size(w, h, max_pixels, min_pixels):
        x01 = resize_nearest_like_pil(x01, (nh, nw))
    return x01


def minio3_crop(obs01: torch.Tensor, box) -> torch.Tensor:
    """Mini-o3 crop_image(..., resize=1) as used by the async rollout: a plain PIL crop of the raw source
    observation (box already in its pixel coordinates and clipped by prepare_grounding_inputs_multi_turn)."""
    x1, y1, x2, y2 = box
    return obs01[:, y1:y2, x1:x2]
