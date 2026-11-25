"""CPU checks: torch NEAREST == PIL NEAREST bit-for-bit on random sizes; minio3_process == official process_image."""
import math, random, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa.processors.qwen_diff import resize_nearest_like_pil, minio3_process


def official_process_image(image, max_pixels=2_000_000, min_pixels=40_000):   # copied from vllm_async_engine.py:189
    if (image.width * image.height) > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height)); image = image.resize((int(image.width * f), int(image.height * f)), resample=Image.Resampling.NEAREST)
    if (image.width * image.height) < min_pixels:
        f = math.sqrt(min_pixels / (image.width * image.height)); image = image.resize((int(image.width * f), int(image.height * f)), resample=Image.Resampling.NEAREST)
    if image.width < 28 or image.height < 28:
        f = 28 / min(image.width, image.height); image = image.resize((int(image.width * f + 1), int(image.height * f + 1)), resample=Image.Resampling.NEAREST)
    if image.width / image.height >= 200:
        image = image.resize((image.width, int(image.width / 190 + 1)), resample=Image.Resampling.NEAREST)
    return image


def t2pil(x):
    return Image.fromarray((x * 255).round().byte().permute(1, 2, 0).numpy())


random.seed(0)
bad = 0
cases = [(2246, 1582), (3840, 2160), (7680, 4320), (120, 90), (37, 410), (1, 300), (64, 64), (5000, 23)]
cases += [(random.randint(10, 3000), random.randint(10, 3000)) for _ in range(40)]
for (w, h) in cases:
    x = torch.randint(0, 256, (3, h, w)).float() / 255
    for (ow, oh) in [(random.randint(1, 2 * w), random.randint(1, 2 * h)) for _ in range(3)]:
        a = np.asarray(t2pil(x).resize((ow, oh), Image.NEAREST))
        b = np.asarray(t2pil(resize_nearest_like_pil(x, (oh, ow))))
        bad += int(not np.array_equal(a, b))
    a = np.asarray(official_process_image(t2pil(x)))
    b = np.asarray(t2pil(minio3_process(x)))
    bad += int(a.shape != b.shape or not np.array_equal(a, b))
x = torch.rand(3, 50, 60, requires_grad=True)
resize_nearest_like_pil(x, (20, 30)).sum().backward()
assert x.grad.abs().sum() > 0
print("mismatches:", bad, "of", len(cases) * 4)
assert bad == 0
