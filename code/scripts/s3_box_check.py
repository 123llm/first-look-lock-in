"""S3: validate the box-coordinate convention of each victim by comparing executed zoom boxes (mapped to
original pixels by QwenAgent) with the V* ground-truth target boxes ([x, y, w, h] in original pixels).
Reports, per trajectory file, the fraction of first boxes that contain the centre of some GT target box and
the median IoU-with-best-GT.  Also re-maps the raw boxes under the alternative convention for comparison."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tpa.data.vstar import load_vstar
from tpa.processors.qwen_diff import smart_resize

root = "."
gt = {s.image_id: (bb, p) for s, p, bb in load_vstar(f"{root}/datasets/craigwu__vstar_bench")}


def contains_centre(b, g):
    cx, cy = g[0] + g[2] / 2, g[1] + g[3] / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def iou(b, g):
    g = (g[0], g[1], g[0] + g[2], g[1] + g[3])
    ix = max(0, min(b[2], g[2]) - max(b[0], g[0])); iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
    inter = ix * iy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    return inter / ua if ua > 0 else 0.0


def remap(b, h, w, used, alt, factor, max_pixels):
    """b: box in original pixels produced under convention `used`; return it under convention `alt`."""
    rh, rw = smart_resize(h, w, factor, 56 * 56 if factor == 28 else 65536, max_pixels)
    if used == "abs_resized":
        raw = (b[0] * rw / w, b[1] * rh / h, b[2] * rw / w, b[3] * rh / h)
    else:
        raw = (b[0] * 1000 / w, b[1] * 1000 / h, b[2] * 1000 / w, b[3] * 1000 / h)
    if alt == "abs_resized":
        return (raw[0] * w / rw, raw[1] * h / rh, raw[2] * w / rw, raw[3] * h / rh)
    return (raw[0] * w / 1000, raw[1] * h / 1000, raw[2] * w / 1000, raw[3] * h / 1000)


from PIL import Image
out = {}
for tag, used, factor in [("deepeyes", "abs_resized", 28), ("qwen3vl4b_thinking", "rel1000", 32)]:
    f = f"{root}/results/S3/model_check_{tag}_traj.jsonl"
    if not os.path.exists(f):
        continue
    alt = "rel1000" if used == "abs_resized" else "abs_resized"
    res = {used: [], alt: []}
    for line in open(f):
        d = json.loads(line)
        boxes = [b for b in d["boxes"] if b]
        if not boxes:
            continue
        g_list, p = gt[d["id"]]
        w, h = Image.open(p).size
        b = boxes[0]
        for conv, bb in ((used, b), (alt, remap(b, h, w, used, alt, factor, 1003520))):
            res[conv].append((any(contains_centre(bb, g) for g in g_list), max(iou(bb, g) for g in g_list)))
    summ = {}
    for conv, v in res.items():
        if v:
            ious = sorted(x[1] for x in v)
            summ[conv] = {"n": len(v), "centre_hit": sum(x[0] for x in v) / len(v), "median_iou": ious[len(ious) // 2]}
    out[tag] = {"used": used, **summ}
    print(tag, json.dumps(out[tag]))
json.dump(out, open(f"{root}/results/S3/box_convention_check.json", "w"), indent=1)
