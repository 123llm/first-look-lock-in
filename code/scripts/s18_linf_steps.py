"""S18: measured L_inf of the E3 intermediate-step images (their index rows carry no linf field).
  s18_linf_steps.py <model> [...]  -> prints per-level max L_inf (8-bit) over all items."""
import io, json, sys, os
import numpy as np
from PIL import Image
sys.path.insert(0, os.environ.get("TPA_ROOT", ".") + "/code")
from tpa import io as tio
Image.MAX_IMAGE_PIXELS = None
items = {it["qid"]: it for it in tio.load_split("v2_main")}
for m in sys.argv[1:]:
    for s in (2, 5, 10, 25):
        idx = f"{tio.R}/results/adv/E3/v2_main/sv_s{s}@{m}__eps8__seed0/index.jsonl"
        mx, n = 0, 0
        for r in tio.read_jsonl(idx):
            a = np.asarray(Image.open(f"{tio.R}/{r['png']}").convert("RGB")).astype(int)
            c = np.asarray(Image.open(io.BytesIO(tio.image_bytes_of(items[r["qid"]]))).convert("RGB")).astype(int)
            mx = max(mx, int(np.abs(a - c).max())); n += 1
        print(json.dumps(dict(model=m, step=s, n=n, linf_max_255=mx)), flush=True)
