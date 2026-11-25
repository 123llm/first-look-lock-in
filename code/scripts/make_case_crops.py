"""Crops for the case-study figure (Fig.8): Mini-o3, V* direct_attributes/sa_79105.
  make_case_crops.py SV_PNG TPA_PNG   (the adversarial PNGs from results/adv on the A100)
Writes figures/src/case/*.png: downscaled global views and the zoomed views at the boxes recorded in the trajectories
(single-view and clean-zoom records: E1 eps8; TPA-R record: E4)."""
import io, sys, os
sys.path.insert(0, "./code")
from tpa import io as tio
from PIL import Image
import numpy as np
it = {i["qid"]: i for i in tio.load_split("v2_main")}["vstar/direct_attributes/sa_79105"]
clean = Image.open(io.BytesIO(tio.image_bytes_of(it))).convert("RGB")
sv, tpa = Image.open(sys.argv[1]).convert("RGB"), Image.open(sys.argv[2]).convert("RGB")
assert clean.size == sv.size == tpa.size
for a in (sv, tpa):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(clean, np.int16)).max(); print("linf", d)
out = "./figures/src/case"
B_sv, B_tpa = (450, 600, 1575, 1050), (900, 600, 2025, 1050)
small = lambda im: im.resize((600, 400), Image.LANCZOS)
small(clean).save(f"{out}/clean_global.png"); small(sv).save(f"{out}/sv_global.png"); small(tpa).save(f"{out}/tpa_global.png")
def crop(im, b, name):
    c = im.crop(b); c = c.resize((c.width * 300 // c.height, 300), Image.LANCZOS); c.save(f"{out}/{name}.png")
crop(clean, B_sv, "clean_crop_sv"); crop(sv, B_sv, "sv_crop"); crop(tpa, B_tpa, "tpa_crop")
d = (np.asarray(sv, np.int16) - np.asarray(clean, np.int16))
Image.fromarray(np.clip(128 + d * 8, 0, 255).astype(np.uint8)).resize((600, 400), Image.NEAREST).save(f"{out}/sv_delta.png")
print("ok", clean.size)
