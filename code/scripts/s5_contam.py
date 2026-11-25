"""S5: training-data contamination check.  pHash every image of the victims' released RL/SFT training sets
(DeepEyes-Datasets-47k; Mini-o3 cold-start set and VisualProbe_train; Mini-o3 RL also uses DeepEyes_train_4K, a
subset of DeepEyes-47k) and compare with every evaluation image in results/S5/manifest.jsonl.
A hit is pHash Hamming <= 6 (Hamming 0 listed as such); hits are re-checked visually on a sample before use.
Output: results/S5/contamination_<sources>.json (args: source prefixes, e.g. deepeyes47k minio3_coldstart visualprobe_train) (per eval dataset: #items whose image near-duplicates a training image)."""
import glob, hashlib, io, json, os, sys, time, warnings
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from s5_audit import phash, OUT, NEAR

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
TA = "./datasets/_train_audit"


def iter_images_parquet(f):
    pf = pq.ParquetFile(f)
    cols = [n for n in pf.schema_arrow.names]
    img_cols = [c for c in cols if "image" in c.lower()]
    for b in pf.iter_batches(columns=img_cols, batch_size=32):
        for c in img_cols:
            for v in b.column(c).to_pylist():
                vs = v if isinstance(v, list) else [v]
                for x in vs:
                    if isinstance(x, dict):
                        x = x.get("bytes")
                    if isinstance(x, (bytes, bytearray)) and len(x) > 100:
                        yield bytes(x)


def sources():
    for f in sorted(glob.glob(f"{TA}/deepeyes47k/*.parquet")):
        yield "deepeyes47k/" + os.path.basename(f), iter_images_parquet(f)
    for f in sorted(glob.glob(f"{TA}/Mini-o3__Mini-o3-Coldstart-Dataset/data/*.parquet")):
        yield "minio3_coldstart/" + os.path.basename(f), iter_images_parquet(f)
    vp = sorted(glob.glob(f"{TA}/Mini-o3__VisualProbe_train/data/*"))
    yield "visualprobe_train", (open(p, "rb").read() for p in vp if p.lower().endswith((".jpg", ".jpeg", ".png")))


def main(only=None):
    tag = '_'.join(only) if only else 'all'
    man = [json.loads(l) for l in open(f"{OUT}/manifest.jsonl")]
    ev_ph = np.array([int(r["phash"], 16) for r in man], dtype=np.uint64)
    hits = {}
    stats = {}
    for name, it in sources():
        if only and not any(name.startswith(o) for o in only):
            continue
        t = time.time(); n = 0
        for b in it:
            try:
                ph, w, h = phash(b)
            except Exception:
                continue
            n += 1
            d = np.unpackbits(np.bitwise_xor(ev_ph, np.uint64(ph)).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
            for i in np.nonzero(d <= NEAR)[0]:
                r = man[i]
                hits.setdefault(r["qid"], []).append({"source": name, "hamming": int(d[i]), "train_wh": [w, h],
                                                      "eval_wh": [r["w"], r["h"]]})
        stats[name] = {"n_images": n, "sec": round(time.time() - t, 1)}
        print(name, stats[name], "hits so far", len(hits), flush=True)
    per_ds = {}
    for r in man:
        if r["qid"] in hits:
            per_ds.setdefault(r["dataset"], []).append(r["qid"])
    out = {"threshold_hamming": NEAR, "sources": stats, "n_hit_items": {k: len(v) for k, v in per_ds.items()},
           "hit_qids": per_ds, "hits": hits}
    json.dump(out, open(f"{OUT}/contamination_{tag}.json", "w"), indent=1)
    print(json.dumps(out["n_hit_items"]))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
