"""S5 data audit.
  stage manifest : every item of every evaluation dataset -> results/S5/manifest.jsonl
                   (image sha256, width/height, 64-bit DCT pHash, option count, answer index)
  stage dups     : exact (sha256) and near (pHash Hamming <= 6) duplicate images within / across datasets
                   -> results/S5/duplicates.json
  stage splits   : frozen splits per protocol (seed 0, stratified) -> code/configs/splits/*.json
Usage: python s5_audit.py [manifest] [dups] [splits]   (default: all three)
"""
import collections, hashlib, io, json, os, sys, time, warnings
import numpy as np
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tpa.data.benchmarks import LOADERS, image_bytes

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")
ROOT = "."
OUT = f"{ROOT}/results/S5"
SPL = f"{ROOT}/code/configs/splits"
os.makedirs(OUT, exist_ok=True)
NEAR = 6
DATASETS = ["vstar", "hrbench4k", "hrbench8k", "mme_general", "mme_rs", "mathvista_mc", "xlrs", "vrsbench_yesno"]


def phash(b: bytes):
    im = Image.open(io.BytesIO(b))
    w, h = im.size
    if im.format == "JPEG":
        im.draft("L", (128, 128))
    im = im.convert("L").resize((32, 32), Image.LANCZOS)
    c = dctn(np.asarray(im, dtype=np.float64), norm="ortho")[:8, :8].flatten()
    bits = c > np.median(c[1:])
    return int("".join("1" if x else "0" for x in bits), 2), w, h


def stage_manifest():
    seen = {}
    with open(f"{OUT}/manifest.jsonl", "w") as f:
        for ds in DATASETS:
            t = time.time(); items = LOADERS[ds]()
            for it in items:
                k = (ds, it["image_key"])
                if k not in seen:
                    b = image_bytes(ds, it["image_key"])
                    ph, w, h = phash(b)
                    seen[k] = dict(image_sha256=hashlib.sha256(b).hexdigest(), w=w, h=h, phash=f"{ph:016x}")
                f.write(json.dumps({**it, **seen[k]}) + "\n")
            print(f"{ds:15s} {len(items):6d} items {time.time() - t:6.1f}s", flush=True)


def load_manifest():
    return [json.loads(l) for l in open(f"{OUT}/manifest.jsonl")]


def stage_dups():
    m = load_manifest()
    imgs = {}
    for r in m:
        imgs.setdefault((r["dataset"], r["image_sha256"]), r)
    keys = list(imgs)
    ph = np.array([int(imgs[k]["phash"], 16) for k in keys], dtype=np.uint64)
    ds = [k[0] for k in keys]
    near = collections.Counter(); examples = collections.defaultdict(list)
    for i in range(len(keys)):
        x = np.bitwise_xor(ph[i + 1:], ph[i])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(d <= NEAR)[0]:
            j = i + 1 + j
            exact = keys[i][1] == keys[j][1]
            pair = tuple(sorted((ds[i], ds[j])))
            near[(pair, "exact" if exact else "near")] += 1
            if len(examples[pair]) < 20:
                examples[pair].append([imgs[keys[i]]["qid"], imgs[keys[j]]["qid"], int(d[j - i - 1])])
    out = {"threshold_hamming": NEAR, "n_unique_images": {d: sum(1 for k in keys if k[0] == d) for d in DATASETS},
           "pairs": {f"{a}|{b}|{t}": n for ((a, b), t), n in sorted(near.items())},
           "examples": {f"{a}|{b}": v for (a, b), v in examples.items()}}
    json.dump(out, open(f"{OUT}/duplicates.json", "w"), indent=1)
    print(json.dumps(out["n_unique_images"]), "\n", json.dumps(out["pairs"], indent=1))


def _strat_sample(items, n, key, rng):
    """Proportional stratified sample (largest remainder), deterministic order within strata."""
    groups = collections.defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    tot = len(items); alloc = {}
    raw = {g: n * len(v) / tot for g, v in groups.items()}
    for g in raw:
        alloc[g] = int(raw[g])
    for g in sorted(raw, key=lambda g: -(raw[g] - alloc[g]))[: n - sum(alloc.values())]:
        alloc[g] += 1
    out = []
    for g in sorted(groups):
        v = groups[g]; idx = rng.permutation(len(v))[: alloc[g]]
        out += [v[i] for i in sorted(idx)]
    return out


def clusters(m):
    """Connected components of the near-duplicate graph (pHash Hamming <= NEAR) over all unique images, per dataset
    family; used as the bootstrap cluster unit and for one-question-per-image sampling."""
    keys = sorted({(r["dataset"], r["image_sha256"], r["phash"]) for r in m})
    par = list(range(len(keys)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    ph = np.array([int(k[2], 16) for k in keys], dtype=np.uint64)
    for i in range(len(keys)):
        d = np.unpackbits(np.bitwise_xor(ph[i + 1:], ph[i]).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(d <= NEAR)[0]:
            a, b = f(i), f(i + 1 + j)
            if a != b:
                par[b] = a
    root = {}
    out = {}
    for i, k in enumerate(keys):
        r = f(i)
        root.setdefault(r, keys[r][1][:16])
        out[(k[0], k[1])] = root[r]
    return out


def _one_per_image(items, rng):
    by = collections.defaultdict(list)
    for it in items:
        by[it["image_cluster"]].append(it)
    return [v[rng.integers(len(v))] for _, v in sorted(by.items())]


def _ordered(items, rng, key):
    """Stratified random permutation: interleave strata so that every prefix is approximately stratified."""
    groups = collections.defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups:
        groups[g] = [groups[g][i] for i in rng.permutation(len(groups[g]))]
    n = len(items); pos = []
    for g, v in groups.items():
        for k, it in enumerate(v):
            pos.append(((k + 0.5) / len(v), g, it))
    return [it for _, _, it in sorted(pos, key=lambda x: (x[0], x[1]))]


FIELDS = ["qid", "dataset", "question", "options", "answer_idx", "image_key", "image_sha256", "image_cluster",
          "category", "w", "h", "meta"]


def write_split(name, items, note):
    d = {"name": name, "protocol": "v2", "seed": 0, "note": note, "n": len(items),
         "n_images": len({x["image_sha256"] for x in items}), "n_clusters": len({x["image_cluster"] for x in items}), "items": [{k: x[k] for k in FIELDS} for x in items]}
    p = f"{SPL}/{name}.json"
    json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
    return p, d["n"], d["n_images"], d["n_clusters"]


def stage_splits():
    os.makedirs(SPL, exist_ok=True)
    m = load_manifest()
    cl = clusters(m)
    by = collections.defaultdict(list)
    for r in m:
        r["image_cluster"] = cl[(r["dataset"], r["image_sha256"])]
        by[r["dataset"]].append(r)
    dup = json.load(open(f"{OUT}/duplicates.json"))
    # images in V*/HR-Bench-4K (main eval); calibration and other eval pools must not near-duplicate them
    main_ph = np.array([int(r["phash"], 16) for r in by["vstar"] + by["hrbench4k"]], dtype=np.uint64)

    def far_from_main(r):
        x = np.bitwise_xor(main_ph, np.uint64(int(r["phash"], 16)))
        return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).min() > NEAR

    res = {}
    rng = np.random.default_rng(0)
    vs = _ordered(by["vstar"], rng, lambda x: x["category"])
    hr = _ordered(by["hrbench4k"], rng, lambda x: x["category"])
    res["vstar_eval"] = write_split("vstar_eval", vs, "all 191; stratified random order (prefixes are nested subsets)")
    res["hrbench4k_eval"] = write_split("hrbench4k_eval", hr, "all 200 cycle-0 questions; stratified random order")
    for name, k_v, k_h in [("s11_pilot", 50, 50), ("e1_subset", 50, 50), ("e8_subset", 25, 25)]:
        res[name] = write_split(name, vs[:k_v] + hr[:k_h], f"prefix V* {k_v} + HR-Bench-4K {k_h}")
    res["e4b_eps4"] = write_split("e4b_eps4", vs[:130], "prefix V* 130")
    res["e6_budget"] = write_split("e6_budget", vs[:100], "prefix V* 100")

    rng = np.random.default_rng(1)
    pool = [r for r in _one_per_image(by["mme_general"], rng) if far_from_main(r)]
    sub = lambda x: x["category"].split("/")[0] + "/" + x["category"].split("/")[1]
    order = _ordered(pool, rng, sub)
    ev = _strat_sample(order, 150, sub, np.random.default_rng(2))
    ev_sha = {x["image_cluster"] for x in ev}
    cal = _strat_sample([x for x in order if x["image_cluster"] not in ev_sha], 50, sub, np.random.default_rng(3))
    res["mme_general_eval"] = write_split("mme_general_eval", ev, "150, one question per image, stratified by Subtask/Task")
    res["calib_mme"] = write_split("calib_mme", cal, "50 calibration items, image-disjoint from every eval split")

    rng = np.random.default_rng(4)
    res["mathvista_mc"] = write_split("mathvista_mc", _strat_sample(by["mathvista_mc"], 150, lambda x: x["category"], rng),
                                      "150 of 540 multi-choice testmini, stratified by task")
    rng = np.random.default_rng(5)
    rs = _strat_sample(_one_per_image(by["mme_rs"], rng), 100, lambda x: x["category"], rng)
    res["e7_mme_rs"] = write_split("e7_mme_rs", rs, "100 of MME-RW-Lite Remote Sensing, one per image, stratified")
    rs_ph = np.array([int(r["phash"], 16) for r in rs], dtype=np.uint64)

    def far_from_rs(r):
        x = np.bitwise_xor(rs_ph, np.uint64(int(r["phash"], 16)))
        return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).min() > NEAR

    xl = [r for r in _one_per_image(by["xlrs"], rng) if far_from_rs(r)]
    res["e7_xlrs"] = write_split("e7_xlrs", _strat_sample(xl, 100, lambda x: x["category"], rng),
                                 "100 XLRS-Bench-lite MC, one per image, stratified by category, no near-dup with e7_mme_rs")
    vr = _one_per_image(by["vrsbench_yesno"], rng)
    yes = [x for x in vr if x["answer_idx"] == 0]; no = [x for x in vr if x["answer_idx"] == 1]
    vsel = _strat_sample(yes, 50, lambda x: x["category"], rng) + _strat_sample(no, 50, lambda x: x["category"], rng)
    res["e7_vrsbench_yesno"] = write_split("e7_vrsbench_yesno", vsel, "50 yes + 50 no, one per image, stratified by type")
    print(json.dumps(res, indent=1))
    lines = []
    for f in sorted(os.listdir(SPL)):
        lines.append(f"{hashlib.sha256(open(f'{SPL}/{f}', 'rb').read()).hexdigest()}  {f}")
    open(f"{OUT}/splits_sha256.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    st = sys.argv[1:] or ["manifest", "dups", "splits"]
    for s in st:
        {"manifest": stage_manifest, "dups": stage_dups, "splits": stage_splits}[s]()
