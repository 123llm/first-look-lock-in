"""Unified multiple-choice loaders for all evaluation datasets (S5).

Every loader returns a list of dicts:
    {qid, dataset, question, options (list[str], no letter prefixes), answer_idx, image_key, category, meta}
`image_key` identifies the image inside the dataset; `image_bytes(dataset, image_key)` returns the original encoded
bytes (never re-encoded, so pixels are exactly the dataset's).  `materialize()` writes those bytes to
datasets/_materialized/<dataset>/<safe_key>.<ext> for the experiment runners.

Option order: V* and HR-Bench are shuffled with the per-question seed default_rng([0, crc32(qid)]) (V*: the
official correct option is always options[0]; HR-Bench cycle 0).  MME-RealWorld-Lite,
MathVista, XLRS-Bench keep their published order; VRSBench yes/no uses ["Yes", "No"].
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import re
import zipfile
from functools import lru_cache

import numpy as np

from tpa.data.vstar import SPLITS as VSTAR_SPLITS, shuffle_options

ROOT = __import__("os").environ.get("TPA_ROOT", ".")
D = f"{ROOT}/datasets"
LETTERS = "ABCDEFGH"
_PREFIX = re.compile(r"^\s*\(?[A-H][\).:]\s*")


def _strip(opts):
    return [_PREFIX.sub("", o).strip() for o in opts]


# --------------------------------------------------------------------------------------------- V*
def load_vstar():
    out = []
    root = f"{D}/craigwu__vstar_bench"
    for sp in VSTAR_SPLITS:
        d = os.path.join(root, sp)
        for fn in sorted(f for f in os.listdir(d) if f.endswith(".json")):
            anno = json.load(open(os.path.join(d, fn)))
            stem = fn[:-5]
            img = next(stem + e for e in (".jpg", ".JPG", ".png", ".jpeg") if os.path.exists(os.path.join(d, stem + e)))
            qid = f"vstar/{sp}/{stem}"
            opts, ans = shuffle_options(qid, anno["options"], 0)
            out.append(dict(qid=qid, dataset="vstar", question=anno["question"], options=opts, answer_idx=ans,
                            image_key=f"{sp}/{img}", category=sp, meta={"bbox": anno.get("bbox")}))
    return out


# --------------------------------------------------------------------------------------------- HR-Bench
@lru_cache(maxsize=2)
def _hr_table(res):
    import pyarrow.parquet as pq
    return pq.read_table(f"{D}/DreamMr__HR-Bench/hr_bench_{res}.parquet").to_pylist()


def load_hrbench(res="4k"):
    """Cycle 0 only: the 4 cycles repeat each question with rotated options (circular evaluation)."""
    out = []
    for r in _hr_table(res):
        if r["cycle_category"] != 0:
            continue
        qid = f"hrbench{res}/{r['index']}"
        opts0 = [r[k] for k in "ABCD"]
        opts, ans = shuffle_options(qid, opts0, "ABCD".index(r["answer"]))
        out.append(dict(qid=qid, dataset=f"hrbench{res}", question=r["question"], options=opts, answer_idx=ans,
                        image_key=str(r["index"]), category=r["category"], meta={"orig_answer": r["answer"]}))
    return out


# --------------------------------------------------------------------------------------------- MME-RealWorld-Lite
@lru_cache(maxsize=1)
def _mme_zip():
    return zipfile.ZipFile(f"{D}/yifanzhang114__MME-RealWorld-Lite/data.zip")


def load_mme(remote_sensing: bool | None = None):
    m = json.loads(_mme_zip().read("data/MME-RealWorld-Lite.json"))
    out = []
    for x in m:
        rs = x["Subtask"] == "Remote Sensing"
        if remote_sensing is not None and rs != remote_sensing:
            continue
        if x.get("Question Type", "Multiple Choice") != "Multiple Choice" or not x.get("Answer choices"):
            continue
        opts = _strip(x["Answer choices"])
        out.append(dict(qid=f"mme/{x['Question_id']}", dataset="mme_rs" if rs else "mme_general", question=x["Text"],
                        options=opts, answer_idx=LETTERS.index(x["Ground truth"].strip()[0]), image_key=x["Image"],
                        category=f"{x['Subtask']}/{x['Task']}/{x['Category']}", meta={}))
    return out


# --------------------------------------------------------------------------------------------- MathVista
@lru_cache(maxsize=1)
def _mv_table():
    import pyarrow.parquet as pq
    f = glob.glob(f"{D}/AI4Math__MathVista/data/testmini-*.parquet")[0]
    return pq.read_table(f).to_pylist()


def load_mathvista_mc():
    out = []
    for r in _mv_table():
        if r["question_type"] != "multi_choice":
            continue
        ch = list(r["choices"])
        if r["answer"] not in ch:
            continue
        out.append(dict(qid=f"mathvista/{r['pid']}", dataset="mathvista_mc", question=r["question"], options=ch,
                        answer_idx=ch.index(r["answer"]), image_key=str(r["pid"]),
                        category=(r.get("metadata") or {}).get("task", "na"), meta={}))
    return out


# --------------------------------------------------------------------------------------------- XLRS-Bench-lite
def _xlrs_files():
    return sorted(glob.glob(f"{D}/initiacms__XLRS-Bench-lite/train/*.arrow"))


def _read_arrow(f, columns=None):
    import pyarrow as pa
    src = pa.memory_map(f)
    try:
        t = pa.ipc.open_stream(src).read_all()
    except Exception:
        t = pa.ipc.open_file(src).read_all()
    return t.select(columns) if columns else t


def load_xlrs():
    out = []
    for fi, f in enumerate(_xlrs_files()):
        t = _read_arrow(f, ["path", "index", "question", "multi-choice options", "answer", "category"])
        for ri, r in enumerate(t.to_pylist()):
            opts = _strip(r["multi-choice options"])
            a = r["answer"].strip()
            if not opts or not a or a[0] not in LETTERS[:len(opts)]:
                continue
            out.append(dict(qid=f"xlrs/{os.path.basename(f)[:-6]}/{ri}", dataset="xlrs", question=r["question"],
                            options=opts, answer_idx=LETTERS.index(a[0]), image_key=f"{fi}:{ri}",
                            category=r["category"], meta={"path": r["path"], "index": r["index"]}))
    return out


# --------------------------------------------------------------------------------------------- VRSBench yes/no
def load_vrsbench_yesno():
    v = json.load(open(f"{D}/xiang709__VRSBench/VRSBench_EVAL_vqa.json"))
    out = []
    for x in v:
        g = x["ground_truth"].strip().lower()
        if g not in ("yes", "no"):
            continue
        out.append(dict(qid=f"vrsbench/{x['question_id']}", dataset="vrsbench_yesno", question=x["question"],
                        options=["Yes", "No"], answer_idx=0 if g == "yes" else 1, image_key=x["image_id"],
                        category=x["type"], meta={}))
    return out


LOADERS = {"vstar": load_vstar, "hrbench4k": lambda: load_hrbench("4k"), "hrbench8k": lambda: load_hrbench("8k"),
           "mme_general": lambda: load_mme(False), "mme_rs": lambda: load_mme(True),
           "mathvista_mc": load_mathvista_mc, "xlrs": load_xlrs, "vrsbench_yesno": load_vrsbench_yesno}


# --------------------------------------------------------------------------------------------- image bytes
@lru_cache(maxsize=1)
def _vrs_zip():
    return zipfile.ZipFile(f"{D}/xiang709__VRSBench/Images_val.zip")


@lru_cache(maxsize=1)
def _vrs_index():
    return {os.path.basename(n): n for n in _vrs_zip().namelist() if not n.endswith("/")}


@lru_cache(maxsize=4)
def _xlrs_image_col(fi):
    return _read_arrow(_xlrs_files()[fi], ["image"]).column("image").to_pylist()


def image_bytes(dataset: str, key: str) -> bytes:
    if dataset == "vstar":
        return open(f"{D}/craigwu__vstar_bench/{key}", "rb").read()
    if dataset.startswith("hrbench"):
        rows = _hr_table(dataset[len("hrbench"):])
        return base64.b64decode(next(r["image"] for r in rows if str(r["index"]) == key))
    if dataset in ("mme_general", "mme_rs"):
        return _mme_zip().read(f"data/imgs/{key}")
    if dataset == "mathvista_mc":
        r = next(r for r in _mv_table() if str(r["pid"]) == key)
        return r["decoded_image"]["bytes"]
    if dataset == "xlrs":
        fi, ri = map(int, key.split(":"))
        im = _xlrs_image_col(fi)[ri]
        return im[0]["bytes"]
    if dataset == "vrsbench_yesno":
        return _vrs_zip().read(_vrs_index()[key])
    raise KeyError(dataset)


def _ext(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    return ".bin"


def materialize(item, out_root=f"{D}/_materialized") -> str:
    """Write the original image bytes to disk (idempotent) and return the path."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", item["image_key"])
    d = os.path.join(out_root, item["dataset"])
    os.makedirs(d, exist_ok=True)
    for e in (".png", ".jpg", ".tif", ".bin"):
        p = os.path.join(d, safe + e)
        if os.path.exists(p):
            return p
    b = image_bytes(item["dataset"], item["image_key"])
    p = os.path.join(d, safe + _ext(b))
    with open(p, "wb") as f:
        f.write(b)
    return p
