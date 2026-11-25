"""Unified result schema and paths (protocol_v2 `results:`).

Two stages, so that attack generation and evaluation can run on different models / at different times:
  adversarial images  results/adv/<E>/<split>/<method>@<source>__eps<k>__seed<s>/<qid_file>.png
                      + index.jsonl  one ADV row per item (attack cost and summary log)
  evaluations         results/raw/<E>/<split>__<eval_model>__<method>@<source>__eps<k>__seed<s>.jsonl
                      one EVAL row per item (clean record + adversarial record from tpa.rollout.record_item)
  clean cache         results/raw/clean/<split>__<eval_model>.jsonl
Every output file has a sibling <file>.meta.json with git commit, protocol sha256, split sha256, model revisions,
command line.  `source` is the model whose gradients built the image ("none" for surrogate/generator methods,
whose surrogate is named in the method, e.g. mca_style@clip3).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

R = __import__("os").environ.get("TPA_ROOT", ".")
PROTOCOL_ID = os.environ.get("TPA_PROTOCOL", "v2")          # v2: NF4 LLM, 15 GB cap (RTX 4080); v3: bf16, 30 GB cap
PROTOCOL = f"{R}/code/configs/protocol_{PROTOCOL_ID}.yaml"


def _protocol_hw():
    import yaml
    p = yaml.safe_load(open(PROTOCOL))
    q = p["models"]["quantization"]["llm"]
    return ("nf4" if q == "nf4" else "bf16"), float(p["budget"].get("vram_cap_gb", 15.0))


PRECISION, VRAM_GB = _protocol_hw()     # every record carries `precision`; clean caches are only reused within it
VRAM_GB = float(os.environ.get("TPA_VRAM_GB", VRAM_GB))   # per-process share when eval shards run concurrently
Image.MAX_IMAGE_PIXELS = None

ADV_FIELDS = ["qid", "split", "method", "source", "eps", "seed", "png", "png_sha256", "clean_sha256", "linf_255",
              "attack"]
EVAL_FIELDS = ["qid", "split", "cluster", "eval_model", "method", "source", "eps", "seed", "png_sha256", "clean",
               "adv", "target", "contaminated"]


def qid_file(qid: str) -> str:
    return qid.replace("/", "__").replace(":", "_")


def load_split(name: str) -> list[dict]:
    d = json.load(open(f"{R}/code/configs/splits/{name}.json"))
    return [dict(it, split=name) for it in d["items"]]


def image_bytes_of(item: dict) -> bytes:
    from tpa.data.benchmarks import image_bytes
    return image_bytes(item["dataset"], item["image_key"])


def to01(b: bytes) -> torch.Tensor:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def contaminated_qids() -> set[str]:
    out = set()
    for f in ("contamination_deepeyes47k_minio3_coldstart", "contamination_visualprobe_train"):
        p = f"{R}/results/S5/{f}.json"
        if os.path.exists(p):
            out |= set(json.load(open(p))["hit_qids"])
    return out


def adv_dir(exp: str, split: str, method: str, source: str, eps: int, seed: int) -> str:
    return f"{R}/results/adv/{exp}/{split}/{method}@{source}__eps{eps}__seed{seed}"


def parse_adv_spec(spec: str):
    """'<method>@<source>__eps<k>__seed<s>' -> (method, source, eps, seed)."""
    method_source, eps, seed = spec.split("__")
    method, source = method_source.split("@")
    return method, source, int(eps[3:]), int(seed[4:])


def load_adv_index(spec: str, exp: str, split: str, adv_from: str = "") -> dict:
    """qid -> ADV row (last row wins).  By default from results/adv/<exp>/<split>/<spec>; `adv_from` =
    'E4:vstar_eval,E4:hrbench4k_eval' reuses images generated for another (exp, split), e.g. the E1/E8 subsets,
    whose items are prefixes of the evaluation splits."""
    method, source, eps, seed = parse_adv_spec(spec)
    srcs = [s.split(":") for s in adv_from.split(",")] if adv_from else [(exp, split)]
    rows = {}
    for e, sp in srcs:
        for r in read_jsonl(f"{adv_dir(e, sp, method, source, eps, seed)}/index.jsonl"):
            rows[r["qid"]] = r
    return rows


def raw_path(exp: str, split: str, eval_model: str, method: str, source: str, eps: int, seed: int) -> str:
    return f"{R}/results/raw/{exp}/{split}__{eval_model}__{method}@{source}__eps{eps}__seed{seed}.jsonl"


def clean_path(split: str, eval_model: str) -> str:
    return f"{R}/results/raw/clean/{split}__{eval_model}{CLEAN_SUFFIX}.jsonl"


CLEAN_SUFFIX = "" if PRECISION == "nf4" else f".{PRECISION}"     # NF4 (v2) and bf16 (v3) caches never share a file


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def append_jsonl(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import fcntl
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with open(path, "a") as f:                       # several eval shards may append to the same file
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def cap_oom_gb(err) -> float:
    """The per-process cap (GiB) that an OOM error message hit, or 0.0 if the error is not an OOM against the cap
    (another error, a message truncated before the cap, or a device that itself lacked the requested memory)."""
    import re
    m = re.search(r"Tried to allocate ([\d.]+) ([GM])iB.*?of which ([\d.]+) ([GM])iB is free.*?([\d.]+) GiB allowed",
                  err or "", re.S)
    if not m:
        return 0.0
    gib = lambda v, u: float(v) / (1024 if u == "M" else 1)
    return float(m.group(5)) if gib(m.group(3), m.group(4)) > gib(m.group(1), m.group(2)) else 0.0


def gpu_admission():
    """Shared-GPU admission (scheduling only: no effect on any result).  If $TPA_ROOT/logs/gpu_max_procs exists and
    holds "N [M]", wait until fewer than N processes of this project (cwd == TPA_ROOT, same user) hold a CUDA context
    and at least M GB are free, then return an exclusive lock that the caller releases (gpu_release) once its own
    CUDA context exists, so that simultaneous launches are admitted one at a time.  Without the file: no-op.
    Reason: on the shared A100 several queues launched against the free memory of the moment and then grew together
    until the device was full, so items failed with OOM after hours of work."""
    cfg = f"{R}/logs/gpu_max_procs"
    if not os.path.exists(cfg) or not torch.cuda.is_available() or torch.cuda.is_initialized():
        return None             # no limit, no GPU, or this process was already admitted (second model load)
    import fcntl, time
    lock = open(f"{R}/logs/gpu_admission.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    said = False
    while True:
        try:
            v = open(cfg).read().split()
            n, m = int(v[0]), float(v[1]) if len(v) > 1 else 0.0
        except Exception:
            return lock
        m = max(m, VRAM_GB)         # a process with a larger cap waits until that much is free
        budget = float(v[2]) if len(v) > 2 else 0.0   # optional: others' use + our caps + own cap <= budget (GB)
        q = [l.split(",") for l in subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines() if l.strip()]
        mine, caps, used_mine = 0, 0.0, 0.0
        for p, u in ((a.strip(), float(b)) for a, b in q):
            try:
                if os.stat(f"/proc/{p}").st_uid == os.getuid() and os.readlink(f"/proc/{p}/cwd") == os.path.realpath(R):
                    mine += 1; used_mine += u / 1024
                    env = dict(e.split("=", 1) for e in open(f"/proc/{p}/environ").read().split("\0") if "=" in e)
                    caps += max(float(env.get("TPA_VRAM_GB", VRAM_GB)), u / 1024)
            except (OSError, ValueError):
                pass
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.replace(",", " ").split()
        free_gb = float(smi[0]) / 1024 if smi else 0.0
        others = max(float(smi[1]) / 1024 - used_mine, 0.0) if len(smi) > 1 else 0.0
        fits = budget <= 0 or others + caps + VRAM_GB <= budget
        if n <= 0 or (mine < n and free_gb >= m and fits):
            if said:
                print(f"[gpu_admission] admitted ({mine} running, {free_gb:.1f} GB free)", flush=True)
            return lock
        if not said:
            print(f"[gpu_admission] waiting: {mine} project processes on the GPU (max {n}), {free_gb:.1f} GB free "
                  f"(need {m:g}); caps {caps:.0f} + own {VRAM_GB:g} + others {others:.1f} (budget {budget:g})",
                  flush=True)
            said = True
        time.sleep(30)


def gpu_release(lock):
    if lock is not None:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", R, "rev-parse", "HEAD"], text=True).strip()
    except Exception:                           # server copy without .git: commit written at sync time
        p = f"{R}/code/GIT_COMMIT"
        return open(p).read().strip() if os.path.exists(p) else "unknown"


def model_revision(path: str) -> str:
    """sha256 of config.json + the sorted weight-file names and sizes (cheap, identifies the local snapshot)."""
    if not path or not os.path.isdir(path):
        return "n/a"
    h = hashlib.sha256()
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        if f == "config.json":
            h.update(open(p, "rb").read())
        elif f.endswith((".safetensors", ".bin", ".pth")):
            h.update(f"{f}:{os.path.getsize(p)}".encode())
    return h.hexdigest()[:16]


def write_meta(path: str, split: str, models: dict[str, str], **extra):
    split_file = f"{R}/code/configs/splits/{split}.json"
    meta = dict(git_commit=_git_commit(), protocol=PROTOCOL_ID, protocol_sha256=sha256(open(PROTOCOL, "rb").read()),
                precision=PRECISION, vram_cap_gb=VRAM_GB, gpu=_gpu_name(),
                split_sha256=sha256(open(split_file, "rb").read()) if os.path.exists(split_file) else None,
                model_revisions={k: model_revision(v) for k, v in models.items()}, cmdline=" ".join(sys.argv), **extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta_path = path + ".meta.json" if not os.path.isdir(path) else os.path.join(path, "meta.json")
    json.dump(meta, open(meta_path, "w"), indent=1)
    return meta
