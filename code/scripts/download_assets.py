"""Download models / datasets from hf-mirror with resume, and record revision + file list.

Usage (WSL):
  HF_ENDPOINT=https://hf-mirror.com venv/bin/python code/scripts/download_assets.py --batch models1
Writes a manifest line per repo to docs/download_manifest.jsonl.
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download, HfApi  # noqa: E402

ROOT = Path(__import__("os").environ.get("TPA_ROOT", "."))

BATCHES = {
    # first batch: white-box candidates and controlled pairs (S0)
    "models1": [
        ("model", "Qwen/Qwen3-VL-4B-Instruct"),
        ("model", "Qwen/Qwen3-VL-4B-Thinking"),
        ("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
        ("model", "ChenShawn/DeepEyes-7B"),
        ("model", "Mini-o3/Mini-o3-7B-v1"),
        ("model", "Osilly/Vision-R1-7B"),
        ("model", "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B"),
    ],
    # main-domain + extension datasets (S0)
    "data1": [
        ("dataset", "craigwu/vstar_bench"),
        ("dataset", "DreamMr/HR-Bench"),
        ("dataset", "Mini-o3/VisualProbe_Easy"),
        ("dataset", "Mini-o3/VisualProbe_Medium"),
        ("dataset", "Mini-o3/VisualProbe_Hard"),
        ("dataset", "yifanzhang114/MME-RealWorld-Lite"),
        ("dataset", "AI4Math/MathVista"),
        ("dataset", "xiang709/VRSBench"),
        ("dataset", "initiacms/XLRS-Bench-lite"),
    ],
    # victims added by innovations v2 (2026-11-07)
    "models3": [
        ("model", "Qwen/Qwen3-VL-8B-Thinking"),
        ("model", "TIGER-Lab/PixelReasoner-RL-v1"),
        ("model", "zai-org/GLM-4.1V-9B-Thinking"),
    ],
    # second batch, only after the S11 pilot gate
    "models2": [
        ("model", "Qwen/Qwen3-VL-8B-Instruct"),
        ("model", "Qwen/Qwen3-VL-8B-Thinking"),
        ("model", "TIGER-Lab/PixelReasoner-RL-v1"),
        ("model", "OpenGVLab/InternVL3_5-8B-HF"),
        ("model", "Video-R1/Video-R1-7B"),
        ("model", "chs20/fare2-clip"),
        ("model", "chs20/fare4-clip"),
        ("model", "chs20/tecoa4-clip"),
        ("model", "openai/clip-vit-large-patch14"),
        ("model", "openai/clip-vit-base-patch16"),
        ("model", "openai/clip-vit-base-patch32"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=sorted(BATCHES))
    ap.add_argument("--only", default=None, help="comma-separated repo ids to restrict")
    args = ap.parse_args()
    api = HfApi()
    manifest = ROOT / "docs" / "download_manifest.jsonl"
    only = set(args.only.split(",")) if args.only else None
    for kind, repo in BATCHES[args.batch]:
        if only and repo not in only:
            continue
        sub = "models" if kind == "model" else "datasets"
        dest = ROOT / sub / repo.replace("/", "__")
        t0 = time.time()
        for attempt in range(5):
            try:
                info = (api.model_info if kind == "model" else api.dataset_info)(repo)
                path = snapshot_download(repo_id=repo, repo_type=kind, revision=info.sha,
                                         local_dir=str(dest), max_workers=4)
                break
            except Exception as e:  # network hiccups: retry with backoff
                print(f"[{repo}] attempt {attempt} failed: {e}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            print(f"[{repo}] FAILED after retries", flush=True)
            with open(manifest, "a") as f:
                f.write(json.dumps({"repo": repo, "kind": kind, "status": "failed"}) + "\n")
            continue
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        rec = {"repo": repo, "kind": kind, "revision": info.sha, "path": str(dest),
               "bytes": size, "seconds": round(time.time() - t0), "status": "ok",
               "date": time.strftime("%Y-%m-%d %H:%M")}
        print(json.dumps(rec), flush=True)
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
