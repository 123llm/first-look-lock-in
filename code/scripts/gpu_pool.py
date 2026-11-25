"""Run stages of shell commands over the free GPUs of a multi-GPU machine (the 8x T4 server).

  gpu_pool.py STAGEFILE [--gpus 0,1,...] [--log_dir logs/X]

STAGEFILE: one command per line; a line "---" is a barrier (all earlier commands must finish first).
Each command runs with CUDA_VISIBLE_DEVICES set to one GPU.  A GPU is used only while no process of another user
holds memory on it (protocol budget.gpu_sharing: never preempt or kill other projects' jobs); busy GPUs are skipped and
re-checked.  A non-zero exit code stops the pool after the running commands finish.
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time


def foreign_busy() -> set[int]:
    """GPU indices on which a process not started by this pool holds memory."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    bus = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_bus_id", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    b2i = {b.split(",")[1].strip(): int(b.split(",")[0]) for b in bus}
    busy = set()
    for line in q:
        if not line.strip():
            continue
        bid, pid = [x.strip() for x in line.split(",")]
        if int(pid) not in OURS:
            busy.add(b2i.get(bid, -1))
    return busy


OURS: set[int] = set()


def free_gb() -> dict[int, float]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {int(l.split(",")[0]): float(l.split(",")[1]) / 1024 for l in out if l.strip()}


def descendants(pid: int) -> set[int]:
    out = subprocess.run(["ps", "-eo", "pid,ppid", "--no-headers"], capture_output=True, text=True).stdout.split("\n")
    kids = {}
    for l in out:
        if l.strip():
            p, pp = map(int, l.split())
            kids.setdefault(pp, []).append(p)
    seen, st = set(), [pid]
    while st:
        p = st.pop()
        if p not in seen:
            seen.add(p); st.extend(kids.get(p, []))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stagefile"); ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--log_dir", default="logs/pool")
    ap.add_argument("--slots", type=int, default=1, help="concurrent commands per GPU")
    ap.add_argument("--allow_shared", action="store_true",
                    help="also use GPUs that hold other users' processes (only with the GPU owner's permission; "
                         "their processes are never touched)")
    ap.add_argument("--min_free_gb", type=float, default=0.0,
                    help="start a new command on a GPU only if it has at least this much free memory (protects "
                         "other users' jobs and our own running shards from OOM when slots are packed)")
    ap.add_argument("--retries", type=int, default=0, help="re-queue a failed (resumable) command up to N times")
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    slots = [(g, j) for j in range(a.slots) for g in gpus]
    os.makedirs(a.log_dir, exist_ok=True)
    stages, cur = [], []
    for line in open(a.stagefile):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "---":
            stages.append(cur); cur = []
        else:
            cur.append(line)
    if cur:
        stages.append(cur)
    k = 0
    for si, stage in enumerate(stages):
        queue = list(stage)
        running = {}                       # (gpu, slot) -> (Popen, cmd, logfile, t0)
        failed = False
        tries = {c: 0 for c in stage}
        while queue or running:
            for sl, (p, c, f, t0) in list(running.items()):
                g = sl[0]
                if p.poll() is not None:
                    f.close()
                    print(time.strftime("%H:%M:%S"), f"gpu{g} rc={p.returncode} {time.time() - t0:.0f}s  {c[:150]}", flush=True)
                    del running[sl]
                    if p.returncode != 0:
                        tries[c] += 1
                        if tries[c] <= a.retries:     # commands are resumable: re-queue (e.g. OOM on a shared GPU)
                            print(time.strftime("%H:%M:%S"), f"  retry {tries[c]}/{a.retries}", flush=True)
                            queue.append(c)
                        else:
                            failed = True
            if failed:
                queue = []
            OURS.clear()
            for p, *_ in running.values():
                OURS.update(descendants(p.pid))
            busy = foreign_busy() if (queue and not a.allow_shared) else set()
            free = free_gb() if (queue and a.min_free_gb > 0) else {}
            launched_on = set()
            for sl in slots:
                g = sl[0]
                if not queue:
                    break
                if sl in running or g in busy or g in launched_on:
                    continue
                if a.min_free_gb > 0 and free.get(g, 0.0) < a.min_free_gb:
                    continue
                launched_on.add(g)          # at most one new start per GPU per poll, so free memory is re-measured
                c = queue.pop(0)
                log = open(f"{a.log_dir}/{k:03d}_gpu{g}.log", "a"); k += 1
                log.write(f"# {time.ctime()} {c}\n"); log.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                running[sl] = (subprocess.Popen(c, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT), c, log, time.time())
                print(time.strftime("%H:%M:%S"), f"gpu{g} start  {c[:150]}", flush=True)
            time.sleep(10)
        if failed:
            print(f"stage {si} had a failure; stopping", flush=True)
            sys.exit(1)
        print(time.strftime("%H:%M:%S"), f"stage {si} done ({len(stage)} commands)", flush=True)
    print("POOL DONE", flush=True)


if __name__ == "__main__":
    main()
