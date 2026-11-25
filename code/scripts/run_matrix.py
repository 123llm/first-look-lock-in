"""Experiment matrix (protocol_v2) as explicit command lists.  Every command is resumable, so the whole list can be
re-run after an interruption.  Before each command the runner waits until no foreign process uses the GPU
(protocol budget.gpu_sharing: never preempt other projects' jobs).

  run_matrix.py --exp S11 --dry      # print the commands
  run_matrix.py --exp S11            # run them; log to logs/<exp>/<nn>_<name>.log

Experiments: S11 (pilot), E4 (includes the E3 single-view arms of the white-box models), E3 (Mini-o3 arm),
E4b, E1, E2, E5, E6, E7, E8.  Generation is grouped by the attacked model and evaluation by the evaluation
model, so each model is loaded once per command.
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time

R = __import__("os").environ.get("TPA_ROOT", ".")
PY = f"{R}/venv/bin/python"
WB = ["deepeyes_7b", "qwen3vl_4b_thinking"]
OFF = {"deepeyes_7b": "deepeyes_7b_tool_off", "qwen3vl_4b_thinking": "qwen3vl_4b_thinking_tool_off",
       "minio3_7b": "minio3_7b_tool_off"}
MAIN = ["vstar_eval", "hrbench4k_eval"]
TRANSFER = ["minio3_7b", "vision_r1_7b", "vlaa_thinker_7b"]
MCA, RB = "mca_style@clip3", "reasonbreak@qwen3vl_4b_instruct"
H4_GRID = {"qwen3vl_4b_thinking": [(512, 1), (1536, 3), (4096, 6)], "deepeyes_7b": [(1024, 1), (1024, 3), (1024, 6)],
           "vision_r1_7b": [(256, 0), (768, 0), (2048, 0)]}


def gen(exp, split, method, source, eps=8, extra=""):
    return f"{PY} code/scripts/gen_adv.py --exp {exp} --split {split} --method {method} --source {source} --eps {eps} {extra}".strip()


def ev(exp, split, adv, model, extra=""):
    return f"{PY} code/scripts/eval_adv.py --exp {exp} --split {split} --adv {adv} --eval_model {model} {extra}".strip()


def spec(method, source, eps=8):
    return f"{method}@{source}__eps{eps}__seed0"


def plan(exp):
    C = []
    if exp == "S11":
        for m in WB:
            C += [gen("S11", "s11_pilot", "single_view", m), gen("S11", "s11_pilot", "tpa", m)]
        for m in WB:
            for em in (m, OFF[m]):
                for meth in ("single_view", "tpa"):
                    C.append(ev("S11", "s11_pilot", spec(meth, m), em))
    elif exp == "E4":
        for sp in MAIN + ["mme_general_eval"]:
            for m in WB:
                if sp == "mme_general_eval" and m != "deepeyes_7b":
                    continue                      # MME-RW-Lite: DeepEyes white box only (S4 budget)
                C += [gen("E4", sp, meth, m) for meth in ("single_view", "tpa", "eot")]
            if sp != "mme_general_eval":
                C += [gen("E4", sp, "mca_style", "clip3"), gen("E4", sp, "reasonbreak", "qwen3vl_4b_instruct")]
        for sp in MAIN + ["mme_general_eval"]:
            for m in WB:
                if sp == "mme_general_eval" and m != "deepeyes_7b":
                    continue
                for meth in ("single_view", "tpa"):
                    C += [ev("E4", sp, spec(meth, m), m, "--sampled 0,1,2"), ev("E4", sp, spec(meth, m), OFF[m])]
                C.append(ev("E4", sp, spec("eot", m), m, "--sampled 0,1,2"))
                if sp != "mme_general_eval":
                    C += [ev("E4", sp, f"{MCA}__eps8__seed0", m), ev("E4", sp, f"{RB}__eps8__seed0", m)]
        for t in TRANSFER:                        # transfer on V* only
            for m in WB:
                C += [ev("E4", "vstar_eval", spec(meth, m), t) for meth in ("single_view", "tpa", "eot")]
            C += [ev("E4", "vstar_eval", f"{MCA}__eps8__seed0", t), ev("E4", "vstar_eval", f"{RB}__eps8__seed0", t)]
    elif exp == "E3":                             # Mini-o3 arm of H2a (white-box single view through its own loop)
        for sp in MAIN:
            C += [gen("E3", sp, "single_view", "minio3_7b")]
            C += [ev("E3", sp, spec("single_view", "minio3_7b"), em) for em in ("minio3_7b", OFF["minio3_7b"])]
    elif exp == "E4b":
        for m in WB:
            C += [gen("E4b", "e4b_eps4", meth, m, 4) for meth in ("single_view", "tpa", "eot")]
        for m in WB:
            C += [ev("E4b", "e4b_eps4", spec(meth, m, 4), m) for meth in ("single_view", "tpa", "eot")]
    elif exp == "E1":
        frm = "--adv_from E4:vstar_eval,E4:hrbench4k_eval"
        attacks = [spec("tpa", "deepeyes_7b"), spec("single_view", "deepeyes_7b"), f"{MCA}__eps8__seed0", f"{RB}__eps8__seed0"]
        models = ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "vision_r1_7b", "vlaa_thinker_7b", "qwen25vl_7b"]
        for em in models:
            for adv in attacks:
                C.append(ev("E1", "e1_subset", adv, em, frm))
                C.append(f"{PY} code/scripts/h1_caption_judge.py caption --exp E1 --split e1_subset --adv {adv} --eval_model {em} {frm}")
        C.append(f"{PY} code/scripts/h1_caption_judge.py judge --exp E1 --judge qwen3vl_8b_instruct")
    elif exp == "E2":
        m = "qwen3vl_4b_thinking"
        C += [gen("E2", "mathvista_mc", meth, m) for meth in ("single_view", "tpa")]
        C += [ev("E2", "mathvista_mc", spec(meth, m), m, "--sampled 0,1,2") for meth in ("single_view", "tpa")]
    elif exp == "E5":
        pairs = [("qwen25vl_7b", ["deepeyes_7b", "minio3_7b", "vision_r1_7b", "vlaa_thinker_7b"]),
                 ("qwen3vl_4b_instruct", ["qwen3vl_4b_thinking"]), ("deepeyes_7b_tool_off", ["deepeyes_7b"])]
        for sp in MAIN:
            for b, rs in pairs:
                C.append(gen("E5", sp, "single_view", b))
                C.append(ev("E5", sp, spec("single_view", b), b))
                for r in rs:
                    C.append(ev("E5", sp, spec("single_view", b), r))
                    C.append(ev("E5", sp, spec("single_view", b), r, "--token_budget 512 --call_cap 0"))
            for r in ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking", "vision_r1_7b", "vlaa_thinker_7b"]:
                C.append(f"{PY} code/scripts/e5_regrounding.py --split {sp} --model {r}")
    elif exp == "E6":
        sp = "e6_budget"
        for m, grid in H4_GRID.items():
            src = "qwen25vl_7b" if m == "vision_r1_7b" else m       # Vision-R1: non-adaptive only
            C.append(gen("E6", sp, "single_view", src))
            for tok, calls in grid:
                C.append(ev("E6", sp, spec("single_view", src), m, f"--token_budget {tok} --call_cap {calls}"))
                if m != "vision_r1_7b":            # adaptive TPA at each budget, T/2 steps
                    v = f"_b{tok}c{calls}"
                    C.append(gen("E6", sp, "tpa", m, 8, f"--steps 25 --token_budget {tok} --call_cap {calls} --max_turns {calls + 1} --variant {v}"))
                    C.append(ev("E6", sp, spec("tpa" + v, m), m, f"--token_budget {tok} --call_cap {calls}"))
    elif exp == "E7":
        for sp in ["e7_mme_rs", "e7_xlrs", "e7_vrsbench_yesno"]:
            C += [gen("E7", sp, meth, "deepeyes_7b") for meth in ("single_view", "tpa")]
            for meth in ("single_view", "tpa"):
                C += [ev("E7", sp, spec(meth, "deepeyes_7b"), em) for em in ("deepeyes_7b", "deepeyes_7b_tool_off",
                                                                              "qwen3vl_4b_thinking", "qwen3vl_4b_thinking_tool_off")]
    elif exp == "E8":
        m, sp = "qwen3vl_4b_thinking", "e8_subset"
        frm = "--adv_from E4:vstar_eval,E4:hrbench4k_eval"
        variants = [("_noprobe", "--no_probes"), ("_noresample", "--no_resample"), ("_K10", "--K 10"), ("_K50", "--K 50"),
                    ("_w128", "--tf_window 128")]
        for v, arg in variants:
            C.append(gen("E8", sp, "tpa", m, 8, f"{arg} --variant {v}"))
        for eps in (2, 16):
            C.append(gen("E8", sp, "tpa", m, eps))
        C.append(gen("E8", sp, "tpa", m, 8, "--target lowest_clean_prob --variant _targeted"))
        C.append(ev("E8", sp, spec("tpa", m), m, frm))        # default TPA: reuse the E4 images
        C.append(ev("E8", sp, spec("single_view", m), m, frm))
        C.append(ev("E8", sp, spec("eot", m), m, frm))
        for v, _ in variants + [("_targeted", "")]:
            C.append(ev("E8", sp, spec("tpa" + v, m), m))
        for eps in (2, 16):
            C.append(ev("E8", sp, spec("tpa", m, eps), m))
    else:
        raise SystemExit(f"unknown experiment {exp}")
    return C


def foreign_gpu_mib() -> int:
    """VRAM used by processes that are not this runner's children (nvidia-smi total minus nothing: the runner only
    starts a command when the GPU is idle, so any use at that point is foreign)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else 0


def wait_gpu(limit_mib=1500, poll=120):
    while (u := foreign_gpu_mib()) > limit_mib:
        print(time.strftime("%H:%M:%S"), f"GPU busy ({u} MiB), waiting", flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--order", default="", help="comma-separated command indices to run in this order (default: all from --start)")
    ap.add_argument("--par", type=int, default=1, help="run each eval_adv command as this many concurrent shards")
    a = ap.parse_args()
    sys.path.insert(0, f"{R}/code")
    cmds = plan(a.exp)
    if a.dry:
        for i, c in enumerate(cmds):
            print(f"{i:03d} {c.replace(PY + ' ', '')}")
        print(f"{len(cmds)} commands")
        return
    os.makedirs(f"{R}/logs/{a.exp}", exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    idx = [int(i) for i in a.order.split(",")] if a.order else list(range(a.start, len(cmds)))
    for i in idx:
        c = cmds[i]
        wait_gpu()
        name = c.split("code/scripts/")[1].split()[0].replace(".py", "")
        log = f"{R}/logs/{a.exp}/{i:03d}_{name}.log"
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), f"[{i}/{len(cmds)}]", c.replace(PY + " ", ""), flush=True)
        npar = a.par if name == "eval_adv" else 1
        if npar == 1:
            with open(log, "a") as f:
                f.write(f"\n# {time.ctime()} {c}\n"); f.flush()
                rc = subprocess.run(c, shell=True, cwd=R, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        else:                                   # evaluation shards share the GPU; each gets 1/npar of the VRAM cap
            from tpa import io as tio
            penv = dict(env, TPA_VRAM_GB=str(tio.VRAM_GB / npar))
            procs, fs = [], []
            for k in range(npar):
                f = open(log.replace(".log", f".s{k}.log"), "a")
                f.write(f"\n# {time.ctime()} {c} --shard {k}/{npar}\n"); f.flush()
                procs.append(subprocess.Popen(f"{c} --shard {k}/{npar}", shell=True, cwd=R, env=penv, stdout=f,
                                              stderr=subprocess.STDOUT)); fs.append(f)
            rc = max(p.wait() for p in procs)
            for f in fs:
                f.close()
        print(time.strftime("%H:%M:%S"), f"  rc={rc} {time.time() - t0:.0f}s log={log}", flush=True)
        if rc != 0:
            print("command failed; fix and re-run with --start", i, flush=True)
            sys.exit(rc)


if __name__ == "__main__":
    main()
