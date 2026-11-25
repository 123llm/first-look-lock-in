"""S3 item 7: GPU-hour extrapolation for the experiment matrix from measured per-call costs.
Measured: results/S3/model_check_{deepeyes,qwen3vl4b_thinking}.json.  Unmeasured costs are marked ASSUMED and are
re-measured in S6 smoke tests.  Output: results/S3/cost_estimate.json (+ printed table)."""
import json

R = "./results/S3"
de = json.load(open(f"{R}/model_check_deepeyes.json"))
q3 = json.load(open(f"{R}/model_check_qwen3vl4b_thinking.json"))

# seconds per call --------------------------------------------------------------------------------------
cost = {
    # white-box victims (measured)
    "deepeyes": {"roll": de["mean_t_rollout"], "grad": de["mean_t_grad"], "src": "measured"},
    # Qwen3-VL-Thinking measured with max_new_tokens 2048 (2/12 hit the cap); 4096 cap -> x1.25 ASSUMED
    "qwen3_think": {"roll": q3["mean_t_rollout"] * 1.25, "grad": q3["mean_t_grad"], "src": "measured x1.25 (4096 cap)"},
    # Mini-o3: multi-turn (more crops) -> ASSUMED 2x DeepEyes for both
    "minio3": {"roll": 2 * de["mean_t_rollout"], "grad": 2 * de["mean_t_grad"], "src": "ASSUMED 2x DeepEyes"},
    # transfer-only reasoning victims (Vision-R1, VLAA-Thinker, Qwen2.5-VL, Qwen3-VL-Instruct): ASSUMED 15 s/rollout
    "transfer": {"roll": 15.0, "grad": None, "src": "ASSUMED 15 s/rollout"},
}

# attack hyper-parameters (frozen in S4; these are the S3 planning values) -----------------------------
import sys
PLAN = sys.argv[1] if len(sys.argv) > 1 else "full"
T = 50                                                  # PGD steps
MK = {"full": {"deepeyes": (2, 10), "minio3": (2, 10), "qwen3_think": (2, 10)},      # (trajectories/step, resample K)
      "recommended": {"deepeyes": (2, 10), "minio3": (2, 10), "qwen3_think": (1, 25)}}[PLAN]
SIZES = {"full": dict(N_MAIN=191 + 300, N_E4=191 + 300 + 300, TR=4, E6_N=191, E6_B=4, E7_N=300, E8_N=100, E8_V=10,
                      SAMP=0.2, E5_M=6),
         "recommended": dict(N_MAIN=191 + 200, N_E4=191 + 200 + 150, TR=2, E6_N=100, E6_B=3, E7_N=150, E8_N=60,
                             E8_V=10, SAMP=0.1, E5_M=6)}[PLAN]
SV_GRAD_FRAC = 0.6           # single-view PGD: original view only, shorter context -> ASSUMED 0.6 x grad cost
EVAL_ROLL = 1 + SIZES["SAMP"] * 3   # greedy on all + n=3 samples on a subset


def tpa(m):      # one image, white-box TPA
    c = cost[m]; M, K = MK[m]
    return T * M * c["grad"] + (T // K) * M * c["roll"]


def single_view(m):
    return T * SV_GRAD_FRAC * cost[m]["grad"]


def eot(m):      # EOT random crops: same #grad as TPA, no resampling rollouts
    return T * MK[m][0] * cost[m]["grad"]


def ev(m, conds=1):
    return conds * EVAL_ROLL * cost[m]["roll"]


N_MAIN, N_E4 = SIZES["N_MAIN"], SIZES["N_E4"]
WB = ["qwen3_think", "deepeyes"]
TOOL = ["deepeyes", "minio3", "qwen3_think"]
TR_MODELS = SIZES["TR"]

rows = []
# clean evaluation (shared by all experiments): all white-box + tool models, tool on/off
rows.append(("clean eval (tool on/off)", sum(N_E4 * ev(m, 2) for m in TOOL) + N_E4 * TR_MODELS * ev("transfer")))
# E3: single-view PGD on 3 tool models, eval tool on/off
rows.append(("E3 single-view PGD, tool on/off", sum(N_MAIN * (single_view(m) + ev(m, 2)) for m in TOOL)))
# E4: TPA + EOT on 2 white-box models (single-view reused from E3 on N_MAIN, extra 300 MME imgs), eval on
#     self + transfer to Mini-o3 and 4 transfer models; MCA-style CLIP proxy attack ~ 60 s/img ASSUMED
e4 = 0
for m in WB:
    e4 += N_E4 * (tpa(m) + eot(m)) + (N_E4 - N_MAIN) * single_view(m) + N_E4 * 60
    e4 += N_E4 * 4 * (ev(m) + ev("minio3") + TR_MODELS * ev("transfer"))   # 4 methods' adversarial images
rows.append(("E4 TPA main + transfer", e4))
# E5: probes are cheap forwards (~grad/3), budget forcing 2 extra evals, on 6 models x N_MAIN, reuse images
rows.append(("E5 re-grounding probes", SIZES["E5_M"] * N_MAIN * (3 * cost["deepeyes"]["grad"] / 3 + 2 * 15.0)))
# E6: 4 budgets x {non-adaptive eval, adaptive re-attack} x 3 models x V*
e6 = 0
for m in ["qwen3_think", "deepeyes", "minio3"]:
    e6 += SIZES["E6_N"] * SIZES["E6_B"] * (ev(m) + tpa(m) * 0.5 + ev(m))     # adaptive attack at half T per budget
rows.append(("E6 reasoning budget", e6))
# E7: remote sensing 300 imgs x TPA + single-view on 2 white-box + defenses (CLIP-based, cheap) ~ 0.5 h
rows.append(("E7 generalisation (RS, defenses)", sum(SIZES["E7_N"] * (tpa(m) + single_view(m) + 2 * ev(m)) for m in WB) + 1800))
# E8: 6 ablation variants x 100 imgs on 4B + eps curve 4 values x 100
rows.append(("E8 ablations + eps curve", SIZES["E8_V"] * SIZES["E8_N"] * (tpa("qwen3_think") + ev("qwen3_think"))))
# S11 pilot: 100 imgs x 2 models x (TPA + single-view + evals)
rows.append(("S11 pilot", sum(100 * (tpa(m) + single_view(m) + 4 * ev(m)) for m in WB)))

total = sum(v for _, v in rows)
out = {"plan": PLAN, "cost_per_call_s": cost, "params": {"T": T, "MK": MK, **SIZES},
       "per_image_s": {m: {"tpa": tpa(m), "single_view": single_view(m), "eval_greedy": cost[m]["roll"]} for m in TOOL},
       "rows_gpu_h": {k: v / 3600 for k, v in rows}, "total_gpu_h": total / 3600}
json.dump(out, open(f"{R}/cost_estimate_{PLAN}.json", "w"), indent=1)
for m in TOOL:
    print(f"{m:12s} TPA {tpa(m)/60:6.1f} min/img   single-view {single_view(m)/60:5.1f} min/img   rollout {cost[m]['roll']:5.1f} s  [{cost[m]['src']}]")
for k, v in rows:
    print(f"{k:40s} {v/3600:7.1f} GPU-h")
print(f"{'TOTAL':40s} {total/3600:7.1f} GPU-h  (= {total/3600/24:.1f} days on one RTX 4080)")
