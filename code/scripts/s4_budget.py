"""S4: GPU-hour budget of the frozen protocol v1 matrix (extends s3_cost.py 'recommended' with E1 and the
eps=4/255 H2b arm, and the S4 size choices).  Per-call costs come from results/S3/cost_estimate_recommended.json
(measured for DeepEyes / Qwen3-VL-Thinking; ASSUMED for Mini-o3 and transfer models, re-measured in S6).
Output: results/S4/budget_v1.json + printed table."""
import json, os

ROOT = "."
c = json.load(open(f"{ROOT}/results/S3/cost_estimate_recommended.json"))["cost_per_call_s"]
import os, sys
S6 = f"{ROOT}/results/S6/cost_measured.json"
USE_S6 = "--s6" in sys.argv and os.path.exists(S6)
if USE_S6:   # S6: replace the ASSUMED Mini-o3 / transfer costs by measurements
    m6 = json.load(open(S6))
    c["minio3"] = {k: m6["minio3"][k] for k in ("roll", "grad")}
    c["transfer"] = {"roll": m6["transfer"]["roll"], "grad": None}
T = 50
MK = {"deepeyes": (2, 10), "minio3": (2, 10), "qwen3_think": (1, 25)}
SV = 0.6                     # single-view grad cost fraction (ASSUMED, as in S3)
SAMP = 0.1                   # n=3 sampling subset fraction
EV = 1 + SAMP * 3
MCA_S = 60.0                 # MCA-style CLIP-ensemble attack, s/img (ASSUMED)
JUDGE_S = 3.0                # open judge call, s (ASSUMED)
CAPTION_S = {"deepeyes": c["deepeyes"]["roll"], "minio3": c["minio3"]["roll"], "qwen3_think": c["qwen3_think"]["roll"],
             "vision_r1": 15.0, "vlaa": 15.0, "qwen25vl": 15.0}

N_VSTAR, N_HR4K, N_MME = 191, 200, 150
N_MAIN = N_VSTAR + N_HR4K
N_E4 = N_MAIN + N_MME
N_E1, N_EPS4 = 100, 130
E6_N, E6_B = 100, 3
E7_N = {"mme_rs": 100, "xlrs": 100, "vrsbench": 100}
E8_N, E8_V = 50, 7
E5_M = 6
WB = ["deepeyes", "qwen3_think"]
TOOL = ["deepeyes", "minio3", "qwen3_think"]
TR = 2  # Vision-R1, VLAA-Thinker


def roll(m): return c[m]["roll"] if m in c and c[m]["roll"] else 15.0
def grad(m): return c[m]["grad"]
def tpa(m): M, K = MK[m]; return T * M * grad(m) + (T // K) * M * roll(m)
def sv(m): return T * SV * grad(m)
def eot(m): return T * MK[m][0] * grad(m)
def ev(m, k=1): return k * EV * roll(m)


rows = {}
rows["clean eval (tool on/off, all E4 images)"] = sum(N_E4 * ev(m, 2) for m in TOOL) + N_E4 * TR * ev("transfer")
rows["E3 single-view PGD, tool on/off"] = sum(N_MAIN * (sv(m) + ev(m, 2)) for m in TOOL)
e4 = 0
for m in WB:
    n = N_E4 if m == "deepeyes" else N_MAIN          # MME-RealWorld-Lite subset: DeepEyes white-box only
    e4 += n * (tpa(m) + eot(m) + MCA_S / 2) + (n - N_MAIN) * sv(m)
    # tool on/off only for TPA and single-view (H2b gap); EOT, MCA-style, ReasonBreak tool-on only
    e4 += n * (2 * ev(m, 2) + 3 * ev(m))
    # transfer eval (Mini-o3 tool-on + 2 reasoning models) on V* only, 5 methods
    e4 += N_VSTAR * 5 * (ev("minio3") + TR * ev("transfer"))
rows["E4 TPA main + transfer (eps 8)"] = e4
# H2b eps=4 arm: V* subset, TPA vs single-view vs EOT, tool-on eval only
rows["E4b H2b arm eps 4 (V*)"] = sum(N_EPS4 * (tpa(m) + sv(m) + eot(m) + 3 * ev(m)) for m in WB)
rows["E1 caption+judge protocol"] = N_E1 * 5 * sum(CAPTION_S.values()) + N_E1 * 5 * 6 * JUDGE_S
rows["E2 MathVista-MC (Qwen3-Think, 150)"] = 150 * (tpa("qwen3_think") + sv("qwen3_think") + 3 * ev("qwen3_think"))
rows["E5 probes + interventions"] = E5_M * N_MAIN * (grad("deepeyes") + 2 * 15.0)
rows["E6 budget curves"] = sum(E6_N * E6_B * (2 * ev(m) + 0.5 * tpa(m)) for m in WB) + E6_N * E6_B * 2 * ev("transfer")  # + Vision-R1 non-adaptive
# E7: white-box DeepEyes (tool model; zoom matters most on very large RS images), transfer eval on Qwen3-Think tool mode
rows["E7 generalisation (RS x3)"] = sum(n * (tpa("deepeyes") + sv("deepeyes") + 2 * ev("deepeyes") + 2 * ev("qwen3_think"))
                                        for n in E7_N.values())
rows["E8 ablations + eps curve"] = E8_V * E8_N * (tpa("qwen3_think") + ev("qwen3_think"))
rows["S11 pilot"] = sum(100 * (tpa(m) + sv(m) + 4 * ev(m)) for m in WB)
tot = sum(rows.values())
os.makedirs(f"{ROOT}/results/S4", exist_ok=True)
out = {"rows_gpu_h": {k: round(v / 3600, 1) for k, v in rows.items()}, "total_gpu_h": round(tot / 3600, 1),
       "sizes": dict(N_VSTAR=N_VSTAR, N_HR4K=N_HR4K, N_MME=N_MME, N_E1=N_E1, N_EPS4=N_EPS4, E6_N=E6_N, E6_B=E6_B,
                     E7_N=E7_N, E8_N=E8_N, E8_V=E8_V, E5_M=E5_M, T=T, MK=MK)}
json.dump(out, open(f"{ROOT}/results/S6/budget_v2.json" if USE_S6 else f"{ROOT}/results/S4/budget_v1.json", "w"), indent=1)
for k, v in rows.items():
    print(f"{k:42s} {v / 3600:7.1f}")
print(f"{'TOTAL':42s} {tot / 3600:7.1f} GPU-h")
