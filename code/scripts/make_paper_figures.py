"""Paper data figures from results/release/s18_v1 (reports + derived.json).

  make_paper_figures.py      -> figures/src/fig4.svg ... fig7.svg

Fig.4  lock-in rule per cell: dumbbell of natural correction C_nat and clean-zoom recovery R_clean (+ R_rand).
Fig.5  answer of the forced-answer probe along the trajectory: clean first-look errors vs. adversarial flips.
Fig.6  (a) 3x2 channel replay per model; (b) EIR recovery of white-box flips.
Fig.7  recovery law: (a) first-look confidence and (b) clean-zoom recovery across PGD steps; (c) recovery against
       logit(c) with the fitted law and the natural first-look errors; (d) ASR against the zoom-call cap.
Every plotted number is read from the release files; nothing is typed by hand.
"""
import json, math, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = os.environ.get("TPA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REL = f"{R}/results/release/s18_v1"
OUT = f"{R}/figures/src"
mpl.rcParams.update({"svg.fonttype": "none", "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 300, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.labelsize": 7})
ORDER = ["deepeyes_7b", "minio3_7b", "pixelreasoner_7b", "qwen3vl_4b_thinking", "qwen3vl_8b_thinking",
         "glm41v_9b_thinking"]
NAME = {"deepeyes_7b": "DeepEyes", "minio3_7b": "Mini-o3", "pixelreasoner_7b": "Pixel Reasoner",
        "qwen3vl_4b_thinking": "Qwen3-VL-4B-Think", "qwen3vl_8b_thinking": "Qwen3-VL-8B-Think",
        "glm41v_9b_thinking": "GLM-4.1V-9B-Think"}
SHORT = {"deepeyes_7b": "DeepEyes", "minio3_7b": "Mini-o3", "pixelreasoner_7b": "Pixel R.",
         "qwen3vl_4b_thinking": "Qwen3-VL-4B", "qwen3vl_8b_thinking": "Qwen3-VL-8B", "glm41v_9b_thinking": "GLM-4.1V"}
MCOL = {"deepeyes_7b": "#0072B2", "minio3_7b": "#E69F00", "qwen3vl_4b_thinking": "#009E73"}
MK = {"deepeyes_7b": "o", "minio3_7b": "s", "qwen3vl_4b_thinking": "^"}
ADV, CLEAN, RAND, HOLD, REPLAY = "#D55E00", "#1F5FA0", "#8C8C8C", "#FBE3E0", "#7F3C8D"


def rep(n):
    return json.load(open(f"{REL}/reports/{n}_report.json"))


def derived():
    return json.load(open(f"{REL}/derived.json"))


def frac(rate, n):
    """A proportion from its count (the reports round rates to 3 decimals)."""
    return round(rate * n) / n


def boxed(ax):
    for s in ax.spines.values():
        s.set_visible(True)


def save(fig, name):
    p = f"{OUT}/{name}.svg"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", p)


def fig4():
    e1 = rep("e1")
    cols = [("White-box, $\\epsilon=8/255$", lambda c: c["role"] == "decide" and c["eps"] == 8),
            ("White-box, $\\epsilon=4/255$", lambda c: c["role"] == "decide" and c["eps"] == 4),
            ("Transfer from DeepEyes, $\\epsilon=8/255$", lambda c: c["role"] == "describe")]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 1.95), sharey=True, gridspec_kw=dict(wspace=0.06))
    ys = {m: len(ORDER) - 1 - i for i, m in enumerate(ORDER)}
    for ax, (title, sel) in zip(axs, cols):
        for c in e1["cells"]:
            if not sel(c):
                continue
            y = ys[c["model"]]
            rc = 100 * frac(c["R_clean"]["rate"], c["R_clean"]["n"])
            rr = 100 * frac(c["R_rand"]["rate"], c["R_rand"]["n"])
            cn = 100 * frac(c["C_nat"]["rate"], c["C_nat"]["n"])
            if c.get("holds") and c["role"] == "decide":
                ax.add_patch(Rectangle((-6, y - 0.45), 96, 0.9, color=HOLD, lw=0, zorder=0))
            ax.plot([rc, cn], [y, y], color="#9A9A9A", lw=1.0, zorder=1)
            ax.plot(rr, y, marker="|", ms=7, mew=1.2, color=RAND, ls="none", zorder=2)
            ax.plot(cn, y, "o", ms=4.2, color=CLEAN, mec="white", mew=0.4, zorder=3)
            ax.plot(rc, y, "o", ms=4.2, color=ADV, mec="white", mew=0.4, zorder=3)
            if rc <= cn:
                ax.text((min(rc, rr) if abs(rr - rc) < 6 else rc) - 2.5, y, f"{rc:.0f}", ha="right", va="center", fontsize=6, color=ADV)
                ax.text(cn + 2.5, y, f"{cn:.0f}", ha="left", va="center", fontsize=6, color=CLEAN)
            else:
                ax.text((min(cn, rr) if abs(rr - cn) < 6 else cn) - 2.5, y, f"{cn:.0f}", ha="right", va="center", fontsize=6, color=CLEAN)
                ax.text(rc + 2.5, y, f"{rc:.0f}", ha="left", va="center", fontsize=6, color=ADV)
        if title.startswith("Transfer"):
            ax.text(40, ys["deepeyes_7b"], "(source model)", ha="center", va="center", fontsize=6, color="#777777",
                    style="italic")
        ax.set_xlim(-6, 84); ax.set_ylim(-0.6, len(ORDER) - 0.4)
        ax.set_xticks([0, 20, 40, 60, 80]); ax.set_title(title, fontsize=7, pad=3)
        ax.grid(axis="y", visible=False); boxed(ax)
    axs[0].set_yticks([ys[m] for m in ORDER]); axs[0].set_yticklabels([NAME[m] for m in ORDER])
    axs[1].set_xlabel("flipped and zoomed questions answered correctly (%)")
    h = [plt.Line2D([], [], marker="o", color=CLEAN, ls="none", ms=4.2,
                    label="natural correction on clean images"),
         plt.Line2D([], [], marker="o", color=ADV, ls="none", ms=4.2,
                    label="clean views at the model's own boxes"),
         plt.Line2D([], [], marker="|", color=RAND, ls="none", ms=7, mew=1.2,
                    label="clean views at random boxes"),
         Rectangle((0, 0), 1, 1, color=HOLD, label="cell meets the lock-in rule")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 1.1), handlelength=1.2,
               columnspacing=1.2)
    save(fig, "fig4")


def fig5():
    dyn = derived()["dynamics"]
    e1 = {c["model"]: c for c in rep("e1")["cells"] if c["role"] == "decide" and c["eps"] == 8}
    fig, axs = plt.subplots(1, 6, figsize=(7.1, 1.5), sharey=True, gridspec_kw=dict(wspace=0.08))
    xs = [0, 1, 2, 3]
    keys = ("turn0", "view1", "last", "final")
    for ax, m in zip(axs, ORDER):
        d = dyn[m]
        ax.plot(xs, [100 * d["clean"][k]["correct"] for k in keys], "-o", color=CLEAN, ms=3.2, mec="white", mew=0.4)
        ax.plot(xs, [100 * d["adv"][k]["first"] for k in keys], "--", color=ADV, lw=0.9)
        ax.plot(xs, [100 * d["adv"][k]["correct"] for k in keys], "-o", color=ADV, ms=3.2, mec="white", mew=0.4)
        rc = 100 * frac(e1[m]["R_clean"]["rate"], e1[m]["R_clean"]["n"])
        ax.plot(3, rc, marker="*", ms=6.5, color=REPLAY, ls="none", zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels(["first\nlook", "view\n1", "last\nview", "final"], fontsize=6)
        ax.set_ylim(-5, 105); ax.set_xlim(-0.35, 3.35)
        ax.set_title(f"{NAME[m]}\n($n$ = {d['clean']['n']} / {d['adv']['n']})", fontsize=6.5, pad=2)
        boxed(ax)
    axs[0].set_ylabel("share of questions (%)")
    h = [plt.Line2D([], [], color=CLEAN, marker="o", ms=3.2, label="clean first-look errors: answer correct"),
         plt.Line2D([], [], color=ADV, marker="o", ms=3.2, label="adversarial flips: answer correct"),
         plt.Line2D([], [], color=ADV, ls="--", label="adversarial flips: answer = first-look answer"),
         plt.Line2D([], [], color=REPLAY, marker="*", ls="none", ms=6.5,
                    label="adversarial flips with clean views: final correct")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.3), handlelength=1.6,
               columnspacing=1.0)
    save(fig, "fig5")


def fig6():
    e2 = rep("e2"); e4r = rep("e4r")
    fig = plt.figure(figsize=(7.1, 2.3))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 2.0], wspace=0.42, hspace=0.75)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("w2b", ["#FFFFFF", "#9CC3E4", "#1F5FA0"])
    rows = [("adv", "$\\tilde{x}$"), ("gray", "gray"), ("clean", "$x$")]
    for i, m in enumerate(ORDER):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        a = e2["models"][m]; n = a["n"]
        M = [[frac(a["A"][f"{g}|{t}"], n) for t in ("own", "neutral")] for g, _ in rows]
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for r in range(3):
            for c in range(2):
                v = M[r][c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                        color="white" if v > 0.62 else "#222222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["own", "neutral"], fontsize=6)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels([lab for _, lab in rows], fontsize=6.3)
        ax.tick_params(length=0); ax.grid(False)
        mech = (a["mechanism"] + (" (visual)" if a["mechanism"] == "visual" else "")) if a["sufficiency"] else "not judged"
        mech = {"visual (visual)": "visual dominant", "redundant": "redundant"}.get(mech, mech)
        ax.set_title(f"{NAME[m]} ($n$={n})\n{mech}", fontsize=6.3, pad=2)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    fig.text(0.31, -0.03, "(a) Channel replay: probe accuracy $A(G,T)$ with clean zoomed views", ha="center",
             fontsize=7)
    ax = fig.add_subplot(gs[:, 3])
    w = 0.36
    for j, m in enumerate(ORDER):
        for k, (e, col, hatch) in enumerate(((8, ADV, None), (4, "#F2A97F", "////"))):
            c = e4r["cells"][f"{m}__single_view@{m}__eps{e}__seed0"]
            v = 100 * frac(c["Rec"], c["n_flipped"])
            lo, hi = 100 * c["Rec_ci95"][0], 100 * c["Rec_ci95"][1]
            x = j + (k - 0.5) * w
            ax.bar(x, v, w, color=col, edgecolor="#333333", lw=0.4, hatch=hatch)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="#333333", lw=0.6, capsize=1.2)
            ax.text(x, hi + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.axhline(30, color="#555555", ls=":", lw=0.8)
    ax.set_xticks(range(6)); ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=6.3, rotation=25, ha="right")
    ax.set_ylim(0, 88); ax.set_xlim(-0.6, 5.6); ax.set_ylabel("white-box flips restored by EIR (%)"); boxed(ax)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=ADV, ec="#333333", lw=0.4, label="$\\epsilon=8/255$"),
                       Rectangle((0, 0), 1, 1, fc="#F2A97F", ec="#333333", lw=0.4, hatch="////",
                                 label="$\\epsilon=4/255$")],
              loc="upper right", fontsize=6.3, ncol=2, handlelength=1.4)
    ax.set_title("(b) Evidence-isolated re-answering", y=-0.36, fontsize=7)
    save(fig, "fig6")


def fig7():
    e3 = rep("e3"); e6 = rep("e6"); d = derived(); law = d["law"]
    ms = ["deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking"]
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.6), gridspec_kw=dict(wspace=0.45))
    steps = [2, 5, 10, 25, 50]
    for m in ms:
        pl = [e3["per_level"][f"{m}|s{s}"] for s in steps]
        kw = dict(color=MCOL[m], marker=MK[m], ms=3.3, mec="white", mew=0.4, label=NAME[m])
        axs[0].plot(range(5), [p["c_mean"] for p in pl], **kw)
        axs[1].plot(range(5), [100 * frac(p["R_clean"], p["n"]) for p in pl], **kw)
        tr = e3["per_level"].get(f"{m}|transfer")
        if tr:
            axs[0].plot(5.2, tr["c_mean"], marker=MK[m], ms=3.8, color=MCOL[m], mfc="white", mew=0.9, ls="none")
            axs[1].plot(5.2, 100 * frac(tr["R_clean"], tr["n"]), marker=MK[m], ms=3.8, color=MCOL[m], mfc="white",
                        mew=0.9, ls="none")
    for ax in axs[:2]:
        ax.set_xticks([0, 1, 2, 3, 4, 5.2]); ax.set_xticklabels(["2", "5", "10", "25", "50", "tr."])
        ax.set_xlabel("PGD step (tr.: transfer)"); ax.axvline(4.6, color="#BBBBBB", lw=0.6, ls=":")
    axs[0].set_ylabel("mean first-look confidence $\\bar c$"); axs[0].set_ylim(0.55, 1.02)
    axs[1].set_ylabel("clean-zoom recovery (%)"); axs[1].set_ylim(-3, 62)
    axs[0].set_title("(a) Confidence across steps", y=-0.5, fontsize=7)
    axs[1].set_title("(b) Recovery across steps", y=-0.5, fontsize=7)
    ax = axs[2]
    b = law["beta"]
    xx = np.linspace(-1, 9.4, 100)
    for m in ms:
        g = law["gamma"][m]
        ax.plot(xx, [100 / (1 + math.exp(-(b * x + g))) for x in xx], color=MCOL[m], lw=0.9)
        for r in law["bins"][m]:
            ax.plot(r["x_mean"], 100 * r["obs"], marker=MK[m], ms=3.3, color=MCOL[m], mec="white", mew=0.4, ls="none")
        ce = law["clean_errors"][m]
        x0 = math.log(ce["c_mean"] / (1 - ce["c_mean"]))
        ax.plot(x0, 100 * ce["observed"], marker=MK[m], ms=4.2, color=MCOL[m], mfc="white", mew=0.9, ls="none")
    ax.set_xlabel("first-look margin logit$(c)$"); ax.set_ylabel("recovery (%)"); ax.set_ylim(-3, 75)
    ax.set_xlim(-1, 9.6)
    ax.text(9.4, 72, f"fit: $\\beta={b:.2f}$\nhollow: natural errors", ha="right", va="top", fontsize=6)
    ax.set_title("(c) Recovery law", y=-0.5, fontsize=7)
    ax = axs[3]
    caps = ["0", "1", "2", "4"]
    for key, m, ls, lab in (("deepeyes_7b|single_view@deepeyes_7b__eps8__seed0", "deepeyes_7b", "-", "DeepEyes, WB"),
                            ("minio3_7b|single_view@minio3_7b__eps8__seed0", "minio3_7b", "-", "Mini-o3, WB"),
                            ("minio3_7b|single_view@deepeyes_7b__eps8__seed0", "minio3_7b", "--", "Mini-o3, transfer")):
        lv = e6[key]["levels"]
        ax.plot(range(4), [100 * frac(lv[c]["ASR"]["rate"], lv[c]["ASR"]["n"]) for c in caps], ls=ls, marker=MK[m],
                color=MCOL[m], ms=3.3, mec="white", mew=0.4, label=lab)
    lv = e6["minio3_7b|single_view@minio3_7b__eps8__seed0"]["levels"]
    ax.plot(range(4), [100 * frac(lv[c]["clean_acc"], 150) for c in caps], ls=":", color="#555555", marker="x", ms=3,
            label="Mini-o3, clean acc.")
    ax.set_xticks(range(4)); ax.set_xticklabels(caps); ax.set_xlabel("zoom-call cap"); ax.set_ylabel("ASR / accuracy (%)")
    ax.set_ylim(-3, 105); ax.legend(fontsize=6.2, loc="center right", bbox_to_anchor=(1.0, 0.42), handlelength=1.8)
    ax.set_title("(d) Zoom budget, $\\epsilon=8/255$", y=-0.5, fontsize=7)
    for ax in axs:
        boxed(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.1), handlelength=1.6)
    save(fig, "fig7")


if __name__ == "__main__":
    fig4(); fig5(); fig6(); fig7()
