"""S9: release/<id> -> figure data (figures/data/*.csv) and data figures (figures/src/fig*.svg), matplotlib with
svg.fonttype=none (text stays text).  Style follows the results reference: one legend
on top, light grid, 0.6 pt axes, values printed next to points, our method shaded; fixed semantic colours
(figures/palette.md): attack success = #E04B3A, clean / robustness = #2F4B7C, baselines in neutral greys,
adaptive = #E04B3A, non-adaptive = #4472C4.

  make_figures.py --release results/release/v1 --data figures/data --src figures/src
"""
from __future__ import annotations

import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import font_manager
for _f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):     # WSL has no Arial; use the Windows font files
    if os.path.exists(f"/mnt/c/Windows/Fonts/{_f}"):
        font_manager.fontManager.addfont(f"/mnt/c/Windows/Fonts/{_f}")
mpl.rcParams.update({"svg.fonttype": "none", "font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "figure.dpi": 300, "axes.grid": True, "grid.color": "#E3E3E3",
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False})
RED, NAVY, BLUE, ORANGE, GREY = "#E04B3A", "#2F4B7C", "#4472C4", "#ED7D31", "#8C8C8C"
SHADE = "#FBE3E0"
TW, CW = 7.167, 3.5          # \textwidth, \columnwidth in inches (IEEEtran compsoc, measured in S2)
MODEL = {"deepeyes_7b": "DeepEyes-7B", "minio3_7b": "Mini-o3-7B", "qwen3vl_4b_thinking": "Qwen3-VL-4B-Think",
         "vision_r1_7b": "Vision-R1-7B", "vlaa_thinker_7b": "VLAA-Thinker-7B"}
METHOD = {"single_view": "Single-view", "eot": "EOT crop", "mca_style": "MCA-style", "reasonbreak": "ReasonBreak",
          "tpa": "TPA (ours)"}


def save(fig, src, name):
    p = f"{src}/{name}.svg"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    s = open(p).read()
    i = s.index(">", s.index("<svg")) + 1
    open(p, "w").write(s[:i] + f"\n <title>{name}</title>" + s[i:])


def fig4_flip(c, data, src):
    """Where flips happen: ASR split into perception-level (probe after o_0 already flipped) and reasoning-level
    (RFR), plus SRR, for each tool model and method (E4, V* + HR-Bench pooled by mean)."""
    d = c[(c.exp == "E4") & (c.eps == 8) & c.split.isin(["vstar_eval", "hrbench4k_eval"]) &
          c.eval_model.isin(["deepeyes_7b", "qwen3vl_4b_thinking"]) & c.method.isin(["single_view", "eot", "tpa"])]
    g = d.groupby(["eval_model", "method"])[["ASR", "RFR", "SRR", "FSP"]].mean().reset_index()
    g["perception"] = g["ASR"] - g["RFR"]
    g.to_csv(f"{data}/fig4_flip.csv", index=False)
    models = ["deepeyes_7b", "qwen3vl_4b_thinking"]
    meths = ["single_view", "eot", "tpa"]
    fig, axes = plt.subplots(1, 2, figsize=(TW, 1.9), sharey=True)
    for ax, m in zip(axes, models):
        x = np.arange(len(meths))
        rows = [g[(g.eval_model == m) & (g.method == k)] for k in meths]
        per = [100 * r.perception.iloc[0] if len(r) else 0 for r in rows]
        rea = [100 * r.RFR.iloc[0] if len(r) else 0 for r in rows]
        srr = [100 * r.SRR.iloc[0] if len(r) else np.nan for r in rows]
        ax.axvspan(len(meths) - 1.5, len(meths) - 0.5, color=SHADE, zorder=0)
        ax.bar(x - 0.17, per, 0.32, color=RED, label="ASR: flipped at the original view")
        ax.bar(x - 0.17, rea, 0.32, bottom=per, color="#F4A79D", label="ASR: flipped during reasoning (RFR)")
        ax.bar(x + 0.17, srr, 0.32, color=NAVY, label="Self-correction rate (SRR)")
        for xi, (p_, r_, s_) in enumerate(zip(per, rea, srr)):
            ax.text(xi - 0.17, p_ + r_ + 1.5, f"{p_ + r_:.1f}", ha="center", fontsize=6.5)
            if not np.isnan(s_):
                ax.text(xi + 0.17, s_ + 1.5, f"{s_:.1f}", ha="center", fontsize=6.5)
        ax.set_xticks(x, [METHOD[k] for k in meths])
        ax.set_xlabel(f"({'ab'[models.index(m)]}) {MODEL[m]}")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, src, "fig4_flip")


def fig5_h3(release, h, data, src):
    p = f"{release}/fig5_h3_items.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    g = d.groupby(["RG", "VEG"]).success.agg(["mean", "count"]).reset_index()
    g.to_csv(f"{data}/fig5_h3.csv", index=False)
    fig, ax = plt.subplots(figsize=(CW, 2.0))
    for veg, col, lab in [(0, GREY, "re-observation not used (VEG = 0)"), (1, RED, "re-observation used (VEG = 1)")]:
        s = g[g.VEG == veg]
        ax.plot(s.RG, 100 * s["mean"], "o-", color=col, label=lab, ms=3)
        for _, r in s.iterrows():
            ax.text(r.RG, 100 * r["mean"] + 2.5, f"{100 * r['mean']:.0f}", ha="center", fontsize=6.5, color=col)
    ax.set_xlabel("Re-observations per trajectory (RG)")
    ax.set_ylabel("Transfer success (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.32))
    save(fig, src, "fig5_regrounding")


def fig6_budget_forcing(h, data, src):
    rows = h["H3"].get("intervention", []) if isinstance(h.get("H3"), dict) else []
    if not rows:
        return
    d = pd.DataFrame(rows)
    d["reasoning"] = d.model.str.split("@").str[0]
    # pool V* and HR-Bench (n-weighted) -> one row per controlled pair
    d = (d.assign(w=d.n).groupby(["reasoning", "base"])
         .apply(lambda g: pd.Series({k: np.average(g[k], weights=g.w) for k in
                                     ["asr_default", "asr_forced", "rg_default", "rg_forced"]} | {"n": g.n.sum()}))
         .reset_index())
    base_short = {"qwen25vl_7b": "Qwen2.5-VL", "qwen3vl_4b_instruct": "Qwen3-VL-Inst", "deepeyes_7b_tool_off": "tool off"}
    d["label"] = d.reasoning.map(MODEL).fillna(d.reasoning) + " vs. " + d.base.map(base_short).fillna(d.base)
    d.to_csv(f"{data}/fig6_budget_forcing.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(TW, 0.24 * len(d) + 0.7), sharey=True)
    y = np.arange(len(d))
    for ax, (a, b, lab, xl) in zip(axes, [("asr_default", "asr_forced", "ASR", "Transfer ASR (%)"),
                                          ("rg_default", "rg_forced", "RG", "Re-observations per trajectory")]):
        sc = 100 if lab == "ASR" else 1
        fmt = (lambda v: f"{v:.1f}") if lab == "ASR" else (lambda v: f"{v:.2f}")
        for yi, (_, r) in enumerate(d.iterrows()):
            va, vb = sc * r[a], sc * r[b]
            ax.plot([va, vb], [yi, yi], color="#BBBBBB", lw=1, zorder=1)
            lo_first = va <= vb
            for v, col, left in [(va, NAVY, lo_first), (vb, RED, not lo_first)]:
                ax.annotate(fmt(v), (v, yi), xytext=(-5 if left else 5, 0), textcoords="offset points",
                            ha="right" if left else "left", va="center", fontsize=6.5, color=col)
        ax.scatter(sc * d[a], y, color=NAVY, s=12, zorder=2, label="default budget")
        ax.scatter(sc * d[b], y, color=RED, s=12, zorder=2, label="budget forced (no tool call)")
        lo = min(sc * d[a].min(), sc * d[b].min()); hi = max(sc * d[a].max(), sc * d[b].max())
        pad = 0.18 * (hi - lo if hi > lo else 1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(xl)
    axes[0].set_yticks(y, d.label)
    axes[0].set_ylim(len(d) - 0.5, -0.5)
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=2, bbox_to_anchor=(0.5, axes[0].get_position().y1 + 0.01))
    save(fig, src, "fig6_budget_forcing")


def fig7_budget(h, data, src):
    per = h["H4"].get("per_model", []) if isinstance(h.get("H4"), dict) else []
    if not per:
        return
    rows = []
    for r in per:
        for kind in ("nonadaptive", "adaptive"):
            for p in r.get(f"curve_{kind}", []):
                rows.append(dict(model=r["model"], kind=kind, **p))
    d = pd.DataFrame(rows)
    d.to_csv(f"{data}/fig7_budget.csv", index=False)
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(TW, 1.9))
    axes = np.atleast_1d(axes)
    for i, (ax, m) in enumerate(zip(axes, models)):
        for kind, col, lab in [("nonadaptive", BLUE, "non-adaptive (single-view PGD)"), ("adaptive", RED, "adaptive (TPA at each budget)")]:
            s = d[(d.model == m) & (d.kind == kind)].sort_values("budget")
            if len(s):
                ax.plot(s.budget, 100 * s.robustness, "o-", color=col, ms=3, label=lab)
                for _, r in s.iterrows():
                    up = kind == "nonadaptive"          # non-adaptive labels above, adaptive below the point
                    ax.annotate(f"{100 * r.robustness:.0f}", (r.budget, 100 * r.robustness), xytext=(0, 4 if up else -4),
                                textcoords="offset points", ha="center", va="bottom" if up else "top",
                                fontsize=6.5, color=col)
        ax.set_xscale("log")
        b = sorted(d[d.model == m].budget.unique())
        ax.set_xticks(b, [str(int(v)) for v in b]); ax.minorticks_off()
        ax.set_ylim(0, 105)
        unit = "tool calls + 1" if m == "deepeyes_7b" else "thinking tokens"
        ax.set_xlabel(f"Budget ({unit})\n({'abc'[i]}) {MODEL.get(m, m)}")
    axes[0].set_ylabel("Robust accuracy on attacked items (%)")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    save(fig, src, "fig7_budget")


def fig8_eps(c, data, src):
    d = c[(c.exp.isin(["E8"])) & c.method.isin(["tpa", "single_view", "eot"])]
    g = d.groupby(["method", "eps"]).agg(ASR=("ASR", "mean"), lo=("ASR_lo", "mean"), hi=("ASR_hi", "mean")).reset_index()
    if g.empty:
        return
    g.to_csv(f"{data}/fig8_eps.csv", index=False)
    fig, ax = plt.subplots(figsize=(CW, 2.0))
    for meth, col in [("single_view", GREY), ("eot", "#555555"), ("tpa", RED)]:
        s = g[g.method == meth].sort_values("eps")
        ax.plot(s.eps, 100 * s.ASR, "o-", color=col, ms=3, label=METHOD[meth])
        ax.fill_between(s.eps, 100 * s.lo, 100 * s.hi, color=col, alpha=0.15, lw=0)
    ax.axvline(8, color="#555555", ls="--", lw=0.6)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"]); ax.minorticks_off()
    ax.set_xlabel(r"$\epsilon$ (/255)"); ax.set_ylabel("ASR (%)"); ax.set_ylim(0, 105)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2))
    save(fig, src, "fig8_eps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--data", required=True); ap.add_argument("--src", required=True)
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True); os.makedirs(a.src, exist_ok=True)
    c = pd.read_csv(f"{a.release}/cells.csv")
    h = json.load(open(f"{a.release}/hypotheses.json"))
    fig4_flip(c, a.data, a.src); fig5_h3(a.release, h, a.data, a.src); fig6_budget_forcing(h, a.data, a.src)
    fig7_budget(h, a.data, a.src); fig8_eps(c, a.data, a.src)
    print("figures:", sorted(os.listdir(a.src)))


if __name__ == "__main__":
    main()
