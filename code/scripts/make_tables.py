"""S9: release/<id>/{cells.csv, hypotheses.json, cost.csv} -> LaTeX tables (booktabs; table* bodies use
tabular*{\\textwidth} with \\extracolsep{\\fill}, never \\resizebox).  Numbers are percentages with one decimal; CIs
in brackets where the table reports them.  Output: <out>/tab*.tex (default results/tables/ or the synthetic root).

  make_tables.py --release results/release/v1 --out results/tables
"""
from __future__ import annotations

import argparse, json, os

import numpy as np
import pandas as pd

MODEL = {"deepeyes_7b": "DeepEyes-7B", "minio3_7b": "Mini-o3-7B", "qwen3vl_4b_thinking": "Qwen3-VL-4B-Thinking",
         "vision_r1_7b": "Vision-R1-7B", "vlaa_thinker_7b": "VLAA-Thinker-7B", "qwen25vl_7b": "Qwen2.5-VL-7B",
         "qwen3vl_4b_instruct": "Qwen3-VL-4B-Instruct"}
METHOD = {"single_view": "Single-view PGD", "eot": "EOT random crop", "mca_style": "MCA-style (reimpl.)",
          "reasonbreak": "ReasonBreak", "tpa": "TPA (ours)"}
DATA = {"vstar_eval": "V$^*$", "hrbench4k_eval": "HR-Bench 4K", "mme_general_eval": "MME-RW-Lite",
        "e7_mme_rs": "MME-RW RS", "e7_xlrs": "XLRS-Bench", "e7_vrsbench_yesno": "VRSBench"}
SHORT = {"minio3_7b": "Mini-o3", "vision_r1_7b": "Vision-R1", "vlaa_thinker_7b": "VLAA-Th."}
ORDER = ["single_view", "eot", "mca_style", "reasonbreak", "tpa"]


def pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def get(c, **kw):
    d = c
    for k, v in kw.items():
        d = d[d[k] == v]
    return d.iloc[0] if len(d) else None


def tab2_main(c, out):
    """White-box ASR (tool on / tool off) and invalid rate, eps 8, by dataset and method."""
    models = ["deepeyes_7b", "qwen3vl_4b_thinking"]
    off = {"deepeyes_7b": "deepeyes_7b_tool_off", "qwen3vl_4b_thinking": "qwen3vl_4b_thinking_tool_off"}
    datasets = ["vstar_eval", "hrbench4k_eval", "mme_general_eval"]
    e4 = c[(c.exp == "E4") & (c.eps == 8)]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{White-box attack success at $\epsilon=8/255$. ASR (\%) with the tool loop on and off, and the rate of "
         r"invalid outputs with the tool on. Denominator: items answered correctly on the clean image under the same setting; --: not evaluated (the tool-off arm is run for single-view PGD and TPA; MCA-style and ReasonBreak are not run on MME-RW-Lite).}",
         r"\label{tab:main}", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll" + "ccc" * len(datasets) + "}", r"\toprule",
         " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DATA[d]}}}" for d in datasets) + r" \\",
         " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(datasets))),
         "Model & Method & " + " & ".join(["On", "Off", "Inv."] * len(datasets)) + r" \\", r"\midrule"]
    for mi, m in enumerate(models):
        for k, meth in enumerate(ORDER):
            src = m if meth in ("single_view", "eot", "tpa") else ("clip3" if meth == "mca_style" else "qwen3vl_4b_instruct")
            cells = []
            for d in datasets:
                on = get(e4, split=d, eval_model=m, method=meth, source=src)
                of = get(e4, split=d, eval_model=off[m], method=meth, source=src)
                cells += [pct(on.ASR if on is not None else None), pct(of.ASR if of is not None else None),
                          pct(on.INV if on is not None else None)]
            name = MODEL[m] if k == 0 else ""
            row = f"{name} & {METHOD[meth]} & " + " & ".join(cells) + r" \\"
            L.append(row)
        if mi < len(models) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    open(f"{out}/tab2_main.tex", "w").write("\n".join(L) + "\n")


def tab3_transfer(c, out):
    e4 = c[(c.exp == "E4") & (c.split == "vstar_eval") & (c.eps == 8)]
    targets = ["minio3_7b", "vision_r1_7b", "vlaa_thinker_7b"]
    srcs = ["deepeyes_7b", "qwen3vl_4b_thinking"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Transfer ASR (\%) on V$^*$ at $\epsilon=8/255$ from the two white-box models (DeepEyes / Qwen3-Think) "
         r"to models whose gradients are not used.}", r"\label{tab:transfer}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l" + "c" * len(targets) + "}", r"\toprule",
         "Method & " + " & ".join(SHORT[t] for t in targets) + r" \\", r"\midrule"]
    for meth in ORDER:
        cells = []
        for t in targets:
            if meth in ("mca_style", "reasonbreak"):
                r = get(e4, eval_model=t, method=meth)
                cells.append(pct(r.ASR if r is not None else None))
            else:
                v = [get(e4, eval_model=t, method=meth, source=s) for s in srcs]
                cells.append("/".join(pct(x.ASR if x is not None else None) for x in v))
        L.append(f"{METHOD[meth]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    open(f"{out}/tab3_transfer.tex", "w").write("\n".join(L) + "\n")


def tab4_h1(h, out):
    H = h["H1"]
    if H.get("verdict") == "no_data":
        return
    attacks = sorted({x["attack"] for x in H["cells"]})
    models = sorted({x["model"] for x in H["cells"]})
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{The two evaluation protocols on the same 24 cells (E1). Each entry is ASR under RTAP / under "
         rf"captioning with an open judge (\%). Kendall $\tau_b={H['tau_b']:.2f}$ [{H['ci'][0]:.2f}, {H['ci'][1]:.2f}] "
         r"(image-cluster bootstrap).}", r"\label{tab:protocols}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(attacks) + "}", r"\toprule",
         "Model & " + " & ".join(METHOD.get(a.split("@")[0], a) for a in attacks) + r" \\", r"\midrule"]
    for m in models:
        cells = []
        for a in attacks:
            x = next((y for y in H["cells"] if y["model"] == m and y["attack"] == a), None)
            cells.append(f"{pct(x['asr_rtap'])} / {pct(x['asr_judge'])}" if x else "--")
        L.append(f"{MODEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    open(f"{out}/tab4_protocols.tex", "w").write("\n".join(L) + "\n")


def tab5_rs(c, out):
    e7 = c[c.exp == "E7"]
    ds = ["e7_mme_rs", "e7_xlrs", "e7_vrsbench_yesno"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Remote-sensing VQA at $\epsilon=8/255$: ASR (\%) on the white-box DeepEyes-7B and transferred to "
         r"Qwen3-VL-4B-Thinking (tool mode).}", r"\label{tab:rs}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}ll" + "c" * len(ds) + "}", r"\toprule",
         "Victim & Method & " + " & ".join({"e7_mme_rs": "MME-RS", "e7_xlrs": "XLRS", "e7_vrsbench_yesno": "VRSB."}[d]
                                           for d in ds) + r" \\", r"\midrule"]
    for m in ["deepeyes_7b", "qwen3vl_4b_thinking"]:
        for k, meth in enumerate(["single_view", "tpa"]):
            cells = [pct(getattr(get(e7, split=d, eval_model=m, method=meth), "ASR", None)) for d in ds]
            vic = {"deepeyes_7b": "DeepEyes", "qwen3vl_4b_thinking": "Qwen3-Think"}[m]
            L.append(f"{vic if k == 0 else ''} & {'Single-view' if meth == 'single_view' else 'TPA (ours)'} & "
                     + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    open(f"{out}/tab5_rs.tex", "w").write("\n".join(L) + "\n")


def tab6_ablation(c, cost, out):
    e8 = c[(c.exp == "E8") & (c.eps == 8)]
    names = {"tpa": "TPA ($K$=25, window 256)", "tpa_noprobe": "w/o re-observation probes",
             "tpa_noresample": "w/o trajectory resampling", "tpa_K10": "$K=10$", "tpa_K50": "$K=50$ (one sample)",
             "tpa_w128": "window 128", "tpa_w512": "window 512", "single_view": "Single-view PGD", "eot": "EOT random crop"}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Ablations on Qwen3-VL-4B-Thinking (E8 subset, $\epsilon=8/255$): ASR (\%) with 95\% CI, wall-clock "
         r"seconds per image and model forward--backward passes.}", r"\label{tab:ablation}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc}", r"\toprule",
         r"Variant & ASR & s/img & Passes \\", r"\midrule"]
    for v, nm in names.items():
        r = get(e8, method=v)
        if r is None:
            continue
        cs = cost[(cost.exp == "E8") & (cost.cell.str.startswith(v + "@"))] if len(cost) else []
        s = f"{cs.wall_s.iloc[0]:.0f}" if len(cs) else "--"
        p = f"{cs.fwd_bwd.iloc[0]:.0f}" if len(cs) else "--"
        L.append(f"{nm} & {pct(r.ASR)} [{pct(r.ASR_lo)}, {pct(r.ASR_hi)}] & {s} & {p}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    open(f"{out}/tab6_ablation.tex", "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = pd.read_csv(f"{a.release}/cells.csv")
    h = json.load(open(f"{a.release}/hypotheses.json"))
    cost = pd.read_csv(f"{a.release}/cost.csv") if os.path.getsize(f"{a.release}/cost.csv") > 1 else pd.DataFrame()
    tab2_main(c, a.out); tab3_transfer(c, a.out); tab4_h1(h, a.out); tab5_rs(c, a.out); tab6_ablation(c, cost, a.out)
    print("tables:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
