"""Paper tables from results/release/s18_v1 (reports + verify.json).  Writes paper/tables/tab_*.tex.

  make_paper_tables.py
Every number in the tables comes from the release files; nothing is typed by hand.
"""
import json, os

R = os.environ.get("TPA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REL = f"{R}/results/release/s18_v1"
OUT = f"{R}/paper/tables"
NAME = {"deepeyes_7b": "DeepEyes-7B", "minio3_7b": "Mini-o3-7B", "pixelreasoner_7b": "PixelReasoner-7B",
        "qwen3vl_4b_thinking": "Qwen3-VL-4B-Think", "qwen3vl_8b_thinking": "Qwen3-VL-8B-Think",
        "glm41v_9b_thinking": "GLM-4.1V-9B-Think"}
ORDER = ["deepeyes_7b", "minio3_7b", "pixelreasoner_7b", "qwen3vl_4b_thinking", "qwen3vl_8b_thinking",
         "glm41v_9b_thinking"]


def rep(name):
    return json.load(open(f"{REL}/reports/{name}_report.json"))


def f2(v):
    return "--" if v is None else f"{v:.2f}"


def fr(rate, n):
    """A proportion k/n shown with two decimals from the count, not from the 3-decimal rounded rate
    (rounding 0.0847 to 0.085 and then to 0.09 would be a double-rounding error)."""
    if rate is None or not n:
        return f2(rate)
    k = round(rate * n)
    assert abs(k / n - rate) <= 0.0005 + 1e-9, (rate, n)
    return f"{k / n:.2f}"


def c_exact(model, adv):
    """Unrounded mean first-look confidence of a cell, recomputed from the item records with the E1 report code."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import s12_e1_report as E
    from tpa import io as tio
    keep = set(E.first_qids())
    method, src, eps, seed = tio.parse_adv_spec(adv)
    ev = {r["qid"]: r for r in tio.read_jsonl(tio.raw_path("E1", "v2_main", model, method, src, eps, seed))
          if r["qid"] in keep}
    fl = [q for q, r in ev.items() if E.ok(r["clean"]) and not E.ok(r["adv"]) and r["adv"]["n_reobs"] >= 1]
    ps = [max(ev[q]["adv"]["probes"][0]["probs"]) for q in fl if ev[q]["adv"].get("probes")]
    return sum(ps) / len(ps)


def ci(s):
    return "--" if not s or s.get("ci95") is None else f"[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"


def write(name, lines):
    """Writes the table body only; the tabular* line with the column spec stays in the section file."""
    os.makedirs(OUT, exist_ok=True)
    assert lines[0].startswith(r"\begin{tabular*}") and lines[1] == r"\toprule" and lines[-1] == r"\end{tabular*}"
    # \toprule is kept in the section file: a \noalign right after \input inside an alignment is a LaTeX error
    body = "% body of " + lines[0] + " after \\toprule\n" + "\n".join(lines[2:-1]) + "\n"
    open(f"{OUT}/{name}.tex", "w", encoding="utf-8", newline="\n").write(body)
    print("wrote", name)


def tab_e1():
    e1 = rep("e1"); ver = json.load(open(f"{REL}/verify.json"))["E1"]
    L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccccccc}", r"\toprule",
         r"Victim & Perturbation & ASR & Wrong letter & $|\mathcal{F}_z|$ & Lock rate & $\bar c$ & $R_{\mathrm{clean}}$ [95\% CI] & $R_{\mathrm{rand}}$ & $C_{\mathrm{nat}}$ ($n$) & Holds \\",
         r"\midrule"]
    for role in ("decide", "describe"):
        if role == "describe":
            L += [r"\midrule", r"\multicolumn{11}{l}{\emph{Transfer from DeepEyes-7B, $\epsilon=8/255$ (descriptive)}} \\"]
        for m in ORDER:
            for c in e1["cells"]:
                if c["model"] != m or c["role"] != role:
                    continue
                v = ver[f"{m}|{c['adv']}"]
                fl = sum(v["flipped_status"].values())
                wl = v["flipped_status"].get("wrong_letter", 0) / fl if fl else None
                pert = f"WB $\\epsilon={c['eps']}$" if role == "decide" else "transfer"
                holds = ("yes" if c["holds"] else "no") if role == "decide" else "--"
                cx = c_exact(m, c["adv"])
                assert abs(cx - c["turn0_maxprob"]) <= 0.0005 + 1e-9
                L.append(f"{NAME[m]} & {pert} & {fr(c['asr']['rate'], c['asr']['n'])} & {f2(wl)} & {c['n_flipped_zoomed']} & "
                         f"{fr(c['lock'], c['n_flipped_zoomed'])} & {f2(cx)} & {fr(c['R_clean']['rate'], c['R_clean']['n'])} {{\scriptsize {ci(c['R_clean'])}}} & "
                         f"{fr(c['R_rand']['rate'], c['R_rand']['n'])} & {fr(c['C_nat']['rate'], c['C_nat']['n'])} ({c['C_nat']['n']}) & {holds} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}"]
    write("tab_e1", L)


def tab_mech():
    e2 = rep("e2"); e4r = rep("e4r")
    L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccccc}", r"\toprule",
         r" & \multicolumn{7}{c}{Channel replay, $A(G,T)$, white-box $\epsilon=8/255$} & \multicolumn{5}{c}{EIR} \\",
         r"\cmidrule(lr){2-8}\cmidrule(lr){9-13}",
         r"Victim & $n$ & $\tilde x$, own & $\tilde x$, neu. & gray, own & gray, neu. & $V$ & Mech. & $\Delta A$ [95\% CI] & Rec$_8$ & Rec$_4$ & Fallb. & Pass \\",
         r"\midrule"]
    for m in ORDER:
        a = e2["models"][m]; A = a["A"]
        mech = a["mechanism"] if a["sufficiency"] else "insuff."
        r1 = e4r["models"][m]["R1"]
        def rr(e):
            c = e4r["cells"].get(f"{m}__single_view@{m}__eps{e}__seed0")
            return fr(c["Rec"], c["n_flipped"]) if c else "--"
        n = a["n"]; k = {g: round(A[g] * n) for g in A}
        assert all(abs(k[g] / n - A[g]) <= 0.0005 + 1e-9 for g in A)
        V = (k["gray|own"] - k["adv|own"]) / n
        assert abs(V - a["V"]) <= 0.001 + 1e-9
        L.append(f"{NAME[m]} & {n} & {fr(A['adv|own'], n)} & {fr(A['adv|neutral'], n)} & {fr(A['gray|own'], n)} & "
                 f"{fr(A['gray|neutral'], n)} & {V:.2f} & {mech} & {r1['dA']:+.2f} [{r1['dA_ci95'][0]:.2f}, {r1['dA_ci95'][1]:.2f}] & "
                 f"{rr(8)} & {rr(4)} & {r1['fallback_reanswer']} & {'yes' if e4r['models'][m]['passes'] else 'no'} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}"]
    write("tab_mech", L)


def tab_e4a():
    e4 = rep("e4a")
    L = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc}", r"\toprule",
         r" & DeepEyes & Mini-o3 & PixelReasoner \\", r"\midrule",
         r"\multicolumn{4}{l}{\emph{ASR of the defended system (agent + EIR)}} \\"]
    for att, lab in (("non-adaptive", "Non-adaptive (WB)"), ("EOT-R", "EOT-R (random)"),
                     ("TPA-R", "TPA-R"), ("TPA-R+J", "TPA-R+J")):
        row = [f2(e4[m]["attacks"][att]["ASR_def"]["rate"]) for m in ("deepeyes_7b", "minio3_7b", "pixelreasoner_7b")]
        L.append(f"{lab} & " + " & ".join(row) + r" \\")
    L += [r"\midrule", r"\multicolumn{4}{l}{\emph{Paired differences [95\% CI]}} \\"]
    for k, lab in (("TPA-R - non-adaptive", "TPA-R $-$ non-adaptive"), ("TPA-R - EOT-R", "TPA-R $-$ EOT-R"),
                   ("TPA-R+J - TPA-R", "TPA-R+J $-$ TPA-R")):
        vals, cis = [], []
        for m in ("deepeyes_7b", "minio3_7b", "pixelreasoner_7b"):
            s = e4[m][k]; v = f"{s['rate']:+.2f}"
            if s["ci95"][0] > 0 or s["ci95"][1] < 0:
                v = r"\textbf{" + v + "}"
            vals.append(v)
            cis.append(r"{\scriptsize[" + f"{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}" + "]}")
        L.append(f"{lab} & " + " & ".join(vals) + r" \\")
        L.append(" & " + " & ".join(cis) + r" \\[1pt]")
    ag = [f2(e4[m]["attacks"]["TPA-R"]["ASR_agent"]["rate"]) for m in ("deepeyes_7b", "minio3_7b", "pixelreasoner_7b")]
    L += [r"\midrule", r"No defense, TPA-R images & " + " & ".join(ag) + r" \\", r"\bottomrule", r"\end{tabular*}"]
    write("tab_e4a", L)


def tab_ext():
    e5 = rep("e5"); e7 = rep("e7")
    L = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc}", r"\toprule", r" & DeepEyes & Mini-o3 \\", r"\midrule",
         r"\multicolumn{3}{l}{\emph{White-box ASR at $\epsilon=2/255$, first 100 questions}} \\"]
    for k, lab in (("single_view", "Single view"), ("eot", "EOT (random boxes)"), ("tpa", "TPA")):
        L.append(f"{lab} & " + " & ".join(fr(e5[m][k]["rate"], e5[m][k]["n"]) for m in ("deepeyes_7b", "minio3_7b")) + r" \\")
    L.append(r"$|\mathcal{C}|$ & " + " & ".join(str(e5[m]["single_view"]["n"]) for m in ("deepeyes_7b", "minio3_7b")) + r" \\")
    for k, lab in (("tpa_minus_single_view", "TPA $-$ single view"), ("tpa_minus_eot", "TPA $-$ EOT")):
        L.append(f"{lab} & " + " & ".join(f"{e5[m][k]['rate']:+.2f} {ci(e5[m][k])}" for m in ("deepeyes_7b", "minio3_7b")) + r" \\")
    L += [r"\midrule", r"\multicolumn{3}{l}{\emph{XLRS-Bench, white-box single view, $\epsilon=4/255$}} \\"]
    cells = {c["model"]: c for c in e7["cells"] if c["adv"].startswith("single_view")}
    m2 = ("deepeyes_7b", "minio3_7b")
    L.append("Clean accuracy & " + " & ".join(fr(cells[m]["C_nat"]["clean_acc"], cells[m]["n_items"]) for m in m2) + r" \\")
    L.append("ASR ($|\\mathcal{C}|$) & " + " & ".join(f"{f2(cells[m]['asr']['rate'])} ({cells[m]['asr']['n']})" for m in m2) + r" \\")
    L.append(r"$R_{\mathrm{clean}}$ / $R_{\mathrm{rand}}$ & " + " & ".join(f"{fr(cells[m]['R_clean']['rate'], cells[m]['R_clean']['n'])} / {fr(cells[m]['R_rand']['rate'], cells[m]['R_rand']['n'])}" for m in m2) + r" \\")
    L.append(r"$C_{\mathrm{nat}}$ ($n$) & " + " & ".join(f"{fr(cells[m]['C_nat']['rate'], cells[m]['C_nat']['n'])} ({cells[m]['C_nat']['n']})" for m in m2) + r" \\")
    L.append("Lock-in & " + " & ".join("yes" if cells[m]["holds"] else "no" for m in m2) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}"]
    write("tab_ext", L)


def tab_s_e3():
    e3 = rep("e3")
    L = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llccc}", r"\toprule",
         r"Victim & Images & Units & $\bar c$ & $R_{\mathrm{clean}}$ \\", r"\midrule"]
    lab = {"s2": "PGD step 2", "s5": "PGD step 5", "s10": "PGD step 10", "s25": "PGD step 25", "s50": "PGD step 50",
           "transfer": "DeepEyes transfer"}
    for m in ("deepeyes_7b", "minio3_7b", "qwen3vl_4b_thinking"):
        for k in ("s2", "s5", "s10", "s25", "s50", "transfer"):
            v = e3["per_level"].get(f"{m}|{k}")
            if v:
                L.append(f"{NAME[m]} & {lab[k]} & {v['n']} & {v['c_mean']:.3f} & {fr(v['R_clean'], v['n'])} \\\\")
        L.append(r"\addlinespace")
    L[-1] = r"\midrule"
    L += [f"\\multicolumn{{5}}{{l}}{{All: {e3['n_units']} units, {e3['n_recovered']} recoveries, "
          f"slope $\\beta={e3['slope']:.3f}$ [{e3['slope_ci95'][0]:.3f}, {e3['slope_ci95'][1]:.3f}]}} \\\\",
          r"\bottomrule", r"\end{tabular*}"]
    write("tab_s_e3", L)


def tab_s_e6():
    e6 = rep("e6")
    L = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcccc}", r"\toprule",
         r"Victim, images & Cap & Clean acc. & $|\mathcal{C}|$ & ASR & $|\mathcal{F}_z|$ \\", r"\midrule"]
    for key, lab in (("deepeyes_7b|single_view@deepeyes_7b__eps8__seed0", "DeepEyes, WB"),
                     ("minio3_7b|single_view@minio3_7b__eps8__seed0", "Mini-o3, WB"),
                     ("minio3_7b|single_view@deepeyes_7b__eps8__seed0", "Mini-o3, transfer")):
        for c in ("0", "1", "2", "4"):
            l = e6[key]["levels"][c]
            L.append(f"{lab if c == '0' else ''} & {c} & {fr(l['clean_acc'], 150)} & {l['ASR']['n']} & {fr(l['ASR']['rate'], l['ASR']['n'])} & {l['n_rclean_computable']} \\\\")
        s = e6[key]
        sl = "not estimable (ASR constant)" if s["slope"] is None else f"slope {s['slope']:.2f} [{s['slope_ci95'][0]:.2f}, {s['slope_ci95'][1]:.2f}]"
        L.append(f"\\multicolumn{{6}}{{r}}{{\\emph{{{sl}}}}} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular*}"]
    write("tab_s_e6", L)


if __name__ == "__main__":
    tab_e1(); tab_e4a(); tab_ext()
