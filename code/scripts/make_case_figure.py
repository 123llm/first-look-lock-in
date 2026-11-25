"""Fig.8 case study: Mini-o3 on V* direct_attributes/sa_79105 (answer D, red).  -> figures/src/fig8.svg
Images: figures/src/case/*.png (make_case_crops.py).  Quotes are verbatim excerpts of the recorded outputs:
clean and adversarial trajectories and the clean-zoom replay (results/raw/E1, eps8), EIR re-answers
(results/raw/E4/eir__minio3_7b__E1__single_view@... and eir__minio3_7b__E4__tpa_eir@...)."""
import os, textwrap
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

R = os.environ.get("TPA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
C = f"{R}/figures/src/case"
mpl.rcParams.update({"svg.fonttype": "none", "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                     "mathtext.fontset": "stix", "font.size": 7, "figure.dpi": 300})
ADV, CLEAN, OK = "#D55E00", "#1F5FA0", "#2E7D32"
S = 600 / 2250  # global views are stored at 600 x 400
panels = [
    dict(img="clean_global.png", box=(450, 750, 1350, 1200), edge=CLEAN,
         title="(a) Clean image",
         sub="first look: D (red), c = 0.997",
         quote="I can see a red object in the middle of the image that looks like a seesaw.",
         ans="final answer: D", ok=True),
    dict(img="sv_global.png", box=(450, 600, 1575, 1050), edge=ADV,
         title="(b) Adversarial image, ε = 8/255",
         sub="first look: B (black), c = 1.000",
         quote="I can see a seesaw in the middle of the image ... It appears to be black.",
         ans="final answer: B", ok=False),
    dict(img="clean_crop_sv.png", edge=CLEAN,
         title="(c) Clean view at the same box",
         sub="clean-zoom replay of (b)",
         quote="In observation 1, I can clearly see a red seesaw in the background. ... Therefore, the color of "
               "the seesaw is red.",
         ans="final answer: B", ok=False),
    dict(img="sv_crop.png", edge=ADV,
         title="(d) EIR on the view of (b)",
         sub="re-answer from the returned view only",
         quote="I can see a red seesaw on the left side of the image ... The seesaw is clearly red.",
         ans="EIR answer: D", ok=True),
    dict(img="tpa_crop.png", edge=ADV,
         title="(e) EIR under TPA-R",
         sub="view at the victim's own box",
         quote="I can see a black seesaw in the middle of the image ... Its color is clearly black.",
         ans="EIR answer: B", ok=False),
]


def main():
    fig = plt.figure(figsize=(7.1, 1.8))
    W, gap = 0.188, 0.0155
    for i, p in enumerate(panels):
        x0 = 0.005 + i * (W + gap)
        ax = fig.add_axes([x0, 0.37, W, 0.47])
        im = Image.open(f"{C}/{p['img']}")
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(p["edge"]); s.set_linewidth(1.6)
        if "box" in p:
            b = [v * S for v in p["box"]]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#111111", lw=1.0))
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="white", lw=0.5, ls="--"))
        ax.set_aspect("equal"); ax.set_anchor("S")
        fig.text(x0 + W / 2, 0.985, p["title"], ha="center", va="top", fontsize=7.2, weight="bold")
        fig.text(x0 + W / 2, 0.905, p["sub"], ha="center", va="top", fontsize=6.4, color="#444444")
        q = "“" + p["quote"] + "”"
        fig.text(x0 + 0.004, 0.335, "\n".join(textwrap.wrap(q, 37)), ha="left", va="top", fontsize=6.2,
                 style="italic", color="#222222", linespacing=1.15)
        col = OK if p["ok"] else ADV
        fig.text(x0 + W / 2, 0.01, p["ans"] + ("  (correct)" if p["ok"] else "  (wrong)"), ha="center",
                 va="bottom", fontsize=6.8, weight="bold", color=col)
    out = f"{R}/figures/src/fig8.svg"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
