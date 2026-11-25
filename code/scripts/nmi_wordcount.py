"""Word counts of paper_nmi/main.tex by NMI rules: abstract; main text = Introduction + Results + Discussion
(without abstract, Methods, references, figure legends); Methods.  Display items are not counted as text."""
import re, sys
s = open(sys.argv[1] if len(sys.argv) > 1 else "paper_nmi/main.tex", encoding="utf-8").read()


def words(t):
    t = re.sub(r"(?s)\\begin\{(figure|table|algorithm|equation)\}.*?\\end\{\1\}", " ", t)
    t = re.sub(r"(?m)(?<!\\)%.*$", " ", t)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\cite\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}~]", " ", t)
    return len(t.split())


def between(a, b):
    return s[s.index(a) + len(a):s.index(b)]


print("abstract", words(between(r"\begin{abstract}", r"\end{abstract}")))
print("main text", words(between(r"\end{abstract}", r"\section*{Methods}")))
print("methods", words(between(r"\section*{Methods}", r"\section*{Data availability}")))
print("display items (main):", len(re.findall(r"\\begin\{(figure|table)\}", between(r"\end{abstract}", r"\section*{Methods}"))))
