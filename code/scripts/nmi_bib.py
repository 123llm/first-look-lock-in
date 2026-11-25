"""paper_nmi/refs.bib from paper/refs.bib: keep the entries cited in paper_nmi/main.tex and drop the
issue number of conference volumes (naturemag.bst warns when volume and number are both given)."""
import re
s = open("paper/refs.bib", encoding="utf-8").read()
tex = open("paper_nmi/main.tex", encoding="utf-8").read()
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
    cited.update(k.strip() for k in m.group(1).split(","))
out = []
for e in re.split(r"\n(?=@)", s):
    m = re.match(r"\s*@(\w+)\{([^,]+),", e)
    if not m or m.group(2) not in cited:
        continue
    if m.group(1).lower() == "inproceedings" and re.search(r"\n\s*volume\s*=", e):
        e = re.sub(r"\n\s*number\s*=\s*\{[^}]*\},?", "", e)
    e = re.sub(r"\n\s*url\s*=\s*\{[^}]*\},?", "", e)   # naturemag prints URLs; arXiv ids stay in note
    e = re.sub(r",(\s*\})\s*$", r"\1", e.strip())
    out.append(e.strip() + "\n")
open("paper_nmi/refs.bib", "w", encoding="utf-8").write("\n".join(out))
print("entries", len(out), "cited", len(cited), "missing", sorted(cited - {re.match(r"\s*@\w+\{([^,]+),", e).group(1) for e in out}))
