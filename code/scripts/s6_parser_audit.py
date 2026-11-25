"""S6 parser audit: re-parse every saved smoke output with the frozen parser and compare with hand labels
(code/configs/parser_audit_gold.json: one string per run, one char per item in file order, '-' = no answer given /
truncated).  Labels were assigned by reading the final answer region of each output (results/S6/parser_audit_view.txt)."""
import json, sys
sys.path.insert(0, "./code")
from tpa.eval.parser import parse_answer
from tpa.models.registry import REGISTRY
R = "."
gold = json.load(open(f"{R}/code/configs/parser_audit_gold.json"))
n = err = 0; rows_out = []
for run, g in gold.items():
    key = run.replace("_xlrs", "")
    cfg = REGISTRY[key]
    mode = "tool" if cfg["tool_mode"] else cfg.get("answer_mode", "instruct")
    rows = [json.loads(l) for l in open(f"{R}/results/S6/smoke_{run}.jsonl")]
    assert len(rows) == len(g), (run, len(rows), len(g))
    for i, (r, gl) in enumerate(zip(rows, g)):
        if r["status"] == "max_turns":
            L = None
        else:
            hit = len(r["segments"]) == 1 and r["n_tokens"] >= cfg["max_new_tokens"]
            L, st = parse_answer(r["segments"][-1], r["options"], mode, hit)
        gl = None if gl == "-" else gl
        n += 1; ok = L == gl; err += not ok
        if not ok:
            rows_out.append({"run": run, "i": i, "parsed": L, "gold": gl})
res = {"parser": "v2", "n": n, "errors": err, "acc": round(1 - err / n, 4), "mismatches": rows_out}
json.dump(res, open(f"{R}/results/S6/parser_audit.json", "w"), indent=1)
print(json.dumps(res))
