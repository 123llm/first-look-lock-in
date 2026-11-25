import sys
sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from tpa.eval.parser import parse_answer as P

O4 = ["red", "blue", "green", "white"]
cases = [
    # tool mode
    (("<think>x</think><answer>B</answer>", O4, "tool"), ("B", "ok")),
    (("<answer> B. It plateaus. </answer>", O4, "tool"), ("B", "ok")),
    (("<answer>(C)</answer>", O4, "tool"), ("C", "ok")),
    (("<answer> green </answer>", O4, "tool"), ("C", "ok")),
    (("<answer>E</answer>", O4, "tool"), (None, "unparsable")),          # letter outside option range
    (("<answer>The color is red</answer>", O4, "tool"), (None, "unparsable")),
    (("<think>A is wrong, B fits", O4, "tool"), (None, "unparsable")),   # no answer tag -> never parse reasoning
    (("<answer>B", O4, "tool"), (None, "truncated")),
    (("<answer>A</answer> then <answer>D</answer>", O4, "tool"), ("D", "ok")),
    # thinking mode
    (("<think>A or B? ... it's B</think>The answer is B.", O4, "thinking"), ("B", "ok")),
    (("<think>long reasoning A", O4, "thinking"), (None, "truncated")),
    ((r"<think>x</think>\boxed{D}", O4, "thinking"), ("D", "ok")),
    (("<think>x</think><answer>C</answer>", O4, "thinking"), ("C", "ok")),
    (("<think>x</think>White.", O4, "thinking"), ("D", "ok")),
    (("<think>x</think>I cannot tell.", O4, "thinking"), (None, "unparsable")),
    # instruct
    (("B", O4, "instruct"), ("B", "ok")),
    (("B.", O4, "instruct"), ("B", "ok")),
    (("(A) red", O4, "instruct"), ("A", "ok")),
    (("blue", O4, "instruct"), ("B", "ok")),
    # v2 additions (S6 audit)
    (('<think>x</think>To decide: A bus is visible. **Answer: B**', O4, 'thinking'), ('B', 'ok')),
    (('<think>x</think>A. red is wrong. Thus, the correct answer is **C. green**.', O4, 'thinking'), ('C', 'ok')),
    (('<think>x</think><answer>None of the options fit; maybe B.</answer>', O4, 'thinking'), (None, 'unparsable')),
    (('<think>x</think> <answer> Final Answer:D</answer>', O4, 'thinking'), ('D', 'ok')),
    (('A long explanation without a choice', O4, 'instruct'), ('A', 'ok')),
]
bad = [(a, P(*a), e) for a, e in cases if P(*a) != e]
for b in bad: print("FAIL", b)
print(f"{len(cases) - len(bad)}/{len(cases)} passed")
assert not bad
