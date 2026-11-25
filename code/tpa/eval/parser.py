"""Answer parser, protocol parser v2 (replaces the v1 "first standalone letter" rule,
which picked sentence-initial articles "A ..." in verbose post-think summaries).

parse_answer(text, options, mode, hit_max=False) -> (letter | None, status)
  mode = "tool"      : only the last <answer>...</answer> span is read: letter at its start, else exact option
                       text, else an explicit answer marker inside the span.
  mode = "thinking"  : text after the last </think> (no </think> -> "truncated"), then as "instruct".
  mode = "instruct"  : \\boxed{X} > last <answer> span (as in tool mode) > LAST explicit marker
                       ("answer: X", "answer is X", "Final Answer:X", markdown bold allowed) > letter at the start of
                       the text > exact option text of the whole text.
  hit_max=True (generation stopped at max_new_tokens) turns a missing answer into "truncated".
status in {"ok", "truncated", "unparsable"}; "max_turns" is set by the rollout.
Letters are restricted to the first len(options) letters.
"""
from __future__ import annotations

import re

LETTERS = "ABCDEFGH"


def _norm(s: str) -> str:
    s = re.sub(r"[*_`]", "", s)
    return re.sub(r"\s+", " ", s).strip().strip(".").strip().casefold()


def _option_text(body: str, options: list[str]) -> str | None:
    b = _norm(body)
    for i, o in enumerate(options):
        if b == _norm(o):
            return LETTERS[i]
    return None


def _letter_at_start(body: str, valid: str) -> str | None:
    m = re.match(rf"[\s*]*(?:[Oo]ption\s*)?\(?([{valid}])\)?(?![A-Za-z0-9'])", body)
    return m.group(1) if m else None


def _last_marker(body: str, valid: str) -> str | None:
    pat = rf"answer(?:\s+is)?\s*[:：]?\s*[*\s]*(?:option\s*)?\(?([{valid}])\)?(?![A-Za-z0-9'])"
    ms = list(re.finditer(pat, body, re.I))
    return ms[-1].group(1) if ms else None


def _boxed(text: str, valid: str) -> str | None:
    m = re.search(rf"\\boxed\{{\s*(?:\\text\{{)?\s*\(?([{valid}])\)?", text)
    return m.group(1) if m else None


def _answer_span(text: str) -> str | None:
    spans = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return spans[-1] if spans else None


def _from_span(body: str, options, valid) -> str | None:
    return _letter_at_start(body, valid) or _option_text(body, options) or _last_marker(body, valid)


def parse_answer(text: str, options: list[str], mode: str, hit_max: bool = False) -> tuple[str | None, str]:
    valid = LETTERS[:len(options)]
    miss = "truncated" if hit_max else "unparsable"
    if mode == "tool":
        body = _answer_span(text)
        if body is None:
            return None, "truncated" if (hit_max or "<answer>" in text) else "unparsable"
        L = _from_span(body, options, valid)
        return (L, "ok") if L else (None, "unparsable")
    if mode == "thinking":
        if "</think>" not in text:
            return None, "truncated"
        post = text.rsplit("</think>", 1)[1]
    elif mode == "instruct":
        post = text
    else:
        raise ValueError(mode)
    L = _boxed(post, valid)
    if L is None:
        body = _answer_span(post)
        if body is not None:
            L = _from_span(body, options, valid)
            return (L, "ok") if L else (None, "unparsable")   # an answer span without a choice is a non-answer
    if L is None:
        L = _last_marker(post, valid)
    if L is None:
        L = _letter_at_start(post, valid)
    if L is None:
        L = _option_text(post, options)
    return (L, "ok") if L else (None, miss)
