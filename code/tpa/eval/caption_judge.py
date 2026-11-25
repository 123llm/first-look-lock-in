"""H1 reference protocol: captioning + LLM judge (protocol_v2 hypotheses.H1_judge).

caption : the victim describes the clean and the adversarial image with "Describe this image in detail."
          (greedy; no tool prompt; the victim's own no-tool system prompt if it has one; for thinking models the
          text after </think> is the caption).
judge   : a text-only judge (Qwen3-VL-8B-Instruct; fallback Qwen3-VL-4B-Instruct) rates with the frozen prompt
          code/configs/prompts/h1_judge_v1.txt how well the adversarial caption describes the same content as the
          clean caption, integer 1-10.  Success = score <= threshold (5; sensitivity 3 and 7).
"""
from __future__ import annotations

import re

import torch

CAPTION_PROMPT = "Describe this image in detail."
JUDGE_PROMPT_FILE = __import__("os").environ.get("TPA_ROOT", ".") + "/code/configs/prompts/h1_judge_v1.txt"


def _generate_text(agent, msgs, views, max_new_tokens):
    enc = agent._encode(msgs, views, add_generation_prompt=True) if views else \
        {k: v.to(agent.device) for k, v in agent.processor(
            text=[agent.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)],
            return_tensors="pt").items()}
    new = agent.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                               top_k=None, use_cache=True, pad_token_id=agent.tok.pad_token_id)[0, enc["input_ids"].shape[1]:]
    return agent.tok.decode(new, skip_special_tokens=True), int(new.numel())


@torch.no_grad()
def caption(agent, x01: torch.Tensor, max_new_tokens: int | None = None) -> dict:
    msgs = []
    if getattr(agent, "system_prompt", None) and not agent.tool_mode:
        msgs.append({"role": "system", "content": agent.system_prompt})
    msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]})
    mnt = max_new_tokens or (agent.max_new_tokens if (agent.opens_think or agent.answer_mode == "thinking") else 512)
    text, n = _generate_text(agent, msgs, [x01.to(agent.device)], mnt)
    cap = text.split("</think>")[-1]
    cap = re.sub(r"</?answer>", "", cap).strip()
    return dict(caption=cap, raw=text, n_tokens=n, truncated=n >= mnt and "</think>" not in text and agent.opens_think)


def judge_prompt(cap_clean: str, cap_adv: str) -> str:
    tpl = open(JUDGE_PROMPT_FILE).read()
    return tpl.replace("{caption_clean}", cap_clean).replace("{caption_adv}", cap_adv)


def parse_score(text: str) -> int | None:
    m = re.search(r"\b(10|[1-9])\b", text.split("</think>")[-1])
    return int(m.group(1)) if m else None


@torch.no_grad()
def judge(judge_agent, cap_clean: str, cap_adv: str, max_new_tokens: int = 8) -> dict:
    msgs = [{"role": "user", "content": [{"type": "text", "text": judge_prompt(cap_clean, cap_adv)}]}]
    text, _ = _generate_text(judge_agent, msgs, None, max_new_tokens)
    return dict(score=parse_score(text), raw=text)


def judge_success(score: int | None, threshold: int = 5) -> bool | None:
    return None if score is None else score <= threshold
