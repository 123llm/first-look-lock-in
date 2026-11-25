"""Probe / answer prefixes resolve to one letter token per option for every registered victim (tokenizer only)."""
import sys
sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")
from transformers import AutoProcessor
from tpa.models.qwen_agent import QwenAgent, LETTERS
from tpa.models.minio3_agent import MiniO3Agent
from tpa.models.registry import REGISTRY

seen = {}
for key, cfg in REGISTRY.items():
    cls = MiniO3Agent if cfg["cls"] == "minio3" else QwenAgent
    a = cls.__new__(cls)
    if cfg["path"] not in seen:
        seen[cfg["path"]] = AutoProcessor.from_pretrained(cfg["path"])
    a.processor = seen[cfg["path"]]; a.tok = a.processor.tokenizer
    a.tool_mode = cfg.get("tool_mode", True); a.answer_mode = cfg.get("answer_mode", "instruct")
    a.letter_ids = [a.tok.convert_tokens_to_ids(c) for c in LETTERS]
    a._init_answer_tokens(); a._init_probe_tokens()
    assert len(set(a.probe_letter_ids)) == len(LETTERS)
    print(f"{key:30s} opens_think={a.opens_think!s:5s} probe={a.probe_prefix_text!r} -> {a.tok.convert_ids_to_tokens(a.probe_letter_ids[:2])} direct={a.direct_prefix_text!r}")
print("ok")
