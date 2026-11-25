"""Evidence-isolated re-answering (EIR): a training-free remedy for first-glance lock-in (2026-11-06).

After the agent has finished its own tool-using trajectory, the same model is asked again in a fresh single-turn
conversation that contains only the zoomed views the agent re-observed (pixels exactly as the agent saw them; at
deployment they are crops of the perturbed image), the question and the options -- no full view and none of the
agent's earlier reasoning.  Tool off, the model's official tool-off prompt and decoding budget, greedy.
Fallback: no fed-back observation, or an unparsable / truncated re-answer -> the agent's own answer.
Variant EIR+T (thumb=True): a low-resolution overview of the full view (long side 448, bilinear antialias, uint8)
is shown before the zoomed views.

Every view is recomputed from x01 by the agent's own differentiable operators, so an adaptive attacker can take
gradients through the remedy (answer_logits).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tpa.models.qwen_agent import QwenAgent, Sample, Trajectory, LETTERS, NO_TOOL_USER_SUFFIX
from tpa.processors.qwen_diff import ste_round_uint8

CROP_NOTE = "The images above are zoomed-in views of regions of one image.\n"
THUMB_NOTE = ("The first image is a low-resolution overview of one image; the other images are zoomed-in views of "
              "its regions.\n")
THUMB_LONG = 448


def thumbnail(x01: torch.Tensor, long_side: int = THUMB_LONG) -> torch.Tensor:
    h, w = x01.shape[-2:]
    r = long_side / max(h, w)
    if r >= 1:
        return x01
    th, tw = max(28, round(h * r)), max(28, round(w * r))
    t = F.interpolate(x01[None].float(), size=(th, tw), mode="bilinear", antialias=True, align_corners=False)[0]
    return ste_round_uint8(t * 255) / 255


class EIR(QwenAgent):
    """Shares the base agent's weights, processor and vision spec; only the conversation differs."""

    STOP_STRINGS = QwenAgent.STOP_STRINGS
    ANSWER_OPEN = QwenAgent.ANSWER_OPEN
    BOX_HAS_SRC = False

    def __init__(self, base, off_cfg: dict, thumb: bool = False):
        self.__dict__.update(base.__dict__)
        self.base, self.thumb = base, thumb
        self.tool_mode = False
        self.answer_mode = off_cfg["answer_mode"]
        self.system_prompt = off_cfg.get("system_prompt")
        self.user_suffix = off_cfg.get("user_suffix", NO_TOOL_USER_SUFFIX)
        self.max_new_tokens = off_cfg["max_new_tokens"]
        self.zoom_source = self.zoom_box_map = self.zoom_transform = None
        self.tf_window = None
        self._init_answer_tokens()
        self._init_probe_tokens()
        self.src = None                      # agent trajectory whose fed-back observations are re-used

    @property
    def grad_ckpt(self):                     # follow the base agent (the attack enables checkpointing there)
        return self.base.grad_ckpt

    @grad_ckpt.setter
    def grad_ckpt(self, v):
        pass

    # ------------------------------------------------------------ conversation
    def set_source(self, traj: Trajectory):
        self.src = traj

    def _n_views(self):
        return self.src.n_reobs + (1 if self.thumb else 0)

    def _messages(self, s: Sample, traj, upto_turn):
        opts = "\n" + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(s.options))
        note = THUMB_NOTE if self.thumb else CROP_NOTE
        user = note + f"Question: {s.question}\nOptions: {opts}" + self.user_suffix
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        n = self._n_views()
        msgs.append({"role": "user", "content": [{"type": "image"}] * n + [{"type": "text", "text": user}]})
        return msgs, n

    def _views(self, x01: torch.Tensor, traj, upto_turn):
        last = self.src.obs_turns()[-1]
        crops = self.base._views(x01, self.src, last)[1:]
        assert len(crops) == self.src.n_reobs, (len(crops), self.src.n_reobs)
        return ([thumbnail(x01)] if self.thumb else []) + crops

    # ------------------------------------------------------------ remedy
    @staticmethod
    def applies(traj: Trajectory) -> bool:
        return traj.n_reobs >= 1

    @torch.no_grad()
    def answer(self, x01: torch.Tensor, s: Sample, traj: Trajectory) -> dict:
        """Defended decision for one finished agent trajectory."""
        if not self.applies(traj):
            return dict(letter=traj.final_letter, applied=False, fallback="no observation")
        self.set_source(traj)
        r = self.rollout(x01.to(self.device), s, max_turns=1, max_new_tokens=self.max_new_tokens)
        out = dict(eir_letter=r.final_letter, eir_status=r.status, n_tokens=r.n_tokens, n_views=self._n_views(),
                   text=r.final_text[-1500:], applied=True)
        if r.final_letter is None:
            out.update(letter=traj.final_letter, fallback=f"re-answer {r.status}")
        else:
            out.update(letter=r.final_letter, fallback=None)
        return out

    def answer_logits(self, x01: torch.Tensor, s: Sample, traj: Trajectory):
        """Letter logits of the re-answer, read where the tool-off prompt writes the letter (the position the
        single-view attack uses); differentiable w.r.t. x01."""
        self.set_source(traj)
        return self.probe_logits(x01, s, None, 0, direct=True)


def make_eir(base, variant: str = "eir"):
    """The re-answer uses the conversation and answer format of the model's tool-off entry: QwenAgent for the
    Qwen-based victims, GLMAgent (<answer><|begin_of_box|>X) for GLM-4.1V."""
    from tpa.models.registry import REGISTRY
    assert variant in ("eir", "eirt"), variant
    off = REGISTRY[base.key + "_tool_off"]
    cls = EIR
    if off["cls"] == "glm":
        from tpa.models.glm_agent import GLMAgent
        cls = type("GLMEIR", (EIR, GLMAgent), dict(ANSWER_OPEN=GLMAgent.ANSWER_OPEN, ANSWER_CLOSE=GLMAgent.ANSWER_CLOSE,
                                                    STOP_STRINGS=GLMAgent.STOP_STRINGS))
    return cls(base, off, thumb=variant == "eirt")
