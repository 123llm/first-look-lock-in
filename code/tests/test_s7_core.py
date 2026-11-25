"""S7 unit tests (CPU, no model weights): PGD engine + EoSO term construction with a fake differentiable victim,
RTAP metric definitions on hand-built rows, statistics, VEG box intervention, trajectory bookkeeping."""
import sys
sys.path.insert(0, __import__("os").environ.get("TPA_ROOT", ".") + "/code")

import numpy as np
import torch

from tpa.models.qwen_agent import Trajectory, Sample
from tpa.attacks.tpa import Attack, AttackConfig, sample_eot_box, quantize
from tpa.metrics import rtap
from tpa.metrics.stats import mcnemar_exact, holm, cluster_bootstrap, kendall_tau_b
from tpa.eval.regrounding import random_box_same_size

torch.manual_seed(0)
H, W = 64, 96


class FakeAgent:
    """Letter logits = linear readout of the mean colour of every view the conversation contains.
    The original view and each zoom view have different weights, so terms are distinguishable."""
    BOX_HAS_SRC = False
    device = "cpu"

    def __init__(self):
        g = torch.Generator().manual_seed(1)
        self.Wo = torch.randn(4, 3, generator=g)
        self.Wz = torch.randn(4, 3, generator=g)
        self.tf_window = None
        self.sampling = {}
        self.calls = []
        self.n_rollouts = 0

    def enable_grad_ckpt(self):
        pass

    def _feat(self, x, traj, upto):
        f = self.Wo @ x.mean((1, 2))
        if traj is not None:
            for b in traj.boxes[:upto]:
                if b is not None:
                    f = f + self.Wz @ x[:, b[1]:b[3], b[0]:b[2]].mean((1, 2))
        return f * 10

    def probe_logits(self, x, s, traj, upto, direct=False):
        self.calls.append(("probe", upto, direct))
        return self._feat(x, traj, upto)

    def final_answer_logits(self, x, s, traj):
        self.calls.append(("final", len(traj.segments)))
        return self._feat(x, traj, len(traj.segments)) + 0.1

    def rollout(self, x, s, max_turns, max_new_tokens, gen_kwargs=None, seed=None, **kw):
        self.n_rollouts += 1
        b = (10, 5, 40, 30)
        return Trajectory(segments=["think <tool_call>", "<answer>A</answer>"], boxes=[b, None],
                          final_letter="A", n_tool_calls=1, status="ok")

    def action_segment(self, box, x):
        return "<tool_call>"


S = Sample("img0", "q", ["a", "b", "c", "d"], 0)


def test_pgd_engine():
    x0 = quantize(torch.rand(3, H, W))
    for method in ["single_view", "tpa", "eot"]:
        ag = FakeAgent()
        cfg = AttackConfig(method=method, eps=8, steps=12, M=2, K=5, seed=0)
        xa, log = Attack(ag, cfg).run(x0, S)
        d = (xa - x0).abs().max() * 255
        assert d <= 8 + 1e-4, (method, d)
        assert torch.allclose(xa * 255, (xa * 255).round(), atol=1e-4), "not on the uint8 grid"
        J = [st["J"] for st in log["steps"]]
        assert J[-1] > J[0], (method, J[:3], J[-3:])
        if method == "single_view":
            assert log["n_fwd_bwd"] == 12 and log["n_rollouts"] == 0
            assert all(c == ("probe", 0, True) for c in ag.calls)
        if method == "tpa":
            # resample at k = 0, 5, 10 -> 3 x M rollouts; identical trajectories are deduplicated:
            # final + probe(0) + probe(1) = 3 terms per step
            assert log["n_rollouts"] == 6 and ag.n_rollouts == 6, log["n_rollouts"]
            assert log["n_fwd_bwd"] == 12 * 3, log["n_fwd_bwd"]
            w = {(t["kind"], t["upto"]): t["w"] for t in log["steps"][0]["terms"]}
            assert abs(w[("final", 2)] - 1.0) < 1e-6 and abs(w[("probe", 0)] - 0.5) < 1e-6 and abs(w[("probe", 1)] - 0.5) < 1e-6
        if method == "eot":
            assert log["n_rollouts"] == 0 and len(log["boxes"]) == 12
            # per step: probe(0) shared + one zoom probe per random box (final == probe after zoom)
            assert log["n_fwd_bwd"] == 12 * 3
    # determinism
    a1, _ = Attack(FakeAgent(), AttackConfig(method="eot", steps=4)).run(x0, S)
    a2, _ = Attack(FakeAgent(), AttackConfig(method="eot", steps=4)).run(x0, S)
    assert torch.equal(a1, a2)
    # ablations
    ag = FakeAgent()
    _, log = Attack(ag, AttackConfig(method="tpa", steps=12, K=5, use_probes=False, resample=False)).run(x0, S)
    assert log["n_rollouts"] == 2 and log["n_fwd_bwd"] == 12
    # targeted: CE(target) goes down
    _, log = Attack(FakeAgent(), AttackConfig(method="single_view", steps=10, target_idx=2)).run(x0, S)
    assert log["steps"][-1]["J"] > log["steps"][0]["J"]


def test_eot_boxes():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        b = sample_eot_box(300, 500, rng)
        bw, bh = b[2] - b[0], b[3] - b[1]
        assert 0 <= b[0] < b[2] <= 500 and 0 <= b[1] < b[3] <= 300
        assert 0.04 <= bw * bh / (300 * 500) <= 0.31
        assert 0.49 <= bw / bh <= 2.05


def test_trajectory():
    t = Trajectory(segments=["a", "b", "c"], boxes=[(0, 0, 5, 5), (1, 1, 4, 4), None], status="ok")
    assert t.obs_turns() == [0, 1, 2] and t.n_reobs == 2
    t = Trajectory(segments=["a", "b"], boxes=[(0, 0, 5, 5), (1, 1, 4, 4)], status="max_turns")
    assert t.obs_turns() == [0, 1] and t.n_reobs == 1          # last call was never shown
    assert Trajectory.from_dict(t.to_dict()) == t


def _rec(letter, status="ok", probes=(), toks=100, calls=1, text=""):
    return dict(answer="A", letter=letter, status=status, n_tokens=toks, n_calls=calls, n_reobs=calls,
                probes=[dict(turn=i, letter=p) for i, p in enumerate(probes)], traj=dict(segments=[text]))


def test_rtap():
    rows = [
        # 1: perception-level flip that persists (success, PFR, FSP=0)
        dict(qid="1", cluster="a", clean=_rec("A", probes="AA"), adv=_rec("B", probes="BB", toks=200, calls=2)),
        # 2: reasoning flip (probe0 ok, final flipped) with DbF wording
        dict(qid="2", cluster="b", clean=_rec("A", probes="AA"), adv=_rec("C", probes="AA", text="the image looks noisy")),
        # 3: self-corrected (probe flipped, final correct)
        dict(qid="3", cluster="c", clean=_rec("A", probes="AA"), adv=_rec("A", probes="BA")),
        # 4: invalid adversarial output (not a success)
        dict(qid="4", cluster="d", clean=_rec("A", probes="AA"), adv=_rec(None, status="truncated", probes="AA")),
        # 5: clean wrong -> excluded from D
        dict(qid="5", cluster="e", clean=_rec("B", probes="BB"), adv=_rec("B", probes="BB")),
    ]
    m = rtap.compute(rows, B=200)
    P = {k: v["point"] for k, v in m.items() if isinstance(v, dict)}
    assert m["n_D"] == 4
    assert P["ASR"] == 0.5 and P["INV"] == 0.25
    assert P["PFR"] == 0.5 and P["PFR_clean"] == 0.0
    assert P["RFR"] == 0.25
    assert P["SRR"] == 0.5                       # items 1 and 3 have a probe flip; 3 recovered
    assert P["FSP"] == (0.0 + 1.0) / 2           # item 1 at position 0; item 2 only at the final answer (pos 2 of 3)
    assert P["DbF"] == 0.5 and P["DbF_D"] == 0.25
    assert abs(P["CIR_tokens"] - 500 / 400) < 1e-9 and abs(P["CIR_calls"] - 5 / 4) < 1e-9
    assert m["ASR"]["lo"] <= 0.5 <= m["ASR"]["hi"]
    # targeted
    rows_t = [dict(r, target="C") for r in rows]
    assert rtap.compute(rows_t, B=0)["ASR"]["point"] == 0.25
    # DbF needs the clean trace to be silent
    r = dict(qid="6", cluster="f", clean=_rec("A", probes="A", text="noise"), adv=_rec("B", probes="B", text="noise"))
    assert rtap.compute([r], B=0)["DbF"]["point"] == 0.0


def test_stats():
    a = [True] * 10 + [False] * 10
    b = [False] * 10 + [False] * 10
    r = mcnemar_exact(a, b)
    assert r["n10"] == 10 and r["n01"] == 0 and abs(r["p"] - 2 / 2 ** 10) < 1e-12
    assert mcnemar_exact(a, a)["p"] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    p, lo, hi = cluster_bootstrap(list(range(10)), lambda r: float(np.mean(r)), [i // 2 for i in range(10)], B=500)
    assert p == 4.5 and lo < 4.5 < hi
    assert abs(kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) - 1) < 1e-12


def test_veg_box():
    for q in ["a", "b", "c"]:
        b = random_box_same_size((10, 20, 110, 70), 300, 400, q)
        assert b[2] - b[0] == 100 and b[3] - b[1] == 50 and 0 <= b[0] and b[2] <= 400 and b[3] <= 300
    assert random_box_same_size((10, 20, 110, 70), 300, 400, "a") == random_box_same_size((10, 20, 110, 70), 300, 400, "a")
    b = random_box_same_size((0, 0, 50, 50, 2), 100, 100, "x")
    assert b[4] == 2


if __name__ == "__main__":
    fs = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fs:
        f()
        print("ok", f.__name__)
    print(f"{len(fs)}/{len(fs)} passed")
