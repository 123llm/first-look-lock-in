"""Statistics used across hypotheses (protocol_v2 `statistics:`): image-cluster bootstrap (B=2000, seed 0),
exact McNemar, Holm, Kendall tau-b, Spearman."""
from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np


def cluster_bootstrap(rows: list, stat, clusters: list, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of stat(rows) resampling whole clusters (images) with replacement.
    Returns (point, lo, hi).  NaN resamples (e.g. empty conditional denominators) are dropped."""
    point = stat(rows)
    groups = defaultdict(list)
    for r, c in zip(rows, clusters):
        groups[c].append(r)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        rs = [r for i in pick for r in groups[keys[i]]]
        v = stat(rs)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Paired binary outcomes a_i, b_i.  n01 = a false & b true, n10 = a true & b false.
    Two-sided exact binomial test on the discordant pairs."""
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(n10=n10, n01=n01, p=p)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def kendall_tau_b(x, y) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(x, y, variant="b").statistic)


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)
