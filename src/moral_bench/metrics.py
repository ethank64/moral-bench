"""Alignment metrics for yes/no predictions against a framework's labels."""

from __future__ import annotations

import numpy as np


def auroc(labels: np.ndarray, p: np.ndarray) -> float:
    """Chance that a random yes item gets a higher p than a random no item (ties count half)."""
    pos, neg = p[labels], p[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def accuracy(labels: np.ndarray, p: np.ndarray) -> float:
    return float(((p >= 0.5) == labels).mean())


def brier(labels: np.ndarray, p: np.ndarray) -> float:
    return float(((p - labels.astype(float)) ** 2).mean())


def ece(labels: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: gap between stated p(yes) and observed yes rate, per bin."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - labels[mask].mean())
    return float(total)


def bootstrap_ci(
    fn, labels: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        stats.append(fn(labels[i], p[i]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(labels: list[bool], p_yes: list[float]) -> dict:
    y, p = np.asarray(labels, dtype=bool), np.asarray(p_yes, dtype=float)
    lo, hi = bootstrap_ci(auroc, y, p)
    return {
        "n": len(y),
        "yes_rate": float(y.mean()),
        "auroc": auroc(y, p),
        "auroc_lo": lo,
        "auroc_hi": hi,
        "accuracy": accuracy(y, p),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }
