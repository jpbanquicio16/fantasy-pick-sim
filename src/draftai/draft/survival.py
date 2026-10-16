"""Probability a player is still available at your next pick.

Draft position D ~ Normal(ADP, sigma^2), sigma = a + b * ADP, conditioned on what the
board already shows: he has survived to the current pick t.

    s = P(D >= t' | D >= t) = (1 - Phi((t' - ADP) / sigma)) / (1 - Phi((t - ADP) / sigma))

Without the conditioning, a player who has already fallen past his ADP looks almost
certain to be gone, the opposite of what the board says. Picks are integers, so both
bounds get a half-pick continuity correction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def sigma(adp: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.asarray(adp, dtype=float)


def survival(adp: np.ndarray, current_pick: int, next_pick: int | None, a: float, b: float) -> np.ndarray:
    """P(still available at ``next_pick`` | available at ``current_pick``)."""
    adp = np.asarray(adp, dtype=float)
    if next_pick is None:
        return np.zeros_like(adp)
    if next_pick <= current_pick:
        return np.ones_like(adp)
    s = sigma(adp, a, b)
    log_num = norm.logsf((next_pick - 0.5 - adp) / s)
    log_den = norm.logsf((current_pick - 0.5 - adp) / s)
    return np.clip(np.exp(log_num - log_den), 0.0, 1.0)


def fit_sigma(adp: np.ndarray, picked_at: np.ndarray) -> tuple[float, float]:
    """Fit sigma = a + b * ADP from observed (ADP, pick) pairs in past drafts.

    Uses E|D - ADP| = sigma * sqrt(2 / pi) and least squares on the absolute deviations.
    """
    adp, picked_at = np.asarray(adp, float), np.asarray(picked_at, float)
    dev = np.abs(picked_at - adp) / np.sqrt(2 / np.pi)
    X = np.column_stack([np.ones_like(adp), adp])
    (a, b), *_ = np.linalg.lstsq(X, dev, rcond=None)
    return float(max(a, 0.5)), float(max(b, 0.01))


def brier(pred: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(outcome)) ** 2))
