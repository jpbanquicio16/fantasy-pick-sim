"""Per-minute rates with hierarchical (Bayesian) shrinkage: a regression prior.

A 300-minute season produces wild per-minute rates. Each player's weighted rate is shrunk
toward a prior. The original prior (``bucket``, kept for comparison) is the mean rate of
his height bucket, with a method-of-moments strength. The hierarchical prior (V5) lets the
prior mean depend on who he is:

    lambda_j ~ Gamma(shape k * mu_j, rate k)        mean mu_j, variance mu_j / k
    mu_j     = exp(z_j . beta)                      z: height bucket, age, age^2,
                                                    experience, log draft pick
    x_j | lambda_j ~ Poisson(lambda_j * m_j)        x counts in m minutes

so a second-year player with 400 minutes is pulled toward what young players of his size,
age and draft slot do, not toward every guard in the league: the design's "rookie prior
shrunk toward his rookie season in proportion to minutes played". The marginal of x is
negative binomial; beta and k are fitted per stat by maximum marginal likelihood on the
player-seasons before the cut-off (empirical Bayes on a hierarchical model: the
hyperparameters are point estimates, not integrated over). The posterior mean for a
player with weighted counts X over weighted minutes M is

    E[lambda | X] = (X + k * mu) / (M + k)

which is the same form as the bucket prior, with a player-specific mu.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

K_BOUNDS = (10.0, 5000.0)  # prior strength in minutes, as the bucket prior's cap
RIDGE = 1e-2  # small penalty on the prior's slopes, for stability
AGE_CENTRE = 26.0
PRIORS = ("bucket", "hierarchical", "hierarchical_scaled")


def design(df: pl.DataFrame) -> np.ndarray:
    """Prior covariates: intercept, wing, big, unknown height, age, age^2, experience
    (capped at 6 seasons), first and second season flags, log draft pick."""
    age = df["age"].cast(pl.Float64).fill_null(AGE_CENTRE).to_numpy() - AGE_CENTRE
    exp = df["experience"].cast(pl.Float64).fill_null(4.0).clip(0.0, 6.0).to_numpy()
    hb = df["hbucket"].fill_null("unk").to_numpy()
    pick = np.log(df["draft_pick"].cast(pl.Float64).fill_null(75.0).clip(1.0, 75.0).to_numpy())
    return np.column_stack([
        np.ones(df.height), hb == "wing", hb == "big", hb == "unk", age / 5.0, (age / 5.0) ** 2,
        exp / 6.0, exp <= 1.0, exp == 2.0, (pick - np.log(30.0)) / 1.5,
    ]).astype(float)


def _negloglik(theta: np.ndarray, X: np.ndarray, x: np.ndarray, m: np.ndarray) -> tuple[float, np.ndarray]:
    beta, logk = theta[:-1], theta[-1]
    k = np.exp(logk)
    mu = np.exp(np.clip(X @ beta, -30, 5))
    a = k * mu
    ll = gammaln(x + a) - gammaln(a) - gammaln(x + 1) + a * np.log(k / (k + m)) + x * np.log(m / (k + m))
    dll_da = digamma(x + a) - digamma(a) + np.log(k / (k + m))
    g_beta = X.T @ (dll_da * a)
    # d/dlogk: through a = k * mu, and directly through the two log terms.
    direct = a * (1.0 / k - 1.0 / (k + m)) - x / (k + m)
    g_logk = float(np.sum(k * (dll_da * mu + direct)))
    pen = 0.5 * RIDGE * float(beta[1:] @ beta[1:])
    g_pen = np.concatenate([[0.0], RIDGE * beta[1:]])
    return -float(ll.sum()) + pen, -np.concatenate([g_beta, [g_logk]]) + np.concatenate([g_pen, [0.0]])


def fit_prior(X: np.ndarray, counts: np.ndarray, minutes: np.ndarray) -> tuple[np.ndarray, float]:
    """(beta, k) by maximum marginal likelihood (negative binomial) over player-seasons."""
    ok = minutes >= 1.0
    X, x, m = X[ok], counts[ok].astype(float), minutes[ok].astype(float)
    mu0 = max(x.sum() / max(m.sum(), 1.0), 1e-6)
    theta0 = np.zeros(X.shape[1] + 1)
    theta0[0], theta0[-1] = np.log(mu0), np.log(500.0)
    bounds = [(None, None)] * X.shape[1] + [tuple(np.log(K_BOUNDS))]
    res = minimize(_negloglik, theta0, args=(X, x, m), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})
    return res.x[:-1], float(np.exp(res.x[-1]))


def hierarchical_rates(rows: pl.DataFrame, hist: pl.DataFrame, components: list[str], target_season: int,
                       scaled: bool = False) -> list[pl.Series]:
    """Posterior mean per-minute rates for the feature rows.

    ``hist``: one row per player-season before the cut-off with player_id, season, min,
    the component counts, hbucket, age (that season), experience (that season) and
    draft_pick. ``rows``: the feature rows with min_w, the ``{c}_wsum`` weighted counts,
    min_1..3 and the covariates as of last season (age, experience at ``target_season - 1``).

    The strength k is fitted on single seasons, while the evidence is 5/3/2-weighted, which
    counts a minute up to five times. ``scaled`` puts k on the same scale (k times the
    player's average weight per minute), so that one 1,000-minute season is shrunk as the
    fitted prior says; unscaled is the bucket prior's convention.
    """
    Xh = design(hist)
    Xr = design(rows.with_columns((pl.col("age") - 1).alias("age"), (pl.col("experience") - 1).alias("experience")))
    M = rows["min_w"].fill_null(0.0).to_numpy()
    raw = sum(rows[f"min_{l}"].fill_null(0.0).to_numpy() for l in (1, 2, 3))
    w_bar = np.where(raw > 0, M / np.maximum(raw, 1e-9), 1.0) if scaled else np.ones_like(M)
    out = []
    for c in components:
        beta, k = fit_prior(Xh, hist[c].to_numpy().astype(float), hist["min"].to_numpy().astype(float))
        mu = np.exp(np.clip(Xr @ beta, -30, 5))
        X = rows[f"{c}_wsum"].fill_null(0.0).to_numpy()
        kk = k * w_bar
        out.append(pl.Series(f"rate_{c}", (X + kk * mu) / (M + kk)))
    return out
