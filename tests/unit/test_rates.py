"""Hierarchical per-minute rates (V5): the gamma-Poisson prior and its posterior."""

import numpy as np
import polars as pl
from scipy.optimize import check_grad

from draftai.features.rates import _negloglik, design, fit_prior, hierarchical_rates


def _people(n: int, rng) -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": np.arange(n), "season": np.full(n, 2020),
        "hbucket": rng.choice(["guard", "wing", "big"], n), "age": rng.integers(19, 35, n).astype(float),
        "experience": rng.integers(0, 12, n).astype(float), "draft_pick": rng.integers(1, 61, n).astype(float),
    })


def test_marginal_likelihood_recovers_the_prior():
    rng = np.random.default_rng(0)
    df = _people(4000, rng)
    X = design(df)
    beta = np.zeros(X.shape[1])
    beta[0], beta[2] = np.log(0.2), 0.6  # bigs produce more of this stat
    k = 400.0
    mu = np.exp(X @ beta)
    lam = rng.gamma(k * mu, 1.0 / k)
    minutes = rng.uniform(100, 2500, df.height)
    counts = rng.poisson(lam * minutes)
    b_hat, k_hat = fit_prior(X, counts, minutes)
    assert abs(b_hat[0] - beta[0]) < 0.05 and abs(b_hat[2] - beta[2]) < 0.05
    assert 250 < k_hat < 650


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    df = _people(300, rng)
    X = design(df)
    m = rng.uniform(50, 2000, df.height)
    x = rng.poisson(0.3 * m).astype(float)
    theta = np.concatenate([rng.normal(0, 0.1, X.shape[1]), [np.log(300.0)]])
    theta[0] = np.log(0.3)
    err = check_grad(lambda t: _negloglik(t, X, x, m)[0], lambda t: _negloglik(t, X, x, m)[1], theta)
    assert err / np.linalg.norm(_negloglik(theta, X, x, m)[1]) < 1e-4


def test_few_minutes_are_pulled_to_the_prior_and_many_are_not():
    rng = np.random.default_rng(2)
    hist = _people(2000, rng).with_columns(pl.Series("min", rng.uniform(200, 2500, 2000)))
    true_rate = rng.gamma(300 * 0.5, 1 / 300, 2000)  # players differ: prior strength about 300 minutes
    hist = hist.with_columns(pl.Series("pts", rng.poisson(true_rate * hist["min"].to_numpy())))
    rows = pl.DataFrame({"player_id": [1, 2], "hbucket": ["wing", "wing"], "age": [24.0, 24.0],
                         "experience": [3.0, 3.0], "draft_pick": [10.0, 10.0], "min_w": [100.0, 20000.0],
                         "pts_wsum": [100.0 * 1.2, 20000.0 * 1.2], "min_1": [20.0, 4000.0], "min_2": [0.0, 0.0],
                         "min_3": [0.0, 0.0]})
    r = hierarchical_rates(rows, hist, ["pts"], 2021)[0].to_numpy()
    assert abs(r[0] - 0.5) < abs(r[0] - 1.2)  # 100 weighted minutes: mostly prior
    assert abs(r[1] - 1.2) < 0.05  # 20,000: his own rate
    scaled = hierarchical_rates(rows, hist, ["pts"], 2021, scaled=True)[0].to_numpy()
    assert scaled[1] < r[1]  # scaled evidence: a 4,000-minute season counts once, not five times
