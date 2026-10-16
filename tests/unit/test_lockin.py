import numpy as np
import pytest

from draftai.value.lockin import (
    EmpiricalScores,
    kappa,
    lockin_normal,
    lockin_value,
    prophet_kappa,
    realised_lock,
    stopping_values,
    thresholds,
)
from draftai.value.weekly import Availability, weekly_values


def test_kappa_closed_forms():
    assert kappa(1) == pytest.approx(0.0)
    assert kappa(2) == pytest.approx(0.3989, abs=1e-4)  # phi(0)
    assert kappa(3) == pytest.approx(0.630, abs=1e-3)
    assert kappa(4) == pytest.approx(0.790, abs=1e-3)


def test_prophet_gap_at_three_games():
    assert prophet_kappa(3) - kappa(3) == pytest.approx(0.217, abs=2e-3)


@pytest.mark.parametrize(
    "mu,sd,n,q,expected",
    [(40, 9, 2, 0.0, 43.6), (40, 9, 3, 0.0, 45.7), (40, 9, 3, 0.1, 44.0), (40, 9, 4, 0.0, 47.1),
     (39, 13, 2, 0.0, 44.2), (39, 13, 3, 0.0, 47.2), (39, 13, 3, 0.1, 45.3), (39, 13, 4, 0.0, 49.3)],
)
def test_design_table_veteran_vs_breakout(mu, sd, n, q, expected):
    assert lockin_normal(mu, sd, n, q) == pytest.approx(expected, abs=0.05)


def test_one_game_equals_mean_and_dnp_last_game_is_zero():
    x = EmpiricalScores(np.random.default_rng(0).normal(30, 8, 20000))
    assert lockin_value(x, [0.0]) == pytest.approx(x.mean)
    assert lockin_value(x, [1.0]) == 0.0
    # Two games, certain to miss the second: lock-or-zero, so take the first game always.
    assert lockin_value(x, [0.0, 1.0]) == pytest.approx(x.emax(0.0))


def test_stopping_value_bounded_by_hindsight_maximum():
    rng = np.random.default_rng(1)
    samples = rng.gamma(4, 7, 50000)
    x = EmpiricalScores(samples)
    for n in range(1, 6):
        draws = rng.choice(samples, size=(40000, n))
        assert lockin_value(x, [0.0] * n) <= draws.max(axis=1).mean() + 0.1
        assert lockin_value(x, [0.0] * n) >= x.mean - 1e-9


def test_empirical_matches_normal_for_normal_samples():
    x = EmpiricalScores(np.random.default_rng(2).normal(35, 10, 400000))
    for n in (2, 3, 4):
        assert lockin_value(x, [0.0] * n) == pytest.approx(lockin_normal(35, 10, n), abs=0.1)


def test_emax_and_tail():
    x = EmpiricalScores(np.array([10.0, 20.0, 30.0, 40.0]))
    assert x.emax(25.0) == pytest.approx((25 + 25 + 30 + 40) / 4)
    assert x.emax(0.0) == pytest.approx(25.0)
    assert x.tail_prob(30.0) == 0.5


def test_realised_lock_rule():
    thr = [35.0, 30.0, -np.inf]
    assert realised_lock([40.0, 10.0, 10.0], thr) == 40.0  # lock the first
    assert realised_lock([20.0, 32.0, 50.0], thr) == 32.0  # lock the second
    assert realised_lock([20.0, 25.0, 12.0], thr) == 12.0  # forced last game
    assert realised_lock([20.0, 25.0, None], thr) == 0.0  # forced last game, did not play
    assert realised_lock([None, 31.0, 50.0], thr) == 31.0


def test_thresholds_decrease_toward_the_end():
    x = EmpiricalScores(np.random.default_rng(3).normal(30, 8, 50000))
    t = thresholds(x, [0.0] * 4)
    assert t[0] > t[1] > t[2] and t[3] == -np.inf
    v = stopping_values(x, [0.0] * 4)
    assert v[0] > v[1] > v[2] > v[3]


def test_lockin_is_positively_homogeneous():
    s = np.random.default_rng(4).gamma(5, 6, 20000)
    a, b = EmpiricalScores(s), EmpiricalScores(2.5 * s)
    q = [0.05, 0.1, 0.05]
    assert lockin_value(b, q) == pytest.approx(2.5 * lockin_value(a, q))


def test_weekly_values_use_availability_and_patterns():
    x = EmpiricalScores(np.random.default_rng(5).normal(30, 8, 20000))
    av = Availability.from_share(0.8, 0.05, 1.5)
    w = weekly_values(x, av, [(), (False,), (False, True), (False, False, True, False)])
    assert w[0] == 0.0
    assert 0 < w[1] < w[2] < w[3]
    assert av.q_b2b > av.q
    assert av.a == pytest.approx(0.8 / 0.95)


def test_availability_caps_at_one():
    av = Availability.from_share(0.99, 0.05, 1.0)
    assert av.a == 1.0 and av.q == pytest.approx(0.01)
