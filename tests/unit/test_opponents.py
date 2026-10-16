"""Positional need and the conditional-logit opponent model."""

import numpy as np
import pytest

from draftai.draft.needs import NeedModel
from draftai.draft.opponents import (
    DraftRecord,
    Market,
    OpponentModel,
    calibrate,
    choose_batch,
    fit,
    probabilities,
    simulate_drafts,
    spread_by_band,
)
from draftai.draft.order import pick_owners

SLOTS = ["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL", "UTIL"]
KINDS = [["PG"], ["PG", "SG"], ["SG", "SF"], ["SF", "PF"], ["PF", "C"], ["C"]]


def pool(n: int = 300, seed: int = 0) -> tuple[list[list[str]], np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = [KINDS[i] for i in rng.integers(0, len(KINDS), n)]
    return positions, np.arange(1, n + 1, dtype=float)


def test_need_empty_full_and_crowded():
    nm = NeedModel(KINDS, ["PG", "C", "UTIL"])
    c_only = nm.types.index(("C",))
    pg_only = nm.types.index(("PG",))
    assert nm.fills(np.zeros(nm.n_types, dtype=int)).all()
    two_c = np.zeros(nm.n_types, dtype=int)
    two_c[c_only] = 2  # C and UTIL taken by centres
    f = nm.fills(two_c)
    assert not f[c_only] and f[pg_only]
    full = two_c.copy()
    full[pg_only] = 1
    assert not nm.fills(full).any()


def test_need_batch_matches_rows():
    positions, _ = pool()
    nm = NeedModel(positions, SLOTS)
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 3, size=(500, nm.n_types)) * (rng.random((500, nm.n_types)) < 0.4)
    assert (nm.fills_batch(counts) == np.stack([nm.fills(c) for c in counts])).all()


def test_zero_noise_room_follows_adp():
    positions, adp = pool()
    market = Market(adp, NeedModel(positions, SLOTS))
    strict = OpponentModel(c0=5.0, c1=0.0, gamma=0.0)
    picks = simulate_drafts(strict, market, pick_owners(6, 10), 3, seed=0)
    assert (picks == np.arange(60)).all()


def test_common_random_numbers_and_off_board_picks():
    positions, adp = pool()
    market = Market(adp, NeedModel(positions, SLOTS))
    n = 400
    avail = np.ones((n, market.size), dtype=bool)
    fills = np.ones((n, market.needs.n_types), dtype=bool)
    u = np.random.default_rng(3).random(n)
    m = OpponentModel(c0=0.0, c1=0.0, gamma=0.0, eps=0.2, reach=100)
    a = choose_batch(m, market, avail, fills, 30, u)
    assert (a == choose_batch(m, market, avail, fills, 30, u)).all()
    off = u < m.eps
    assert (a[~off] < m.window).all()  # logit picks stay in the window
    assert (a[off] < m.reach).all() and (a[off] >= m.window).any()


def test_probabilities_sum_to_one():
    positions, adp = pool()
    market = Market(adp, NeedModel(positions, SLOTS))
    cols, p = probabilities(OpponentModel(eps=0.1), market, np.ones(market.size, dtype=bool),
                            np.zeros(market.needs.n_types, dtype=int), 20)
    assert p.sum() == pytest.approx(1.0) and (p >= 0).all() and cols.size == 150


def test_fit_recovers_a_known_room():
    positions, adp = pool(400)
    market = Market(adp, NeedModel(positions, SLOTS))
    truth = OpponentModel(c0=0.0, c1=-0.5, gamma=0.8, eps=0.05)
    owners = pick_owners(10, 12)
    drafts = simulate_drafts(truth, market, owners, 40, seed=5)
    records = [DraftRecord([(t + 1, s, int(market.order[c])) for t, (s, c) in enumerate(zip(owners, d))])
               for d in drafts]
    m = fit(records, market, heldout=False)
    for t in (1, 20, 100):
        assert m.beta(t) == pytest.approx(truth.beta(t), rel=0.3)
    assert m.eps == pytest.approx(truth.eps, abs=0.03)
    small = fit(records[:4], market)
    held = small.diagnostics["heldout_loglik_per_pick"]
    assert held["full"] > held["uniform"] and held["full"] > held["no_off_board"]


def test_calibration_matches_target_spread():
    positions, adp = pool(400)
    market = Market(adp, NeedModel(positions, SLOTS))
    owners = pick_owners(12, 13)
    m = calibrate(market, owners, 1.5, 0.12, n=80)
    target, got = m.diagnostics["target_mad_early_late"], m.diagnostics["simulated_mad_early_late"]
    assert got[0] == pytest.approx(target[0], rel=0.15) and got[1] == pytest.approx(target[1], rel=0.15)
    early, late = spread_by_band(simulate_drafts(m, market, owners, 50, 9), market, 36)
    assert late > early  # drafters stray further from ADP later
