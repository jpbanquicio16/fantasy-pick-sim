import numpy as np
import pytest

from draftai.draft.order import pick_owners, pick_order, picks_for, windows
from draftai.draft.state import DraftState
from draftai.draft.survival import fit_sigma, survival


def test_snake_slot_two():
    owners = pick_owners(12, 13)
    assert picks_for(owners, 2)[:7] == [2, 23, 26, 47, 50, 71, 74]
    assert windows(picks_for(owners, 2))[:4] == [20, 2, 20, 2]


def test_third_round_reversal_slot_two():
    owners = pick_owners(12, 13, reversal_round=3)
    assert picks_for(owners, 2)[:8] == [2, 23, 35, 38, 59, 62, 83, 86]
    assert windows(picks_for(owners, 2))[:6] == [20, 11, 2, 20, 2, 20]
    # Round 3 runs the same direction as round 2.
    order = pick_order(12, 4, reversal_round=3)
    assert [s for r, s in order if r == 3] == [s for r, s in order if r == 2]


def test_linear_and_traded_picks():
    assert picks_for(pick_owners(12, 3, "linear"), 2) == [2, 14, 26]
    owners = pick_owners(12, 3, traded=[(2, 5, 2)])  # slot 2 owns slot 5's round-2 pick
    assert picks_for(owners, 2) == [2, 20, 23, 26]
    assert len(picks_for(owners, 5)) == 2


def test_state_progress_and_rosters():
    s = DraftState.new(4, 3, 2)
    assert s.my_picks() == [2, 7, 10]
    s.add_pick("a")
    assert s.my_turn and s.picks_until_mine() == 0
    s.add_pick("b")
    assert s.roster() == ["b"] and s.version == 2
    assert s.next_pick_after(2) == 7
    with pytest.raises(ValueError):
        s.add_pick("a")
    s.undo()
    assert s.version == 1 and s.roster() == []


def test_survival_window_zero_is_one_and_no_next_pick_is_zero():
    adp = np.array([5.0, 30.0, 80.0])
    assert np.allclose(survival(adp, 26, 26, 1.5, 0.12), 1.0)
    assert np.allclose(survival(adp, 26, None, 1.5, 0.12), 0.0)


def test_survival_conditions_on_having_survived():
    """A player who has fallen 10 picks past his ADP is not 'certainly gone'."""
    adp = np.array([16.0])
    cond = survival(adp, 26, 28, 1.5, 0.12)[0]
    from scipy.stats import norm
    sig = 1.5 + 0.12 * 16
    uncond = norm.sf((28 - 0.5 - 16) / sig)
    assert cond > 5 * uncond
    assert 0 < cond < 1


def test_survival_decreases_with_window_and_increases_with_adp():
    adp = np.array([30.0, 40.0, 60.0])
    s1, s2 = survival(adp, 26, 30, 1.5, 0.12), survival(adp, 26, 47, 1.5, 0.12)
    assert np.all(s2 <= s1)
    assert s2[0] < s2[1] < s2[2]


def test_fit_sigma_recovers_spread():
    rng = np.random.default_rng(0)
    adp = rng.uniform(1, 150, 20000)
    picks = adp + rng.normal(0, 2.0 + 0.1 * adp)
    a, b = fit_sigma(adp, picks)
    assert a == pytest.approx(2.0, abs=0.3) and b == pytest.approx(0.1, abs=0.01)
