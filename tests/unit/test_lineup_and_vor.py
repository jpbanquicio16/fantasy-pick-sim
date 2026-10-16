import itertools

import numpy as np

from draftai.value.lineup import eligibility_matrix, eligible, optimise, starting_slots
from draftai.value.replacement import replacement_levels, vor

SLOTS = ["PG", "SG", "G", "SF", "PF", "F", "C", "C", "UTIL", "UTIL"]
POS = [["PG"], ["SG"], ["PG", "SG"], ["SF"], ["PF"], ["SF", "PF"], ["C"], ["C", "PF"], ["SG", "SF"]]


def brute_force(values, elig):
    n, s = elig.shape
    best = 0.0
    for assign in itertools.product(range(-1, n), repeat=s):
        used = [a for a in assign if a >= 0]
        if len(used) != len(set(used)):
            continue
        if any(a >= 0 and not elig[a, j] for j, a in enumerate(assign)):
            continue
        best = max(best, sum(max(values[a], 0) for a in used))
    return best


def test_assignment_matches_brute_force():
    rng = np.random.default_rng(0)
    slots = ["PG", "G", "C", "UTIL"]
    for _ in range(40):
        n = int(rng.integers(1, 6))
        pos = [POS[i] for i in rng.integers(0, len(POS), n)]
        vals = rng.normal(20, 10, n)
        e = eligibility_matrix(pos, slots)
        total, pairs = optimise(vals, e)
        assert abs(total - brute_force(vals, e)) < 1e-9
        assert all(e[r, c] for r, c in pairs)  # eligibility never violated
        assert len({r for r, _ in pairs}) == len(pairs)


def test_eligibility_rules():
    assert eligible(["PG"], "G") and eligible(["SG"], "UTIL") and not eligible(["C"], "F")
    assert eligible(["PF"], "F") and not eligible(["PG"], "SG")
    assert starting_slots(SLOTS + ["BN", "BN", "IL"]) == SLOTS


def test_multi_eligible_player_only_util_open():
    pos = [["PG"], ["SG"], ["PG", "SG"]]
    e = eligibility_matrix(pos, ["PG", "SG", "UTIL"])
    total, pairs = optimise(np.array([10.0, 10.0, 10.0]), e)
    assert total == 30.0


def test_replacement_and_vor():
    rng = np.random.default_rng(1)
    n, weeks = 120, 5
    pos = [POS[i % len(POS)] for i in range(n)]
    weekly = np.abs(rng.normal(25, 6, (n, weeks)))
    w = np.ones(weeks)
    levels = replacement_levels(weekly, w, pos, SLOTS, teams=4)
    assert set(levels.slot_types) == set(SLOTS)
    assert len(levels.starters) == 4 * len(SLOTS)
    v = vor(weekly, w, pos, levels)
    season = weekly @ w
    # VOR preserves order within a position and the best players are positive.
    best = int(np.argmax(season))
    assert v[best] > 0
    # A flexible player is measured against his cheapest eligible slot.
    p = pos.index(["SF", "PF"])
    s = levels.cheapest_slot(pos[p])
    assert levels.season_value[s] == min(levels.season_value[x] for x in levels.slot_types if eligible(pos[p], x))
