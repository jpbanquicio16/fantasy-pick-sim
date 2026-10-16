"""Head-to-head schedules, the playoff bracket and the vectorised lineup evaluator."""

import numpy as np
import pytest

from draftai.value.lineup import eligibility_matrix, greedy_lineups, optimise, slot_specificity
from draftai.value.season import (
    bracket_positions,
    group_rounds,
    playoff_champion,
    round_robin,
    season_outcomes,
    standings,
)

SLOTS = ["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL", "UTIL"]
KINDS = [["PG"], ["PG", "SG"], ["SG"], ["SG", "SF"], ["SF"], ["SF", "PF"], ["PF"], ["PF", "C"], ["C"]]


@pytest.mark.parametrize("teams", [4, 9, 12])
def test_round_robin_pairs_everyone_once_per_cycle(teams):
    s = round_robin(teams, 30, np.random.default_rng(0))
    for w in range(30):
        opp = s[w]
        has = opp >= 0
        assert (opp[opp[has]] == np.flatnonzero(has)).all()  # symmetric
        assert (opp != np.arange(teams)).all()
    n = teams + teams % 2
    pairs = {tuple(sorted((a, int(s[w][a])))) for w in range(n - 1) for a in range(teams) if s[w][a] >= 0}
    assert len(pairs) == teams * (teams - 1) // 2


def test_standings_and_ties():
    scores = np.array([[10.0, 5.0], [5.0, 5.0]])
    sched = np.array([[1, 0], [1, 0]])
    wins, pf = standings(scores, sched)
    assert wins.tolist() == [1.5, 0.5] and pf.tolist() == [15.0, 10.0]


def test_six_team_bracket_is_sleepers():
    assert bracket_positions(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    seeds = np.arange(6)  # team i is seed i + 1
    r = np.zeros((6, 3))
    r[2, 0], r[3, 0] = 10, 10  # 3 beats 6, 4 beats 5
    r[0, 1], r[3, 1] = 5, 7  # 4 upsets the 1 seed; 2 v 3 tied -> better seed (2)
    r[3, 2], r[1, 2] = 9, 3  # final
    assert playoff_champion(seeds, 6, r) == 3


def test_group_rounds_two_week_final():
    po = np.arange(8.0).reshape(2, 4)
    assert group_rounds(po, 6, round_type=1).tolist() == [[0, 1, 5], [4, 5, 13]]


def test_dominant_team_makes_playoffs_and_usually_wins():
    rng = np.random.default_rng(0)
    reg = rng.normal(100, 10, (10, 20))
    reg[0] += 60
    po = rng.normal(100, 10, (10, 3))
    po[0] += 60
    out = season_outcomes(reg, po, 6, 50, rng)
    assert out.p_playoffs[0] == 1.0 and out.p_title[0] == 1.0
    assert out.p_playoffs.sum() == pytest.approx(6.0) and out.p_title.sum() == pytest.approx(1.0)


def test_greedy_lineups_close_to_exact():
    rng = np.random.default_rng(1)
    m, n, k = 300, 13, 4
    kinds = rng.integers(0, len(KINDS), (m, n))
    values = rng.gamma(4.0, 8.0, (m, n, k))
    elig = np.stack([eligibility_matrix([KINDS[i] for i in row], SLOTS) for row in kinds])
    greedy = greedy_lineups(values, elig, slot_specificity(SLOTS))
    exact = np.array([[optimise(values[i, :, w], elig[i])[0] for w in range(k)] for i in range(m)])
    assert (greedy <= exact + 1e-9).all()
    assert greedy.sum() / exact.sum() > 0.995
