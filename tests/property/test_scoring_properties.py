from hypothesis import given, settings as hsettings
from hypothesis import strategies as st

from draftai.scoring import ScoringSettings, score_line

DEFAULT = ScoringSettings()

box = st.fixed_dictionaries(
    {
        "pts": st.integers(0, 80),
        "reb": st.integers(0, 30),
        "ast": st.integers(0, 25),
        "stl": st.integers(0, 10),
        "blk": st.integers(0, 12),
        "tov": st.integers(0, 12),
        "fg3m": st.integers(0, 14),
        "tf": st.integers(0, 2),
        "ff": st.integers(0, 1),
    }
)


@hsettings(max_examples=400)
@given(box)
def test_one_more_rebound_never_lowers_points(b):
    assert score_line(b | {"reb": b["reb"] + 1}, DEFAULT) >= score_line(b, DEFAULT) + 1


@hsettings(max_examples=400)
@given(box)
def test_one_more_turnover_costs_exactly_one(b):
    assert score_line(b | {"tov": b["tov"] + 1}, DEFAULT) == score_line(b, DEFAULT) - 1


@hsettings(max_examples=400)
@given(box)
def test_results_are_whole_half_points(b):
    v = score_line(b, DEFAULT)
    assert (v * 2) == int(v * 2)


@hsettings(max_examples=200)
@given(box)
def test_stacking_never_pays_less(b):
    assert score_line(b, DEFAULT) >= score_line(b, ScoringSettings(bonus_stacking=False))


@hsettings(max_examples=200)
@given(box, st.sampled_from(["pts", "ast", "stl", "blk"]))
def test_monotone_in_positive_stats(b, stat):
    assert score_line(b | {stat: b[stat] + 1}, DEFAULT) > score_line(b, DEFAULT)
