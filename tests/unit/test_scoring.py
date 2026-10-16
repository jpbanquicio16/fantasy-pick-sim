import json
import warnings

import numpy as np
import polars as pl
import pytest

from draftai.scoring import ScoringSettings, score_frame, score_line
from draftai.scoring.settings import SLEEPER_DEFAULT

DEFAULT = ScoringSettings()
EXCLUSIVE = ScoringSettings(bonus_stacking=False)


def line(**kw):
    base = {c: 0 for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")}
    return base | kw


def test_golden_sleeper_games_reproduced_exactly(fixtures_dir):
    """Release gate: every golden game matches Sleeper's own score to the half-point."""
    data = json.loads((fixtures_dir / "golden" / "sleeper_std_2025.json").read_text())
    settings = ScoringSettings.from_sleeper(data["settings"], bonus_stacking=True)
    bad = [(g["name"], g["week"], score_line(g["box"], settings), g["pts_std"])
           for g in data["games"] if score_line(g["box"], settings) != g["pts_std"]]
    assert not bad, bad[:5]
    assert len(data["games"]) > 600


def test_golden_fifty_point_games_settle_stacking(fixtures_dir):
    """Sleeper pays both the 40+ and 50+ bonus: exclusive scoring is off by exactly 2."""
    data = json.loads((fixtures_dir / "golden" / "sleeper_std_2025.json").read_text())
    fifties = [g for g in data["games"] if g["box"]["pts"] >= 50]
    assert len(fifties) >= 4
    stack = ScoringSettings.from_sleeper(data["settings"], bonus_stacking=True)
    excl = ScoringSettings.from_sleeper(data["settings"], bonus_stacking=False)
    for g in fifties:
        assert score_line(g["box"], stack) == g["pts_std"]
        assert score_line(g["box"], stack) - score_line(g["box"], excl) == 2.0


def test_golden_sleeper_dd_td_flags_match_engine(fixtures_dir):
    data = json.loads((fixtures_dir / "golden" / "sleeper_std_2025.json").read_text())
    for g in data["games"]:
        cats = sum(g["box"][c] >= 10 for c in ("pts", "reb", "ast", "stl", "blk"))
        assert (cats >= 2) == bool(g["sleeper_dd"]), g
        assert (cats >= 3) == bool(g["sleeper_td"]), g


def test_fifty_two_points_both_modes():
    g = line(pts=52)
    assert score_line(g, DEFAULT) == 26 + 4
    assert score_line(g, EXCLUSIVE) == 26 + 2


def test_forty_five_points_both_modes():
    g = line(pts=45)
    assert score_line(g, DEFAULT) == score_line(g, EXCLUSIVE) == 22.5 + 2


def test_double_double_counts_blocks():
    assert score_line(line(pts=12, blk=10), DEFAULT) == 6 + 20 + 1


def test_triple_double_stacks_to_three():
    g = line(pts=20, reb=10, ast=10)
    assert score_line(g, DEFAULT) == 10 + 10 + 10 + 1 + 2


def test_triple_double_on_steals():
    g = line(pts=10, reb=10, stl=10)
    assert score_line(g, DEFAULT) == 5 + 10 + 20 + 3


def test_quadruple_double_still_three():
    g = line(pts=10, reb=10, ast=10, blk=10)
    assert score_line(g, DEFAULT) == 5 + 10 + 10 + 20 + 3


def test_did_not_play_scores_zero():
    assert score_line(line(), DEFAULT) == 0.0


def test_made_three_is_worth_two_points():
    assert score_line(line(pts=3, fg3m=1), DEFAULT) == 2.0


def test_technical_and_flagrant():
    assert score_line(line(tf=1, ff=1), DEFAULT) == -4.0


def test_units_and_resolution():
    assert DEFAULT.units_per_point == 2
    assert ScoringSettings(weights={"reb": 1.2}).units_per_point == 10
    with pytest.raises(ValueError):
        _ = ScoringSettings(weights={"reb": 1.234}).units_per_point


def test_unknown_keys_reported_not_scored():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = ScoringSettings.from_sleeper({**SLEEPER_DEFAULT, "made_up_stat": 3.0})
    assert "made_up_stat" in s.unknown_keys and "made_up_stat" not in s.weights
    assert w


def test_misses_and_combos():
    s = ScoringSettings(weights={"fgmi": -0.5, "pts_reb_ast": 1.0})
    assert score_line({"fga": 10, "fgm": 4, "pts": 10, "reb": 5, "ast": 2}, s) == -3 + 17


def test_frame_matches_scalar():
    rng = np.random.default_rng(0)
    rows = [line(pts=int(rng.integers(0, 60)), reb=int(rng.integers(0, 20)), ast=int(rng.integers(0, 15)),
                 stl=int(rng.integers(0, 6)), blk=int(rng.integers(0, 8)), tov=int(rng.integers(0, 8)),
                 fg3m=int(rng.integers(0, 9))) for _ in range(300)]
    df = score_frame(pl.DataFrame(rows), DEFAULT)
    assert df["fp"].to_list() == [score_line(r, DEFAULT) for r in rows]
    assert (df["fp_units"] == (df["fp"] * 2).cast(pl.Int64)).all()


def test_overtime_counts_in_full():
    """No per-minute normalisation: a 53-minute overtime line scores like any other."""
    g = line(pts=30, reb=10)
    assert score_line(g | {"min": 53}, DEFAULT) == score_line(g, DEFAULT)
