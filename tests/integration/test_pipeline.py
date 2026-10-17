"""End to end on the synthetic league: features -> models -> projections -> board ->
recommendation, plus the recommender's decision rule and latency budget."""

import time
from datetime import date

import numpy as np
import polars as pl
from synthetic import CFG, S

from draftai.config import SimConfig
from draftai.draft.recommender import RosterEvaluator, expected_best, recommend
from draftai.draft.state import DraftState
from draftai.models.artifacts import load_models, save_models
from draftai.models.components import ProjectionModel
from draftai.value.board import Board
from draftai.value.replacement import replacement_levels


def test_training_reports_every_model(training):
    _, r = training
    assert set(r.validation["model"].unique()) >= {"marcel", "last_season", "components_blend", "ridge_baseline"}
    assert set(r.blends) >= {"mpg", "share", "rate_pts", "fp_direct"}
    assert r.test["season"].unique().to_list() == [2022]
    assert "gate_v1_projection_passed" in r.summary


def test_no_test_season_rows_in_training_folds(training):
    ts, _ = training
    from draftai.models.train import run_fold

    fold = run_fold(ts, 2021, CFG)
    assert fold.rows["season"].unique().to_list() == [2021]


def test_fixed_seed_reproduces_predictions(training):
    ts, _ = training
    a = ProjectionModel(CFG.lgbm, CFG.ridge_alpha, 4.0, 7).fit(ts).predict(ts, S)
    b = ProjectionModel(CFG.lgbm, CFG.ridge_alpha, 4.0, 7).fit(ts).predict(ts, S)
    assert a.equals(b)


def test_monotone_constraint_on_minutes(training):
    ts, r = training
    m = r.models_final.projection.models["mpg"]
    row = ts.filter(pl.col("mpg_1").is_not_null()).head(1)
    lo = m.predict_parts(row.with_columns(pl.lit(10.0).alias("mpg_1")))[0]
    hi = m.predict_parts(row.with_columns(pl.lit(36.0).alias("mpg_1")))[0]
    assert hi[0] >= lo[0]


def test_artifacts_round_trip(training, tmp_path):
    ts, r = training
    save_models(r.models_final, tmp_path, CFG, date(2023, 10, 1))
    loaded = load_models(tmp_path)
    a = r.models_final.projection.predict(ts, S)["fp_pg_hat"].to_numpy()
    b = loaded.projection.predict(ts, S)["fp_pg_hat"].to_numpy()
    assert np.allclose(a, b)
    point = a
    assert np.allclose(r.models_final.fp_q.predict(ts, point), loaded.fp_q.predict(ts, point))


def test_training_chooses_and_saves_the_minutes_budget(training, tmp_path):
    from draftai.models.budget import STRENGTH_GRID

    _, r = training
    assert r.budget_strength in STRENGTH_GRID and r.models_final.minutes_budget == r.budget_strength
    assert set(r.budget_grid["strength"].unique()) == set(STRENGTH_GRID)
    assert {"components_blend", "components_no_budget"} <= set(r.validation["model"].unique())
    assert {"all", "changed teams", "returning from a long absence"} <= set(r.cohorts["cohort"].unique())
    loaded = r.models_final
    loaded.minutes_budget = 0.5
    save_models(loaded, tmp_path, CFG, date(2023, 10, 1))
    assert load_models(tmp_path).minutes_budget == 0.5
    loaded.minutes_budget = r.budget_strength


def test_board_carries_role_and_budget_columns(board):
    p = board.players
    assert {"mpg_model", "budget_delta", "team_minutes", "opp_pos_mpg", "new_team"} <= set(p.columns)
    assert np.allclose(p["mpg_hat"] - p["mpg_model"], p["budget_delta"])
    vets = p.filter(~pl.col("rookie"))
    assert vets["opp_pos_mpg"].null_count() == 0


def test_quantiles_ordered_and_centred(board):
    p = board.players.filter(~pl.col("rookie"))
    assert (p["fp_p10"] <= p["fp_p50"]).all() and (p["fp_p50"] <= p["fp_p90"]).all()
    assert (p["share_p10"] <= p["share_p90"]).all()


def test_board_contents(board, tmp_path):
    p = board.players
    assert p.height > 100 and board.weekly.shape == (p.height, len(board.weeks))
    assert (p["lockin_3"] >= p["fp_pg"] - 1e-9).all()  # three games are worth at least one
    assert (p["season_value"] >= 0).all()
    board.save(tmp_path)
    again = Board.load(tmp_path)
    assert np.allclose(again.weekly, board.weekly) and again.players.equals(board.players)


def test_recommendation_and_latency(board):
    state = DraftState.new(8, 13, 2)
    order = board.players.sort("adp")["sleeper_id"].to_list()
    ev = RosterEvaluator(board)
    while state.current_pick < 26:
        nxt = next(s for s in order if s not in state.drafted)
        state.add_pick(nxt)
    t = time.time()
    rec = recommend(board, state, SimConfig(), ev)
    assert time.time() - t < 5.0
    assert rec.best is not None and rec.best.sleeper_id not in state.drafted
    # Not on the clock at 26 (8 teams, slot 2 picks 2, 15, 18, 31, ...): plan the next own pick.
    assert rec.pick == state.my_upcoming()[0] == 31 and rec.version == 25
    assert all(0.0 <= c.reach <= 1.0 for c in rec.candidates)
    scores = [c.score for c in rec.candidates]
    assert scores == sorted(scores, reverse=True)


def test_full_draft_runs_to_the_last_pick(board):
    state = DraftState.new(8, 13, 8)
    order = board.players.sort("adp")["sleeper_id"].to_list()
    ev = RosterEvaluator(board)
    sim = SimConfig(candidate_pool=20, shortlist=4)
    while not state.complete:
        if state.my_turn:
            rec = recommend(board, state, sim, ev)
            if state.current_pick == state.total_picks:
                assert rec.next_pick is None and all(c.e_next == 0 for c in rec.candidates)
            state.add_pick(rec.best.sleeper_id)
        else:
            state.add_pick(next(s for s in order if s not in state.drafted))
    assert len(state.roster()) == 13


def test_now_or_later_rule():
    """With two candidates the score ordering equals (1 - s_A)(d_A - F) vs (1 - s_B)(d_B - F)."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        dA, dB, F = rng.uniform(5, 15), rng.uniform(5, 15), rng.uniform(0, 5)
        sA, sB = rng.uniform(0, 1, 2)
        score_A = dA + expected_best([dB], [sB], F)
        score_B = dB + expected_best([dA], [sA], F)
        rule = (1 - sA) * (dA - F) - (1 - sB) * (dB - F)
        assert np.sign(score_A - score_B) == np.sign(rule) or abs(rule) < 1e-9


def test_design_example_take_a_now():
    """Section 15: B is better, but A is likely gone; take A at 26, expect B at 47."""
    F = 0.0
    dA, sA, dB, sB = 6.0, 0.20, 7.0, 0.85
    assert dA + expected_best([dB], [sB], F) > dB + expected_best([dA], [sA], F)


def test_on_the_clock_plans_the_current_pick(board):
    state = DraftState.new(8, 13, 2)
    state.add_pick(board.players.sort("adp")["sleeper_id"][0])
    rec = recommend(board, state, SimConfig(candidate_pool=15, shortlist=4))
    assert state.my_turn and rec.pick == 2 and rec.next_pick == 15
    assert all(c.reach == 1.0 for c in rec.candidates)


def test_stale_recommendation_detected(board):
    state = DraftState.new(8, 13, 2)
    rec = recommend(board, state, SimConfig(candidate_pool=10, shortlist=3))
    state.add_pick(rec.candidates[0].sleeper_id)
    assert rec.version != state.version  # caller discards it


def test_replacement_levels_need_eligible_players():
    weekly = np.ones((3, 2))
    lv = replacement_levels(weekly, np.ones(2), [["C"], ["C"], ["C"]], ["PG", "C"], teams=1)
    assert lv.player_index["PG"] == -1 and lv.season_value["PG"] == 0.0
