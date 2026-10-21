"""The learning-to-rank experiment (V5) on the synthetic league: queries, labels, rankers
trained leaving one season out, offline metrics and the backtest strategies."""

import numpy as np
import polars as pl
import pytest

from draftai.config import LTRConfig, Paths, SimConfig
from draftai.draft.opponents import OpponentModel
from draftai.draft.ranker import FEATURES, grades, ndcg_at
from draftai.draft.rollout import RolloutConfig, RolloutEngine

SIM = SimConfig(candidate_pool=20, shortlist=6)
ROOM = OpponentModel(c0=1.0, c1=-0.7, gamma=0.8)
CFG = LTRConfig(leagues=2, docs=6, worlds=30, epsilon=0.3, sim_rounds=20, real_rounds=10)


def test_grades_and_ndcg():
    g = grades(np.array([1.0, 3.0, 2.0, 5.0, 5.0, 5.0]), np.array([1, 1, 1, 2, 2, 2]), levels=10)
    assert g.tolist() == [0, 10, 5, 0, 0, 0]  # a flat query: one grade for all
    gains = np.array([0.0, 10.0, 5.0])
    assert ndcg_at(np.array([0.0, 2.0, 1.0]), gains) == pytest.approx(1.0)
    assert ndcg_at(np.array([2.0, 0.0, 1.0]), gains) < 1.0


@pytest.fixture(scope="module")
def queries(past_season):
    from draftai.backtest.ltr import collect_league
    from draftai.backtest.run import Room
    from draftai.backtest.strategies import KeyGreedy

    sd, rules = past_season
    room = Room.build(sd, rules, ROOM)
    rcfg = RolloutConfig(worlds=CFG.worlds, shortlist=CFG.docs, race_after=CFG.worlds, seed=1)
    engine = RolloutEngine(sd.board, ROOM, rcfg, SIM, evaluator=room.evaluator)
    base = KeyGreedy("vor", "vor")
    base.prepare(sd, room.needs, room.evaluator, ROOM, SIM, rcfg)
    rng = np.random.default_rng(0)
    rows = []
    for league in range(CFG.leagues):
        rows += collect_league(room, engine, base, SIM, CFG, 2022, league, rng)
    q = pl.DataFrame(rows)
    # A second "season" (the same draws relabelled) so that leave-one-season-out has folds.
    return pl.concat([q, q.with_columns(pl.lit(2021, dtype=q["season"].dtype).alias("season"),
                                        (pl.col("query") - 10_000_000).alias("query"))])


def test_every_pick_is_a_labelled_query(queries, past_season):
    _, rules = past_season
    q = queries.filter(pl.col("season") == 2022)
    assert q["query"].n_unique() == CFG.leagues * rules.rounds
    per = q.group_by("query").agg(pl.len(), pl.col("chosen").sum())
    assert (per["len"] <= CFG.docs).all() and (per["chosen"] == 1).all()
    assert q["sim_label"].is_finite().all() and q["real_label"].is_finite().all()
    assert set(FEATURES) <= set(q.columns)
    # The realised label is the roster's real season: the same in every world, so it can
    # differ from the simulator's expectation, but both are FP per week of a full roster.
    assert (q["real_label"] > 0).any() and (q["sim_label"] > 0).all()


def test_rankers_train_leaving_one_season_out(queries, tmp_path):
    from draftai.backtest import ltr
    from draftai.draft.ranker import Ranker

    paths = Paths(root=tmp_path)
    ltr.train(queries, CFG, paths, {2021: "k21", 2022: "k22"}, log=lambda *_: None)
    for s in (2021, 2022):
        for st in ltr.STAGES:
            r = Ranker.load(ltr.ranker_path(paths, s, st))
            assert r.meta["trained_on"] == [2022 if s == 2021 else 2021] and r.meta["stage"] == st
    fine = Ranker.load(ltr.ranker_path(paths, 2022, "real")).booster
    assert fine.num_trees() == CFG.sim_rounds + CFG.real_rounds  # continued from the distilled ranker
    off = ltr.offline(queries, paths, CFG.levels)
    assert set(off["method"].unique().to_list()) == set(ltr.METHODS)
    assert off.filter(pl.col("method") == "simulator")["top1_is_simulators"].all()
    assert (off["regret_real"] >= -1e-9).all() and off["ndcg5_real"].is_between(0, 1 + 1e-9).all()
    report = ltr.write_report(queries, off, paths, None, CFG)
    assert "Learning-to-rank" in report.read_text()


def test_ranked_strategies_draft_in_the_backtest(queries, past_season, tmp_path):
    from draftai.backtest import ltr
    from draftai.backtest.run import Room, draft
    from draftai.backtest.strategies import build

    sd, rules = past_season
    paths = Paths(root=tmp_path)
    ltr.train(queries, CFG, paths, {}, log=lambda *_: None)
    room = Room.build(sd, rules, ROOM)
    for s in build(["ltr", "ltr_sim"], paths):
        s.prepare(sd, room.needs, room.evaluator, ROOM, SIM, RolloutConfig())
        state = draft(room, s.choose, room.new_state(1), np.random.default_rng(3).random(rules.teams * rules.rounds + 2))
        assert state.complete and len(set(state.picks)) == len(state.picks)
    with pytest.raises(FileNotFoundError):
        build(["ltr"], Paths(root=tmp_path / "empty"))[0].prepare(sd, room.needs, room.evaluator, ROOM, SIM,
                                                                  RolloutConfig())


def test_query_ids_are_unique_in_long_drafts():
    from draftai.backtest.ltr import query_id

    ids = {query_id(s, lg, p) for s in (2024, 2025) for lg in range(96) for p in range(1, 241)}
    assert len(ids) == 2 * 96 * 240
