"""V4 on the synthetic league: rollouts, offline mocks that draft themselves, and the
draft backtest's mechanics (drafting, season play, real-draft replay)."""

import time

import numpy as np
import polars as pl
import pytest
from synthetic import ROSTER

from draftai.config import BacktestConfig, LeagueConfig, SimConfig
from draftai.draft.league import LeagueContext
from draftai.draft.mock import MockDraft
from draftai.draft.opponents import OpponentModel
from draftai.draft.recommender import RosterEvaluator, recommend
from draftai.draft.rollout import RolloutConfig, RolloutEngine
from draftai.draft.state import DraftState
from draftai.value.worlds import Worlds

SIM = SimConfig(candidate_pool=20, shortlist=6)
ROOM = OpponentModel(c0=1.0, c1=-0.7, gamma=0.8)


def ctx_for(teams: int = 8, rounds: int = 13, slot: int = 2, **kw) -> LeagueContext:
    cfg = LeagueConfig(teams=teams, rounds=rounds, draft_slot=slot, roster_positions=ROSTER, **kw)
    return LeagueContext.offline(cfg)


@pytest.fixture(scope="module")
def engine(board):
    return RolloutEngine(board, ROOM, RolloutConfig(worlds=200, shortlist=5, race_after=100, seed=3), SIM)


def advance_by_adp(board, state: DraftState, until: int) -> None:
    order = board.players.sort("adp")["sleeper_id"].to_list()
    while state.current_pick < until:
        state.add_pick(next(s for s in order if s not in state.drafted))


def test_worlds_average_to_the_board(board):
    w = Worlds.draw(board, 400, seed=1)
    active = board.weights > 0
    ratio = w.expected().sum() / board.weekly[:, active].sum()
    assert 0.85 < ratio < 1.15
    assert (w.out_len >= 0).all() and (w.out_start + w.out_len <= w.weeks).all()


def test_teammates_share_minutes_in_simulated_seasons(board):
    """A teammate's absence raises a healthy player's week; averages are unchanged."""
    on, off = Worlds.draw(board, 400, seed=1), Worlds.draw(board, 400, seed=1, teammates=False)
    assert np.isclose(on.expected().sum(), off.expected().sum(), rtol=0.02)
    p = board.players
    team = p["team_now"].to_list()
    a = int(np.argmax(p["mpg_hat"].to_numpy()))
    mates = [j for j in range(p.height) if team[j] == team[a] and j != a]
    wk = np.arange(on.weeks)
    out = (wk >= on.out_start[..., None]) & (wk < (on.out_start + on.out_len)[..., None])
    gone = out[:, mates].any(axis=1) & ~out[:, a]  # (worlds, weeks): a teammate out, he plays
    full = ~out[:, mates].any(axis=1) & ~out[:, a]
    ratio = on.values(np.arange(400), np.full((400, 1), a))[:, 0] / np.maximum(
        off.values(np.arange(400), np.full((400, 1), a))[:, 0], 1e-9)
    assert ratio[gone].mean() > 1.0 > ratio[full].mean()


def test_rollout_recommends_and_is_reproducible(board, engine):
    state = DraftState.new(8, 13, 2)
    advance_by_adp(board, state, 15)  # slot 2 picks 2, 15, 18, ...
    t = time.time()
    a = engine.recommend(state)
    assert time.time() - t < 5.0
    b = engine.recommend(state)
    assert [c.row for c in a.candidates] == [c.row for c in b.candidates]
    assert a.best.sleeper_id not in state.drafted and a.pick == 15 and a.next_pick == 18
    assert 0.0 <= a.confidence <= 1.0
    live = [c for c in a.candidates if not c.dropped]
    assert [c.mean for c in live] == sorted((c.mean for c in live), reverse=True)
    assert abs(sum(c.p_best for c in a.candidates) - 1.0) < 1e-6


def test_back_to_back_picks_survive_for_sure(board, engine):
    """Slot 8 of 8 picks 8 and 9 in a row: nobody picks in between."""
    state = DraftState.new(8, 13, 8)
    advance_by_adp(board, state, 8)
    res = engine.recommend(state)
    assert res.next_pick == 9
    assert res.survival and all(v == 1.0 for v in res.survival.values())


def test_planning_ahead_scores_only_worlds_where_he_is_there(board, engine):
    state = DraftState.new(8, 13, 2)
    advance_by_adp(board, state, 10)  # five picks before yours
    res = engine.recommend(state)
    assert res.pick == 15
    assert all(0.0 < c.reach <= 1.0 for c in res.candidates)


def test_single_candidate_rollout_is_the_base_policy(board):
    """Design section 26: with the base policy's own choice as the only candidate, the
    rollout's value is the base policy's value from here (same worlds, same opponents)."""
    from draftai.draft.recommender import Candidate, Recommendation
    from draftai.draft.rollout import FILL_FIRST

    eng = RolloutEngine(board, ROOM, RolloutConfig(worlds=80, shortlist=1, seed=2), SIM)
    state = DraftState.new(8, 13, 2)
    advance_by_adp(board, state, 15)
    A, C, M, NM = eng._start(state, 1)
    fills = eng.needs.fills_batch(C[:, state.my_slot])[0][eng.market.type_col]
    col = int(np.argmax(np.where(A[0], eng.policy_col + FILL_FIRST * fills, -np.inf)))
    row = int(eng.market.order[col])
    only = Candidate(row, eng.ids[row], "", [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rec = Recommendation(state.version, 15, 18, 0.0, [only], 0.0, float(board.weights.sum()))
    res = eng.recommend(state, rec)
    assert [c.row for c in res.candidates] == [row] and res.confidence is None

    n = eng.cfg.worlds
    widx = np.arange(n)
    uni = eng._uniforms(state, 0, widx)
    A, C, M, NM = eng._start(state, n)
    upcoming = state.my_upcoming()
    eng._run([(t, state.owners[t - 1]) for t in range(15, upcoming[-1] + 1)], state.my_slot, A, C, M, NM, uni)
    assert res.best.mean == pytest.approx(float(eng.season_values(widx, M).mean()))


def test_mock_room_drafts_itself_and_replays_by_seed(board):
    ctx = ctx_for()
    a, b = MockDraft(ctx, board, ROOM, seed=4), MockDraft(ctx, board, ROOM, seed=4)
    a.advance()
    b.advance()
    assert a.state.my_turn and a.state.current_pick == 2
    first = a.log_frame()["player"].to_list()
    assert first == b.log_frame()["player"].to_list() and len(first) == 1
    ev = RosterEvaluator(board)
    a.pick(recommend(board, a.state, SIM, ev).best.sleeper_id)
    after = a.advance()
    assert a.state.my_turn and a.state.current_pick == 15 and len(after) == 12
    a.undo_mine()
    assert a.state.current_pick == 2 and a.state.my_turn
    a.pick(next(s for s in board.players.sort("adp")["sleeper_id"] if s not in a.state.drafted))
    assert a.state.current_pick == 3
    by_adp = lambda st: next(s for s in board.players.sort("adp")["sleeper_id"] if s not in st.drafted)
    logs = []
    for seed in (4, 4, 5):
        m = MockDraft(ctx, board, ROOM, seed=seed)
        m.run_to_end(by_adp)
        logs.append(m.log_frame()["player"].to_list())
    assert logs[0] == logs[1] and logs[0] != logs[2]


def test_mock_runs_to_the_end_and_projects_the_league(board):
    ctx = ctx_for(teams=6, rounds=12, slot=3, playoff_teams=4, playoff_week_start=6)
    mock = MockDraft(ctx, board, ROOM, seed=1)
    ev = RosterEvaluator(board)
    mock.run_to_end(lambda st: recommend(board, st, SIM, ev).best.sleeper_id)
    assert mock.state.complete and all(len(r) == 12 for r in mock.state.rosters().values())
    assert {m.by for m in mock.log} == {"auto", "engine"}
    proj = mock.projection(ev, n_worlds=60)
    assert proj.height == 6 and "You" in proj["team"].to_list()
    # Each team's odds are rounded to 3 decimals: six of them sum to within 6 x 0.0005.
    assert proj["p_playoffs"].sum() == pytest.approx(4.0, abs=0.003)
    assert proj["p_title"].sum() == pytest.approx(1.0, abs=0.003)


# --- backtest mechanics ----------------------------------------------------------------
def test_realised_season_is_consistent(past_season):
    sd, _ = past_season
    r = sd.realised
    assert r.locked.shape == (sd.board.players.height, r.weights.size)
    assert (r.locked[~r.healthy] == 0).all()  # out all week: nothing to lock
    assert (r.locked >= 0).all() and r.playoff.any() and not r.playoff.all()
    assert r.healthy.mean() > 0.3


def test_paired_backtest_runs_and_oracle_leads(past_season):
    from draftai.backtest.league import play
    from draftai.backtest.run import Room, calibration_rows, draft, paired, rosters_of
    from draftai.backtest.strategies import KeyGreedy, build

    sd, rules = past_season
    room = Room.build(sd, rules, ROOM)
    rcfg = RolloutConfig(worlds=60, shortlist=3, race_after=40, seed=1)
    strategies = build(["adp", "vor", "greedy_survival", "rollout", "oracle"])
    for s in strategies:
        s.prepare(sd, room.needs, room.evaluator, ROOM, SIM, rcfg)
    rows = []
    calib = []
    for league in range(4):
        slot = league % rules.teams + 1
        u = np.random.default_rng([9, league]).random(rules.teams * rules.rounds + 2)
        for s in strategies:
            s.reset()
            state = draft(room, s.choose, room.new_state(slot), u)
            assert state.complete and len(set(state.picks)) == len(state.picks)
            ros = rosters_of(room, state)
            res = play(ros, sd.realised, room.elig, rules, np.random.default_rng([league]), 10, 2)
            rows.append({"season": 2022, "league": league, "strategy": s.name,
                         "p_playoffs": float(res.outcome.p_playoffs[slot - 1]),
                         "realised_per_week": float(res.regular_per_week(sd.realised.playoff)[slot - 1])})
            if s.name == "rollout":
                base = KeyGreedy("vor", "vor")
                base.prepare(sd, room.needs, room.evaluator, ROOM, SIM, rcfg)
                calib += calibration_rows(room, s.decisions, u, base)
    df = pl.DataFrame(rows)
    assert df["p_playoffs"].is_between(0, 1).all()
    means = df.group_by("strategy").agg(pl.col("realised_per_week").mean())
    best = means.sort("realised_per_week", descending=True)["strategy"][0]
    assert best == "oracle"
    d = paired(df, "oracle", "adp", "realised_per_week", reps=200)
    assert d["leagues"] == 4 and d["lo"] <= d["diff"] <= d["hi"]
    assert calib and all(0.0 <= c["confidence"] <= 1.0 for c in calib)


def test_replay_keeps_human_picks(past_season, monkeypatch):
    from draftai.backtest import replay as rp
    from draftai.backtest.run import Room, draft, rosters_of
    from draftai.backtest.strategies import build

    sd, rules = past_season
    room = Room.build(sd, rules, ROOM)
    human = draft(room, lambda st, a, c: int(np.flatnonzero(a)[0]), room.new_state(1),
                  np.random.default_rng(0).random(200))
    picks = [(i + 1, slot, sid) for i, (sid, slot) in enumerate(zip(human.picks, human.pick_slots))]
    rd = rp.RealDraft("d1", 2022, "synthetic 2022", rules, picks, user_slot=2)
    monkeypatch.setattr(rp, "season_data", lambda *a, **k: sd)
    bt = BacktestConfig(rollout_worlds=40, rollout_shortlist=2, replay_schedules=10, game_samples=500)
    df = rp.replay(rd, SIM, None, bt, None, ROOM, ["vor"])
    assert set(df["strategy"]) == {"human", "vor"} and df.filter(pl.col("strategy") == "human").height == rules.teams
    s = rp.summarise(df)
    assert s["seats"][0] == rules.teams
    # The human roster of seat 1 is exactly what that seat drafted.
    assert df.filter((pl.col("strategy") == "human") & (pl.col("seat") == 1))["first_picks"][0] == ", ".join(
        sd.board.players["name"][r] for r in rosters_of(room, human)[0][:3])
    assert build(["vor"])[0].name == "vor"
