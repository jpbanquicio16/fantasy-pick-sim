"""Command-line entry point: ``draftai <command>``.

    draftai ingest              fetch + normalise + validate NBA data and Sleeper players
    draftai league              show the league settings the engine will use
    draftai train               features, rolling-origin evaluation, test season, final models
    draftai project             project the draft season and build the board
    draftai lockin-experiment   the Lock-In headline experiment on the test season
    draftai adp                 show or refresh ADP (search rank, CSV, or your Sleeper drafts)
    draftai recommend           recommendation for a live draft, or an offline position
    draftai mock                offline mock draft; the other teams draft themselves
    draftai opponents           fit the opponent model on your real Sleeper drafts
    draftai backtest            paired draft backtest on past seasons, plus real-draft replays
    draftai app                 launch the Streamlit draft assistant
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from draftai.config import Paths, load_league_config, load_sim_config, load_train_config


def _ctx(args):
    from draftai.draft.league import LeagueContext

    cfg = load_league_config(Path(args.config) if getattr(args, "config", None) else None)
    if getattr(args, "slot", None):
        cfg.draft_slot = args.slot
    if getattr(args, "league_id", None) or getattr(args, "draft_id", None) or cfg.league_id or cfg.draft_id:
        return LeagueContext.from_sleeper(cfg, league_id=getattr(args, "league_id", None),
                                          draft_id=getattr(args, "draft_id", None),
                                          username=getattr(args, "username", None))
    return LeagueContext.offline(cfg)


def cmd_ingest(args) -> None:
    from draftai import pipeline

    paths = Paths().ensure()
    if args.sleeper_only:
        pipeline.refresh_sleeper(paths)
    else:
        pipeline.ingest(load_train_config(), load_league_config(), paths)


def cmd_league(args) -> None:
    for line in _ctx(args).describe():
        print(line)


def cmd_train(args) -> None:
    from draftai import pipeline

    ctx = _ctx(args)
    r = pipeline.train(load_train_config(), ctx.scoring, Paths().ensure())
    s = r.summary
    print(f"\nV1 gate (beat Marcel in >=3 of 4 validation seasons, CI excludes zero): "
          f"{'PASS' if s['gate_v1_projection_passed'] else 'FAIL'} ({s['validation_seasons_better_with_ci_excluding_zero']}/4)")
    g = s["test_gain_vs_marcel"]
    print(f"Test season: FP/G MAE {s['test_fpg_mae_top200']:.2f} (top 200), gain vs Marcel {g[0]:+.2f} "
          f"[{g[1]:+.2f}, {g[2]:+.2f}]; 80% interval coverage {s['test_fpg_cov80_top200']:.0%}")


def cmd_project(args) -> None:
    from draftai import pipeline

    ctx = _ctx(args)
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    board = pipeline.project(ctx, load_sim_config(), Paths().ensure(), as_of=as_of,
                             adp_csv=Path(args.adp_csv) if args.adp_csv else None)
    import polars as pl

    with pl.Config(tbl_rows=args.top, tbl_cols=12, tbl_width_chars=180):
        print(board.players.sort("vor", descending=True).with_row_index("rank", 1).select(
            "rank", "name", "team_now", "positions", "fp_pg", "lockin_3", "share_hat", "vor_per_week", "adp"
        ).head(args.top))


def cmd_lockin(args) -> None:
    from draftai.models.artifacts import load_models
    from draftai.normalise.tables import load_table
    from draftai.value import experiment

    paths = Paths()
    ctx = _ctx(args)
    pg, tg, bio, dr = (load_table(t, paths) for t in ("player_games", "team_games", "player_bio", "draft_class"))
    season = args.season or load_train_config().test_season
    res = experiment.run(pg, tg, bio, dr, load_models(paths.artifacts / "models" / "test"), ctx.scoring, season)
    print(experiment.write(res, paths.artifacts / "reports").read_text())


def cmd_adp(args) -> None:
    import polars as pl

    from draftai import pipeline
    from draftai.draft import adp as adp_mod
    from draftai.draft.survival import fit_sigma
    from draftai.ingestion.sleeper import SleeperClient

    paths = Paths()
    ctx = _ctx(args)
    board = pipeline.load_board(paths, ctx.season)
    if args.csv:
        adp = adp_mod.from_csv(Path(args.csv), board.players)
    else:
        adp = adp_mod.from_search_rank(board.players)
    if args.username:
        client = SleeperClient()
        user = client.user(args.username)
        drafts = [d for d in client.user_drafts(user["user_id"], ctx.season) if d.get("status") == "complete"]
        picks = [[(p.pick_no, p.player_id) for p in client.draft_picks(d["draft_id"], ttl=None)] for d in drafts]
        prior = adp
        adp = adp_mod.from_drafts(picks, prior, ctx.teams * ctx.rounds)
        print(f"aggregated {len(picks)} completed Sleeper drafts")
        if picks:
            pa = dict(zip(prior["sleeper_id"].to_list(), prior["adp"].to_list()))
            pairs = [(pa[pid], no) for d in picks for no, pid in d if pid in pa]
            a, b = fit_sigma([x for x, _ in pairs], [y for _, y in pairs])
            print(f"fitted ADP spread: sigma = {a:.2f} + {b:.3f} * ADP  (set adp_sigma_a/b in configs/sim.yaml)")
    board.players = adp_mod.attach(board.players, adp)
    board.save(pipeline.board_dir(paths, ctx.season))
    with pl.Config(tbl_rows=args.top):
        print(board.players.sort("adp").select("adp", "name", "team_now", "positions", "adp_source").head(args.top))


def _rollout_engine(ctx, board, sim, ev, paths):
    from draftai import pipeline
    from draftai.draft.rollout import RolloutConfig, RolloutEngine

    model = pipeline.live_opponents(ctx, board, sim, paths)
    return RolloutEngine(board, model, RolloutConfig.from_sim(sim), sim, evaluator=ev)


def cmd_recommend(args) -> None:
    from draftai import pipeline
    from draftai.draft.explain import card, render
    from draftai.draft.recommender import RosterEvaluator, recommend
    from draftai.ingestion.sleeper import SleeperClient
    from draftai.live.poller import LiveDraft

    paths = Paths()
    ctx = _ctx(args)
    sim = load_sim_config()
    board = pipeline.load_board(paths, ctx.season)
    ev = RosterEvaluator(board)
    names = dict(zip(board.players["sleeper_id"].to_list(), board.players["name"].to_list()))
    engine = _rollout_engine(ctx, board, sim, ev, paths) if args.rollout else None

    def show(state):
        if state.complete:
            print("Draft complete. Your roster:", ", ".join(names.get(p, p) for p in state.roster()))
            return
        rec = recommend(board, state, sim, ev)
        until = state.picks_until_mine()
        print(f"\n--- pick {state.current_pick} | your next pick {state.my_upcoming()[:1]} | {until} picks away ---")
        ro = engine.recommend(state, rec) if engine is not None else None
        print(render(card(rec, board.players, ctx.scoring, rollout=ro)))
        if ro is not None:
            print(ro.table().head(args.top).select("rank", "player", "pos", "value_per_week", "gap_per_week", "p_best",
                                                   "reach", "survival_sim", "dropped"))
            print(f"({ro.worlds} worlds, {ro.elapsed:.1f} s)")
        else:
            print(rec.table().head(args.top).select("rank", "player", "pos", "adp", "reach", "survival", "delta_per_week"))

    if ctx.draft_id:
        live = LiveDraft(ctx, SleeperClient())
        if args.follow:
            live.follow(show, interval=sim.poll_seconds)
        else:
            live.poll()
            show(live.state)
        return
    # Offline: replay picks given on the command line (Sleeper ids or names).
    state = ctx.new_state()
    by_name = {n.lower(): s for s, n in names.items()}
    for token in args.picks or []:
        state.add_pick(by_name.get(token.lower(), token))
    show(state)


def cmd_mock(args) -> None:
    """Offline mock in the terminal: the other teams draft themselves with the opponent model."""
    import polars as pl

    from draftai import pipeline
    from draftai.draft.explain import card, render
    from draftai.draft.mock import MockDraft
    from draftai.draft.recommender import RosterEvaluator, recommend

    paths = Paths()
    ctx = _ctx(args)
    if args.slot:
        ctx.my_slot = args.slot
    sim = load_sim_config()
    board = pipeline.load_board(paths, ctx.season)
    ev = RosterEvaluator(board)
    model = pipeline.live_opponents(ctx, board, sim, paths)
    engine = _rollout_engine(ctx, board, sim, ev, paths) if args.rollout else None
    mock = MockDraft(ctx, board, model, **({"seed": args.seed} if args.seed is not None else {}))
    print(f"Mock draft: {ctx.teams} teams, {ctx.rounds} rounds, {ctx.draft_type}"
          + (f" (reversal from round {ctx.reversal_round})" if ctx.reversal_round else "")
          + f"; you pick from slot {ctx.my_slot}; seed {mock.seed}. {model.describe()}")
    by_name = {n.lower(): s for s, n in zip(board.players["sleeper_id"].to_list(), board.players["name"].to_list())}

    def best(state) -> str:
        rec = recommend(board, state, sim, ev)
        if engine is not None:
            return engine.recommend(state, rec).best.sleeper_id
        return rec.best.sleeper_id

    auto = args.auto
    while not mock.state.complete:
        for mp in mock.advance():
            if not args.quiet:
                print(f"  {mp.pick:>3}. {mock.team_name(mp.slot):<10} {mp.name} ({'/'.join(mp.positions)}, ADP {mp.adp:.0f})")
        if mock.state.complete:
            break
        if auto:
            mp = mock.pick(best(mock.state), by="engine")
            print(f"  {mp.pick:>3}. {'You':<10} {mp.name} ({'/'.join(mp.positions)}, ADP {mp.adp:.0f})  <- engine")
            continue
        rec = recommend(board, mock.state, sim, ev)
        ro = engine.recommend(mock.state, rec) if engine is not None else None
        print()
        print(render(card(rec, board.players, ctx.scoring, rollout=ro)))
        top = (ro.table().select("rank", "player", "pos", "gap_per_week", "p_best") if ro is not None
               else rec.table().select("rank", "player", "pos", "adp", "survival", "delta_per_week"))
        print(top.head(args.top))
        choice = input("Your pick [Enter = #1, number = that rank, name, 'auto' = engine drafts the rest, 'q' = quit]: ").strip()
        if choice.lower() in ("q", "quit"):
            return
        if choice.lower() == "auto":
            auto = True
            continue
        ranked = [c.sleeper_id for c in (ro.candidates if ro is not None else rec.candidates)]
        if not choice:
            sid = ranked[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(ranked):
            sid = ranked[int(choice) - 1]
        else:
            sid = by_name.get(choice.lower()) or next((s for n, s in by_name.items() if choice.lower() in n), None)
            if sid is None or sid in mock.state.drafted:
                print("Not an available player; try again.")
                continue
        mp = mock.pick(sid)
        print(f"  {mp.pick:>3}. You        {mp.name}")
    names = dict(zip(board.players["sleeper_id"].to_list(), board.players["name"].to_list()))
    print("\nYour roster:", ", ".join(names.get(p, p) for p in mock.state.roster()))
    with pl.Config(tbl_rows=40):
        print(mock.projection(ev, n_worlds=sim.rollout_worlds // 2, waiver_slots=sim.waiver_slots))
    print("Projected by this model, against opponents who draft around ADP: it flatters the engine's own team.")


def cmd_opponents(args) -> None:
    from draftai import pipeline
    from draftai.backtest.replay import fit_opponents, load_real_drafts
    from draftai.config import load_backtest_config
    from draftai.ingestion.sleeper import SleeperClient

    paths = Paths().ensure()
    ctx = _ctx(args)
    cfg = load_league_config(Path(args.config) if args.config else None)
    username = args.username or cfg.username
    bt = load_backtest_config()
    seasons = args.seasons or bt.seasons
    if not username:
        raise SystemExit("set username in configs/league.yaml or pass --username")
    drafts = [d for d in load_real_drafts(SleeperClient(), username, seasons) if d.season < ctx.season]
    sim = load_sim_config()
    if drafts:
        model = fit_opponents(drafts, sim, load_train_config(), paths, bt.game_samples)
        model.save(pipeline.opponents_dir(paths) / "history.json")
        print(model.describe())
        import json

        print(json.dumps(model.diagnostics, indent=2))
    else:
        print(f"no completed drafts of {username} in {seasons}; live opponents use the calibrated defaults")
    board = pipeline.load_board(paths, ctx.season)
    live = pipeline.live_opponents(ctx, board, sim, paths)
    print("live:", live.describe())
    print("live calibration:", {k: v for k, v in live.diagnostics.items() if k != "key"})


def cmd_backtest(args) -> None:
    from draftai.backtest import replay as rp
    from draftai.backtest.history import LeagueRules
    from draftai.backtest.run import gate, run, summary, write_report
    from draftai.config import load_backtest_config
    from draftai.ingestion.sleeper import SleeperClient

    paths = Paths().ensure()
    ctx = _ctx(args)
    bt = load_backtest_config()
    for k in ("seasons", "leagues", "expensive_leagues", "strategies", "jobs"):
        if getattr(args, k, None) is not None:
            setattr(bt, k, getattr(args, k))
    if args.quick:
        bt.leagues, bt.expensive_leagues, bt.schedules, bt.rollout_worlds = 24, 6, 20, 200
    rules = LeagueRules.from_ctx(ctx)
    sim, train_cfg = load_sim_config(), load_train_config()
    t0 = time.time()
    res = run(rules, sim, train_cfg, bt, paths)
    cfg = load_league_config(Path(args.config) if args.config else None)
    username = args.username or cfg.username
    if not args.no_replay and username:
        from draftai.backtest.run import opponent_model

        drafts = [d for d in rp.load_real_drafts(SleeperClient(), username, bt.seasons) if d.season < ctx.season]
        frames = []
        for d in drafts:
            from draftai.backtest.history import season_data

            sd = season_data(d.season, d.rules, sim, train_cfg, paths, bt.game_samples)
            # Substitute picks imitate this real room, so they use the fitted model when there is one.
            model = opponent_model(sd.board, d.rules, sim, paths, room="fitted")
            names = [s for s in bt.strategies if s != "points"]
            frames.append(rp.replay(d, sim, train_cfg, bt, paths, model, names))
        if frames:
            import polars as pl

            res.replay = pl.concat(frames)
            res.replay_meta = {"drafts": [d.label for d in drafts],
                               "unmatched": int(res.replay["unmatched_picks"].max())}
    report = write_report(res, rules, bt, paths.artifacts / "reports")
    import polars as pl

    with pl.Config(tbl_rows=20, tbl_width_chars=160):
        print(summary(res.rows))
        if res.replay is not None:
            print(rp.summarise(res.replay))
    g = gate(res.rows)
    print(f"\nV4 gate (rollout beats ADP and VOR on P(top {rules.playoff_teams})): {'PASS' if g.get('passed') else 'FAIL'}")
    print(f"report: {report}  ({time.time() - t0:.0f} s)")


def cmd_app(args) -> None:
    app = Paths().root / "app" / "streamlit_app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *args.streamlit_args]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="draftai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def league_args(p):
        p.add_argument("--config", help="league YAML (default configs/league.yaml, else the example)")
        p.add_argument("--league-id")
        p.add_argument("--draft-id")
        p.add_argument("--username")
        p.add_argument("--slot", type=int, help="your draft slot (offline)")

    p = sub.add_parser("ingest", help="fetch and normalise data")
    p.add_argument("--sleeper-only", action="store_true", help="only refresh Sleeper players and the crosswalk")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("league", help="show league settings")
    league_args(p)
    p.set_defaults(fn=cmd_league)

    p = sub.add_parser("train", help="train and evaluate projection models")
    league_args(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("project", help="project the season and build the draft board")
    league_args(p)
    p.add_argument("--as-of", help="cut-off date (default: today or opening night)")
    p.add_argument("--adp-csv", help="ADP CSV (sleeper_id or name, adp, optional sd)")
    p.add_argument("--top", type=int, default=30)
    p.set_defaults(fn=cmd_project)

    p = sub.add_parser("lockin-experiment", help="Lock-In headline experiment")
    league_args(p)
    p.add_argument("--season", type=int)
    p.set_defaults(fn=cmd_lockin)

    p = sub.add_parser("adp", help="refresh ADP on the board")
    league_args(p)
    p.add_argument("--csv")
    p.add_argument("--top", type=int, default=30)
    p.set_defaults(fn=cmd_adp)

    p = sub.add_parser("recommend", help="recommend a pick")
    league_args(p)
    p.add_argument("--follow", action="store_true", help="poll a live draft and update after every pick")
    p.add_argument("--picks", nargs="*", help="offline: picks so far, in order (Sleeper ids or exact names)")
    p.add_argument("--rollout", action="store_true", help="Monte Carlo rollouts with confidence (V4)")
    p.add_argument("--top", type=int, default=8)
    p.set_defaults(fn=cmd_recommend)

    p = sub.add_parser("mock", help="offline mock draft; the other teams draft themselves")
    league_args(p)
    p.add_argument("--auto", action="store_true", help="the engine drafts for you too")
    p.add_argument("--rollout", action="store_true", help="use Monte Carlo rollouts for your picks")
    p.add_argument("--seed", type=int, help="replay a mock room")
    p.add_argument("--quiet", action="store_true", help="do not print the other teams' picks")
    p.add_argument("--top", type=int, default=6)
    p.set_defaults(fn=cmd_mock)

    p = sub.add_parser("opponents", help="fit the opponent model on your real Sleeper drafts")
    league_args(p)
    p.add_argument("--seasons", type=int, nargs="*")
    p.set_defaults(fn=cmd_opponents)

    p = sub.add_parser("backtest", help="paired draft backtest on past seasons (slow: minutes)")
    league_args(p)
    p.add_argument("--seasons", type=int, nargs="*")
    p.add_argument("--leagues", type=int, help="simulated leagues per season")
    p.add_argument("--expensive-leagues", dest="expensive_leagues", type=int,
                   help="leagues that also run greedy_survival and rollout")
    p.add_argument("--strategies", nargs="*")
    p.add_argument("--jobs", type=int)
    p.add_argument("--quick", action="store_true", help="small run for a smoke test")
    p.add_argument("--no-replay", action="store_true", help="skip replays of your real drafts")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("app", help="launch the Streamlit app")
    p.add_argument("streamlit_args", nargs="*")
    p.set_defaults(fn=cmd_app)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
