"""Command-line entry point: ``draftai <command>``.

    draftai ingest              fetch + normalise + validate NBA data and Sleeper players
    draftai league              show the league settings the engine will use
    draftai train               features, rolling-origin evaluation, test season, final models
    draftai project             project the draft season and build the board
    draftai lockin-experiment   the Lock-In headline experiment on the test season
    draftai adp                 show or refresh ADP (search rank, CSV, or your Sleeper drafts)
    draftai recommend           recommendation for a live draft, or an offline mock
    draftai app                 launch the Streamlit draft assistant
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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

    def show(state):
        if state.complete:
            print("Draft complete. Your roster:", ", ".join(names.get(p, p) for p in state.roster()))
            return
        rec = recommend(board, state, sim, ev)
        until = state.picks_until_mine()
        print(f"\n--- pick {state.current_pick} | your next pick {state.my_upcoming()[:1]} | {until} picks away ---")
        print(render(card(rec, board.players, ctx.scoring)))
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
    p.add_argument("--top", type=int, default=8)
    p.set_defaults(fn=cmd_recommend)

    p = sub.add_parser("app", help="launch the Streamlit app")
    p.add_argument("streamlit_args", nargs="*")
    p.set_defaults(fn=cmd_app)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
