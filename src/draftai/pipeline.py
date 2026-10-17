"""End-to-end commands: ingest -> train -> project -> board. Each step reads the previous
step's files, so any step can be rerun alone."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from draftai.config import LeagueConfig, Paths, SimConfig, TrainConfig
from draftai.draft import adp as adp_mod
from draftai.draft.league import LeagueContext
from draftai.features.build import availability_stats, build_training_set, league_b2b_multiplier
from draftai.ingestion.sleeper import SleeperClient
from draftai.normalise import tables
from draftai.normalise.crosswalk import build_crosswalk, nba_player_directory, sleeper_players_table
from draftai.scoring import ScoringSettings
from draftai.value.board import Board, build_board
from draftai.value.schedule import add_weeks, week_plan


def board_dir(paths: Paths, season: int) -> Path:
    return paths.artifacts / "board" / str(season)


def models_dir(paths: Paths, kind: str) -> Path:
    return paths.artifacts / "models" / kind


# --- ingest ------------------------------------------------------------------------
def ingest(train: TrainConfig, league: LeagueConfig, paths: Paths, log=print) -> None:
    tables.ingest_seasons(train.ingest_seasons, train.target_season, paths=paths,
                          include_cup_final=league.include_cup_final, log=log)
    refresh_sleeper(paths, log)


def refresh_sleeper(paths: Paths, log=print) -> pl.DataFrame:
    """Daily: the Sleeper players file (eligibility, teams, injuries, search rank) and the
    NBA <-> Sleeper crosswalk."""
    sp = sleeper_players_table(SleeperClient().players())
    tables.write_table(sp, paths, "sleeper_players")
    pg, bio, dr = tables.load_table("player_games", paths), tables.load_table("player_bio", paths), tables.load_table("draft_class", paths)
    directory = nba_player_directory(pg, bio, dr)
    recent = directory.filter(pl.col("last_season") >= int(pg["season"].max()) - 3)
    cw = build_crosswalk(recent, sp, paths.configs / "id_overrides.csv")
    tables.write_table(cw, paths, "crosswalk")
    log(f"sleeper players: {sp.height}; crosswalk: {cw.height} matches")
    return cw


# --- train -------------------------------------------------------------------------
def train(train_cfg: TrainConfig, settings: ScoringSettings, paths: Paths, log=print):
    from draftai.models.artifacts import save_models
    from draftai.models.train import train_all, write_report

    pg, tg = tables.load_table("player_games", paths), tables.load_table("team_games", paths)
    bio, dr = tables.load_table("player_bio", paths), tables.load_table("draft_class", paths)
    seasons = list(range(train_cfg.first_target_season, train_cfg.test_season + 1))
    ts = build_training_set(pg, tg, bio, dr, settings, seasons, log=log)
    paths.features.mkdir(parents=True, exist_ok=True)
    ts.write_parquet(paths.features / "training_set.parquet")
    result = train_all(ts, settings, train_cfg, log=log)
    report = write_report(result, paths.artifacts / "reports")
    snap = date.today()
    save_models(result.models_test, models_dir(paths, "test"), train_cfg, snap,
                {"trained_through": train_cfg.test_season - 1, "summary": result.summary})
    save_models(result.models_final, models_dir(paths, "final"), train_cfg, snap,
                {"trained_through": train_cfg.test_season, "summary": result.summary})
    result.test_rows.write_parquet(paths.artifacts / "reports" / "test_rows.parquet")
    log(f"report: {report}")
    return result


# --- project + board ---------------------------------------------------------------
def live_inputs(paths: Paths) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Roster at the cut-off (NBA id -> Sleeper team), the projection pool, and Sleeper
    players with no NBA history (undrafted signings) keyed by a synthetic negative id."""
    sp = tables.load_table("sleeper_players", paths)
    cw = tables.load_table("crosswalk", paths)
    active = sp.filter(pl.col("active") & pl.col("team").is_not_null() & (pl.col("positions").list.len() > 0))
    matched = active.join(cw, on="sleeper_id", how="inner")
    roster = matched.select(pl.col("nba_id").alias("player_id"), "team")
    pool = roster.select("player_id")
    unmatched = active.join(cw, on="sleeper_id", how="anti").with_columns(
        (-pl.col("sleeper_id").cast(pl.Int64)).alias("player_id")
    )
    return roster, pool, unmatched.select("player_id", "team", "sleeper_id")


FOUL_PRIOR_GAMES = 20.0


def foul_rates(season: int, settings: ScoringSettings, log=print) -> pl.DataFrame | None:
    """Expected fantasy points per game from technical and flagrant fouls.

    Fouls are not in standard box scores; Sleeper's (undocumented) stats feed counts them.
    Each player's rate is shrunk toward the league mean. Optional: on any failure the
    adjustment is skipped and the projection is otherwise unchanged.
    """
    if not settings.scores_fouls:
        return None
    try:
        stats = SleeperClient().season_stats(season)
    except Exception as e:  # undocumented endpoint
        log(f"foul rates unavailable ({e}); skipping the technical/flagrant adjustment")
        return None
    rows = [{"sleeper_id": k, "gp": v.get("gp") or 0.0, "tf": v.get("tf") or 0.0, "ff": v.get("ff") or 0.0}
            for k, v in stats.items() if v and not k.startswith("TEAM")]
    df = pl.DataFrame(rows).filter(pl.col("gp") > 0)
    if df.height == 0:
        return None
    lg_tf, lg_ff = df["tf"].sum() / df["gp"].sum(), df["ff"].sum() / df["gp"].sum()
    k = FOUL_PRIOR_GAMES
    w_tf, w_ff = settings.weights.get("tf", 0.0), settings.weights.get("ff", 0.0)
    return df.select(
        "sleeper_id",
        (w_tf * (pl.col("tf") + k * lg_tf) / (pl.col("gp") + k)
         + w_ff * (pl.col("ff") + k * lg_ff) / (pl.col("gp") + k)).alias("foul_fp_pg"),
    )


def project(ctx: LeagueContext, sim: SimConfig, paths: Paths, as_of: date | None = None,
            adp_csv: Path | None = None, adp_from: pl.DataFrame | None = None, log=print) -> Board:
    """Project the season and build the board. ADP comes from ``adp_csv``, else
    ``adp_from`` (sleeper_id, adp, adp_source[, adp_sd]: e.g. the previous board's, so a
    rebuild keeps ADP aggregated from your drafts), else Sleeper's search rank."""
    from draftai.models.artifacts import load_models
    from draftai.models.project import (
        apply_availability_overrides,
        apply_minutes_overrides,
        season_projections,
    )

    Y = ctx.season
    pg, tg = tables.load_table("player_games", paths), tables.load_table("team_games", paths)
    bio, dr = tables.load_table("player_bio", paths), tables.load_table("draft_class", paths)
    sched = tables.load_table("schedule", paths, seasons=[Y])
    as_of = as_of or min(date.today(), sched["game_date"].min())
    roster, pool, unmatched = live_inputs(paths)
    models = load_models(models_dir(paths, "final"))
    proj = season_projections(pg, tg, bio, dr, models, ctx.scoring, Y, as_of, roster, pool=pool,
                              extra_rookies=unmatched.select("player_id", "team"), explain=True)
    sp = tables.load_table("sleeper_players", paths)
    cw = tables.load_table("crosswalk", paths)
    ids = pl.concat([cw.select(pl.col("nba_id").alias("player_id"), "sleeper_id"),
                     unmatched.select("player_id", "sleeper_id")])
    proj = (
        proj.join(ids, on="player_id", how="inner")
        .join(sp.select("sleeper_id", pl.col("full_name").alias("name"), pl.col("team").alias("sleeper_team"), "positions",
                        "injury_status", "injury_body_part", "search_rank", "years_exp"),
              on="sleeper_id", how="inner")
        .with_columns(pl.coalesce("sleeper_team", "team_now").alias("team_now"))
        .unique("sleeper_id", keep="first")
    )
    proj = apply_availability_overrides(proj, paths.configs / "availability_overrides.csv")
    log(f"team minutes budget: strength {models.minutes_budget:g}, target "
        f"{float(proj['team_minutes_target'][0]):.1f} min per team game")
    proj = apply_minutes_overrides(proj, paths.configs / "minutes_overrides.csv", ctx.scoring,
                                   log_dir=board_dir(paths, Y), log=log)
    fouls = foul_rates(Y - 1, ctx.scoring, log)
    if fouls is not None:
        proj = proj.join(fouls, on="sleeper_id", how="left").with_columns(
            pl.col("foul_fp_pg").fill_null(float(fouls["foul_fp_pg"].mean())))
        log(f"technical/flagrant adjustment: mean {proj['foul_fp_pg'].mean():+.2f} FP/G")
    schedule = add_weeks(sched)
    plan = week_plan(schedule, ctx.playoff_week_start, ctx.playoff_teams, ctx.playoff_weight, ctx.playoff_round_type)
    hist = pg.filter(pl.col("game_date") < as_of)
    b2b = league_b2b_multiplier(availability_stats(hist.filter(pl.col("season") >= Y - 3),
                                                   tg.filter((pl.col("season") >= Y - 3) & (pl.col("game_date") < as_of))))
    log(f"projected {proj.height} players as of {as_of}; back-to-back rest multiplier {b2b:.2f}")
    board = build_board(proj, hist, schedule, plan, ctx.scoring, ctx.roster_positions, ctx.teams, b2b, Y,
                        sim.game_samples, sim.min_donor_games, sim.donor_neighbours, sim.seed, log=log)
    if adp_csv:
        adp = adp_mod.from_csv(adp_csv, board.players)
    elif adp_from is not None:
        adp = adp_from.select([c for c in ("sleeper_id", "adp", "adp_source", "adp_sd") if c in adp_from.columns])
    else:
        adp = adp_mod.from_search_rank(board.players)
    board.players = adp_mod.attach(board.players, adp)
    board.save(board_dir(paths, Y))
    log(f"board saved to {board_dir(paths, Y)}")
    return board


def load_board(paths: Paths, season: int) -> Board:
    d = board_dir(paths, season)
    if not (d / "board.parquet").exists():
        raise FileNotFoundError(f"no board for {season}; run `draftai project` first")
    return Board.load(d)


# --- opponents (V4) -----------------------------------------------------------------
def opponents_dir(paths: Paths) -> Path:
    return paths.artifacts / "opponents"


def live_opponents(ctx: LeagueContext, board: Board, sim: SimConfig, paths: Paths, log=print):
    """The opponent model for a live draft or a mock on this board.

    beta(t) is calibrated so simulated drafts reproduce the configured ADP spread (the same
    spread the survival odds use; fit it from your drafts with `draftai adp --username`).
    The need weight comes from the model fitted on your real drafts (`draftai opponents`)
    when there is one. Its off-board share is not carried over: it was measured against
    the backtest's proxy market, so it mixes drafter noise with proxy error. Cached per
    league shape and board.
    """
    import hashlib

    from draftai.draft.needs import NeedModel
    from draftai.draft.opponents import Market, OpponentModel, calibrate
    from draftai.draft.order import pick_owners

    hist = opponents_dir(paths) / "history.json"
    fitted = OpponentModel.load(hist) if hist.exists() else None
    adp_hash = hashlib.sha256(board.players["adp"].to_numpy().tobytes()).hexdigest()[:10]
    key = (f"{ctx.teams}_{ctx.rounds}_{ctx.draft_type}_{ctx.reversal_round}_{sim.adp_sigma_a}_{sim.adp_sigma_b}_"
           f"{sim.opponent_window}_{adp_hash}_{fitted.gamma if fitted else 0.8:.4f}")
    path = opponents_dir(paths) / f"live_{ctx.teams}t.json"
    if path.exists():
        m = OpponentModel.load(path)
        if m.diagnostics.get("key") == key:
            return m
    log("calibrating the opponent model to this board (once per board)...")
    needs = NeedModel(board.positions(), board.slots)
    owners = pick_owners(ctx.teams, ctx.rounds, ctx.draft_type, ctx.reversal_round, ctx.traded)
    m = calibrate(Market(board.players["adp"].to_numpy(), needs), owners, sim.adp_sigma_a, sim.adp_sigma_b,
                  gamma=fitted.gamma if fitted else 0.8, window=sim.opponent_window)
    m.diagnostics["key"] = key
    m.diagnostics["need_weight_from"] = "fitted on your drafts" if fitted else "default"
    m.save(path)
    return m
