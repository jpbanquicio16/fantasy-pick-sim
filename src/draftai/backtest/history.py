"""Past seasons as they looked on draft day, and what then happened.

Everything a draft sees is frozen at the season's cut-off (opening night; no NBA games are
played between a September draft and opening night, so the features are identical):

* projections from models trained only on earlier seasons;
* the board built from them, on that season's real schedule;
* a market. Historical ADP snapshots are not available, so the market is a naive
  pre-season proxy: last season's FP/G (else the three-season weighted FP/G; rookies:
  the draft-slot projection), with pre-season injury news priced at half strength. Real
  drafters rank mostly on per-game production: on the real 2025 LeFantasy draft this
  proxy has a rank correlation of about 0.75 with the actual pick order, against 0.60 for
  FP/G x games share. Real ADP is sharper still, so gains against this market overstate
  gains against a real draft room; the replay of real drafts (``replay.py``) is the
  honest check.

What happened (``Realised``) is only read afterwards, to score the season: each player's
locked score per week under the optimal lock rule with his pre-season thresholds, whether
he played that week, and the in-season expectation a manager would use to set lineups
(the projection updated with games played before that week).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from draftai.config import Paths, SimConfig, TrainConfig
from draftai.features.asof import season_cutoff
from draftai.features.build import availability_stats, league_b2b_multiplier, roster_from_first_game
from draftai.normalise import tables
from draftai.normalise.crosswalk import build_crosswalk, nba_player_directory
from draftai.scoring import ScoringSettings, score_frame
from draftai.value.board import Board, build_board
from draftai.value.lockin import lockin_value, realised_lock, thresholds
from draftai.value.schedule import add_weeks, week_plan
from draftai.value.weekly import Availability

PERF_PRIOR_GAMES = 15.0  # in-season updates: weight of the projection, in games


@dataclass
class LeagueRules:
    """What defines a backtest league (read from the connected league or a real draft)."""

    teams: int
    rounds: int
    roster_positions: list[str]
    scoring: ScoringSettings
    draft_type: str = "snake"
    reversal_round: int = 0
    playoff_week_start: int = 22
    playoff_teams: int = 6
    playoff_round_type: int = 0
    playoff_weight: float = 1.5

    @classmethod
    def from_ctx(cls, ctx) -> "LeagueRules":
        return cls(ctx.teams, ctx.rounds, list(ctx.roster_positions), ctx.scoring, ctx.draft_type, ctx.reversal_round,
                   ctx.playoff_week_start, ctx.playoff_teams, ctx.playoff_round_type, ctx.playoff_weight)


@dataclass
class Realised:
    locked: np.ndarray  # (players, active weeks) locked score if started all week
    healthy: np.ndarray  # (players, active weeks) played at least one game
    expected: np.ndarray  # (players, active weeks) in-season expectation for setting lineups
    weights: np.ndarray  # (active weeks,)
    playoff: np.ndarray  # (active weeks,) bool
    weeks: list[int]

    @property
    def season_value(self) -> np.ndarray:
        return self.locked @ self.weights


@dataclass
class SeasonData:
    season: int
    board: Board
    realised: Realised
    b2b: float
    meta: dict = field(default_factory=dict)


# --- models -----------------------------------------------------------------------------
def fold_models(season: int, train_cfg: TrainConfig, paths: Paths, log=print, settings: ScoringSettings | None = None):
    """Models trained only on target seasons (and draft classes) before ``season``, cached
    on disk.

    Blend weights and conformal corrections come from the trained manifest; they were
    chosen on the rolling validation seasons, so for those seasons the backtest carries a
    mild hyperparameter look-ahead. The test season is clean.
    """
    import json
    from datetime import date

    from draftai.models.artifacts import load_models, save_models
    from draftai.models.train import _fit, _new_models, projection_signature

    signature = projection_signature(train_cfg)

    def current(directory: Path) -> bool:
        """Saved with today's feature set and projection setup (older folds are refitted)."""
        m = directory / "manifest.json"
        if not m.exists():
            return False
        man = json.loads(m.read_text())
        return man.get("signature", {}) == signature

    def with_rookies(models, directory: Path):
        if train_cfg.rookie_model != "model":
            models.rookies = None
            return models
        if models.rookies is not None:
            return models
        import joblib

        from draftai.models.rookies import RookieModel, class_rows

        pg, tg = tables.load_table("player_games", paths), tables.load_table("team_games", paths)
        bio, dr = tables.load_table("player_bio", paths), tables.load_table("draft_class", paths)
        rows = class_rows(pg, tg, bio, dr, settings or ScoringSettings(), range(int(dr["draft_year"].min()), season))
        final = load_models(paths.artifacts / "models" / "final").rookies
        adj = (final.adj_fp, final.adj_share) if final is not None else ((0.0, 0.0), (0.0, 0.0))
        models.rookies = RookieModel(alpha=train_cfg.rookie_alpha, k=train_cfg.rookie_k, adj_fp=adj[0],
                                     adj_share=adj[1]).fit(rows, settings or ScoringSettings())
        joblib.dump(models.rookies, directory / "rookies.joblib")
        log(f"rookie model for {season}: draft classes before {season}")
        return models

    if season == train_cfg.test_season and current(paths.artifacts / "models" / "test"):
        return with_rookies(load_models(paths.artifacts / "models" / "test"), paths.artifacts / "models" / "test")
    d = paths.artifacts / "models" / f"fold_{season}"
    if current(d):
        return with_rookies(load_models(d), d)

    final = json.loads((paths.artifacts / "models" / "final" / "manifest.json").read_text())
    ts = pl.read_parquet(paths.features / "training_set.parquet")
    train = ts.filter((pl.col("season") < season) & (pl.col("season") >= train_cfg.first_target_season))
    if train.height == 0:
        raise ValueError(f"no training seasons before {season}")
    models = _fit(_new_models(train_cfg), train)
    models.projection.set_blends(final["blends"])
    models.minutes_budget = float(final.get("minutes_budget", 0.0))
    models.fp_q.adj_lo, models.fp_q.adj_hi = final["fp_q"]["adj"]
    models.share_q.adj_lo, models.share_q.adj_hi = final["share_q"]["adj"]
    save_models(models, d, train_cfg, date.today(), {"trained_through": season - 1})
    log(f"fold models for {season}: trained on {train_cfg.first_target_season}-{season - 1}")
    return with_rookies(models, d)


# --- the pool and positions ---------------------------------------------------------------
HEIGHT_POSITIONS = [(76.0, ["PG", "SG"]), (78.5, ["SG", "SF"]), (80.5, ["SF", "PF"]), (82.0, ["PF", "C"]),
                    (999.0, ["C"])]


def height_positions(height: float | None) -> list[str]:
    """Fallback eligibility for players Sleeper's file no longer lists (mostly retired)."""
    h = 79.0 if height is None or np.isnan(height) else height
    return next(p for cut, p in HEIGHT_POSITIONS if h <= cut)


def season_pool(pg: pl.DataFrame, draft: pl.DataFrame, season: int) -> pl.DataFrame:
    """Everyone draftable at the cut-off: players with games in either prior season plus the
    season's draft class, and their team (first team in the season, else last known)."""
    prior = pg.filter(pl.col("season").is_in([season - 1, season - 2])).sort("game_date")
    last_team = prior.group_by("player_id").agg(pl.col("team").last().alias("last_team"))
    first_team = roster_from_first_game(pg, season).rename({"team": "first_team"})
    rookies = draft.filter(pl.col("draft_year") == season).select("player_id", pl.col("team").alias("draft_team"))
    ids = pl.concat([last_team.select("player_id"), rookies.select("player_id")]).unique()
    return (
        ids.join(first_team, on="player_id", how="left")
        .join(last_team, on="player_id", how="left")
        .join(rookies, on="player_id", how="left")
        .select("player_id", pl.coalesce("first_team", "last_team", "draft_team").alias("team"))
        .filter(pl.col("team").is_not_null())
    )


def identities(pool: pl.DataFrame, pg: pl.DataFrame, bio: pl.DataFrame, draft: pl.DataFrame,
               paths: Paths) -> pl.DataFrame:
    """player_id -> sleeper_id, name, positions (Sleeper's; height fallback when unmatched)."""
    sp = tables.load_table("sleeper_players", paths)
    directory = nba_player_directory(pg, bio, draft).join(pool.select("player_id"), on="player_id", how="semi")
    cw = pl.concat([tables.load_table("crosswalk", paths), build_crosswalk(directory, sp, paths.configs / "id_overrides.csv")],
                   how="vertical_relaxed").unique("nba_id", keep="first")
    heights = bio.sort("season").group_by("player_id").agg(pl.col("height_in").last())
    names = directory.select("player_id", "player_name")
    out = (
        pool.select("player_id")
        .join(names, on="player_id", how="left")
        .join(heights, on="player_id", how="left")
        .join(cw.select(pl.col("nba_id").alias("player_id"), "sleeper_id"), on="player_id", how="left")
        .join(sp.select("sleeper_id", pl.col("positions").alias("sp_positions")), on="sleeper_id", how="left")
    )
    rows = []
    for r in out.iter_rows(named=True):
        has = r["sp_positions"] is not None and len(r["sp_positions"]) > 0
        rows.append({
            "player_id": r["player_id"],
            "sleeper_id": r["sleeper_id"] if r["sleeper_id"] is not None else f"nba:{r['player_id']}",
            "name": r["player_name"] or str(r["player_id"]),
            "positions": list(r["sp_positions"]) if has else height_positions(r["height_in"]),
            "position_source": "sleeper" if has else "height",
        })
    return pl.DataFrame(rows, schema={"player_id": pl.Int64, "sleeper_id": pl.Utf8, "name": pl.Utf8,
                                      "positions": pl.List(pl.Utf8), "position_source": pl.Utf8}).unique("sleeper_id")


def apply_news(proj: pl.DataFrame, path: Path, season: int) -> pl.DataFrame:
    """Pre-season news (long injuries, suspensions) reported before the season's draft.

    Historical Sleeper injury status is not available, so a small hand-maintained file
    (configs/backtest_news.csv: season, name, share or multiplier, note) stands in for it.
    It reaches the projections and the market alike: real drafters knew it too.
    """
    from draftai.normalise.crosswalk import norm_name

    proj = proj.with_columns(pl.lit(1.0).alias("news_mult"))
    if not path.exists():
        return proj
    news = pl.read_csv(path, schema_overrides={"share": pl.Float64, "multiplier": pl.Float64}).filter(
        pl.col("season") == season)
    if news.height == 0:
        return proj
    news = news.with_columns(pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8).alias("key"))
    p = proj.with_columns(pl.col("name").map_elements(norm_name, return_dtype=pl.Utf8).alias("key")).join(
        news.select("key", "share", "multiplier"), on="key", how="left")
    new = (pl.when(pl.col("share").is_not_null()).then(pl.col("share"))
           .when(pl.col("multiplier").is_not_null()).then(pl.col("share_hat") * pl.col("multiplier"))
           .otherwise(pl.col("share_hat")).clip(0, 1))
    ratio = new / pl.col("share_hat").clip(1e-6)
    return p.with_columns(
        ratio.alias("news_mult"),
        new.alias("share_hat"),
        *[(pl.col(c) * ratio).clip(0, 1).alias(c) for c in ("share_p10", "share_p50", "share_p90")],
    ).drop("key", "share", "multiplier")


MARKET_NEWS_EXPONENT = 0.5  # how much of the pre-season news the market prices in


def market_proxy(players: pl.DataFrame) -> pl.DataFrame:
    """Naive pre-season market: per-game production, news half-priced, ranked."""
    news = pl.col("news_mult").fill_null(1.0) if "news_mult" in players.columns else pl.lit(1.0)
    value = (pl.coalesce("fp_pg_1", "fp_pg_w", "fp_pg") * news.pow(MARKET_NEWS_EXPONENT)).fill_null(0.0)
    ranked = players.select("sleeper_id", value.alias("v")).sort("v", descending=True)
    return ranked.with_row_index("adp", 1).select("sleeper_id", pl.col("adp").cast(pl.Float64),
                                                   pl.lit("naive_proxy").alias("adp_source"))


# --- building a season --------------------------------------------------------------------
def build_season(season: int, rules: LeagueRules, sim: SimConfig, train_cfg: TrainConfig, paths: Paths,
                 game_samples: int = 2000, log=print) -> SeasonData:
    from draftai.draft import adp as adp_mod
    from draftai.models.project import season_projections

    pg, tg = tables.load_table("player_games", paths), tables.load_table("team_games", paths)
    bio, dr = tables.load_table("player_bio", paths), tables.load_table("draft_class", paths)
    cut = season_cutoff(tg, season)
    models = fold_models(season, train_cfg, paths, log, settings=rules.scoring)
    pool = season_pool(pg, dr, season)
    proj = season_projections(pg, tg, bio, dr, models, rules.scoring, season, cut, pool.select("player_id", "team"),
                              pool=pool.select("player_id"))
    ids = identities(pool, pg, bio, dr, paths)
    proj = proj.join(ids, on="player_id", how="inner").with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("injury_status"))
    proj = apply_news(proj, paths.configs / "backtest_news.csv", season)
    sched = add_weeks(tables.schedule_from_team_games(tg.filter(pl.col("season") == season)))
    plan = week_plan(sched, rules.playoff_week_start, rules.playoff_teams, rules.playoff_weight, rules.playoff_round_type)
    hist = pg.filter(pl.col("game_date") < cut)
    b2b = league_b2b_multiplier(availability_stats(hist.filter(pl.col("season") >= season - 3),
                                                   tg.filter((pl.col("season") >= season - 3) & (pl.col("game_date") < cut))))
    board = build_board(proj, hist, sched, plan, rules.scoring, rules.roster_positions, rules.teams, b2b, season,
                        game_samples, sim.min_donor_games, sim.donor_neighbours, sim.seed)
    board.players = adp_mod.attach(board.players, market_proxy(board.players))
    realised = realise(board, season, pg, tg, rules.scoring, b2b, rules.playoff_week_start)
    top = board.players.sort("adp").head(rules.teams * rules.rounds)
    meta = {
        "cutoff": cut.isoformat(),
        "players": board.players.height,
        "drafted_with_sleeper_positions": float((top["position_source"] == "sleeper").mean()),
        "b2b_multiplier": b2b,
    }
    log(f"season {season}: board {board.players.height} players as of {cut}; "
        f"{meta['drafted_with_sleeper_positions']:.0%} of the draftable top have Sleeper positions")
    return SeasonData(season, board, realised, b2b, meta)


def realise(board: Board, season: int, pg: pl.DataFrame, tg: pl.DataFrame, settings: ScoringSettings,
            b2b: float, playoff_week_start: int) -> Realised:
    """Each board player's real season, week by week, under the pre-season lock rule."""
    active = board.weights > 0
    weeks = [w for w, a in zip(board.weeks, active) if a]
    widx = {w: i for i, w in enumerate(weeks)}
    sched = add_weeks(tables.schedule_from_team_games(tg.filter(pl.col("season") == season)))
    week_start = sched.group_by("week").agg(pl.col("game_date").min()).sort("week")
    starts = {int(w): d for w, d in week_start.iter_rows()}
    games = score_frame(pg.filter(pl.col("season") == season), settings).select(
        "player_id", "game_id", "game_date", "team", "fp")
    # Stints: the first runs from his team's opener, the last to his team's final game, so a
    # player who is hurt at the start or end of the season misses those weeks.
    bounds = sched.group_by("team").agg(pl.col("game_date").min().alias("t0"), pl.col("game_date").max().alias("t1"))
    stints = (
        games.group_by("player_id", "team").agg(pl.col("game_date").min().alias("a"), pl.col("game_date").max().alias("b"))
        .sort("player_id", "a")
        .with_columns(pl.col("a").rank("ordinal").over("player_id").alias("k"),
                      pl.len().over("player_id").alias("n"))
        .join(bounds, on="team")
        .with_columns(pl.when(pl.col("k") == 1).then(pl.col("t0")).otherwise(pl.col("a")).alias("a"),
                      pl.when(pl.col("k") == pl.col("n")).then(pl.col("t1")).otherwise(pl.col("b")).alias("b"))
    )
    team_games = (
        stints.select("player_id", "team", "a", "b")
        .join(sched.select("team", "game_id", "game_date", "week", "b2b2"), on="team")
        .filter(pl.col("game_date").is_between(pl.col("a"), pl.col("b")))
        .unique(["player_id", "game_date"], keep="first")
        .join(games.select("player_id", "game_id", "fp"), on=["player_id", "game_id"], how="left")
        .sort("player_id", "game_date")
    )
    by_player = {int(k[0]): g for k, g in team_games.partition_by("player_id", as_dict=True).items()}
    P, K = board.players.height, len(weeks)
    locked, healthy, expected = np.zeros((P, K)), np.zeros((P, K), dtype=bool), np.zeros((P, K))
    p = board.players
    for row, (pid, share, q, mu) in enumerate(zip(p["player_id"].to_list(), p["share_hat"].to_list(),
                                                  p["q_short_hat"].to_list(), p["fp_pg"].to_list())):
        g = by_player.get(int(pid))
        if g is None:
            continue
        dist = board.game_scores(row)
        avail = Availability.from_share(share or 0.0, q or 0.0, b2b)
        thr_cache, val_cache = {}, {}
        played_dates = [(d, f) for d, f in zip(g["game_date"].to_list(), g["fp"].to_list()) if f is not None]
        for (week,), wg in g.group_by(["week"], maintain_order=True):
            k = widx.get(int(week))
            if k is None:
                continue
            pattern = tuple(wg["b2b2"].to_list())
            if pattern not in thr_cache:
                qs = avail.game_q(pattern)
                thr_cache[pattern], val_cache[pattern] = thresholds(dist, qs), lockin_value(dist, qs)
            scores = wg["fp"].to_list()
            locked[row, k] = realised_lock(scores, thr_cache[pattern])
            healthy[row, k] = any(s is not None for s in scores)
            before = [f for d, f in played_dates if d < starts[int(week)]]
            ratio = (PERF_PRIOR_GAMES * mu + sum(before)) / (PERF_PRIOR_GAMES + len(before)) / max(mu, 1e-6)
            expected[row, k] = val_cache[pattern] * ratio
    po = np.array([wk >= playoff_week_start for wk in weeks])
    return Realised(locked, healthy, expected, board.weights[active], po, weeks)


# --- cache ----------------------------------------------------------------------------------
def season_key(season: int, rules: LeagueRules, sim: SimConfig, game_samples: int,
               train_cfg: TrainConfig | None = None) -> str:
    """Seasons are cached per league shape and projection setup (features, rate prior,
    rookie model); league size only moves VOR, so it is left out and applied with
    ``Board.with_teams`` instead."""
    import hashlib
    import json

    from draftai.models.train import projection_signature

    cfg = train_cfg or TrainConfig()
    blob = json.dumps({"season": season, "roster": rules.roster_positions, "scoring": rules.scoring.weights,
                       "stack": rules.scoring.bonus_stacking, "po": [rules.playoff_week_start, rules.playoff_teams,
                       rules.playoff_round_type, rules.playoff_weight], "seed": sim.seed, "samples": game_samples,
                       "donors": [sim.min_donor_games, sim.donor_neighbours], **projection_signature(cfg),
                       "rookies": [cfg.rookie_model, cfg.rookie_k, cfg.rookie_alpha], "v": 5}, sort_keys=True)
    return f"{season}_{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


def save_season(sd: SeasonData, directory: Path) -> None:
    import json

    sd.board.save(directory)
    r = sd.realised
    np.savez_compressed(directory / "realised.npz", locked=r.locked, healthy=r.healthy, expected=r.expected,
                        weights=r.weights, playoff=r.playoff, weeks=np.array(r.weeks))
    (directory / "season_meta.json").write_text(json.dumps({"season": sd.season, "b2b": sd.b2b, "meta": sd.meta}))


def load_season(directory: Path) -> SeasonData:
    import json

    m = json.loads((directory / "season_meta.json").read_text())
    z = np.load(directory / "realised.npz")
    r = Realised(z["locked"], z["healthy"], z["expected"], z["weights"], z["playoff"], z["weeks"].tolist())
    return SeasonData(m["season"], Board.load(directory), r, m["b2b"], m["meta"])


def season_data(season: int, rules: LeagueRules, sim: SimConfig, train_cfg: TrainConfig, paths: Paths,
                game_samples: int = 2000, log=print) -> SeasonData:
    """Build (or load the cached) season, then size the board to this league."""
    d = paths.artifacts / "backtest" / "seasons" / season_key(season, rules, sim, game_samples, train_cfg)
    if (d / "season_meta.json").exists():
        sd = load_season(d)
    else:
        sd = build_season(season, rules, sim, train_cfg, paths, game_samples, log)
        save_season(sd, d)
    sd.meta["cache_key"] = d.name
    if sd.board.teams != rules.teams:
        sd.board = sd.board.with_teams(rules.teams)
    return sd
