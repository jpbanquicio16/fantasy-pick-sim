"""Team moves and role changes: the minutes a player's team has opened up, or filled.

Each player's team comes from the roster at the cut-off (Sleeper's ``team`` field live;
the first game of the season for history, a small stated leak for preseason moves), never
from the game logs, which cannot show a summer trade. Against last season, the minutes
on that team change through four channels:

* departures: minutes played for the team last season by players now elsewhere;
* arrivals: minutes this season's newcomers played for other teams last season;
* rookies and other newcomers: their draft-slot expected minutes;
* teammates returning from a long absence: the minutes they did not play last season.

    opportunity = departures - arrivals - rookies - returning      (per team game)

Each channel is measured against the league's average team that offseason, so zero means
a typical summer and +3 means three more minutes a game opened up than on a typical team.
The player's own arrival or return is excluded: it is his role, not his competition. Each
channel is also weighted by position (guard, wing, big from height, the same buckets as
the rate priors): a departing guard opens more minutes for guards than for centers.

Departures count only players still on a roster at the cut-off. Retirements and unsigned
free agents cannot be told apart from season-long injuries in historical data without
looking at the season itself, so they count in neither training nor live projections;
live, they leave the team minutes budget instead (models/budget.py).
"""

from __future__ import annotations

import numpy as np
import polars as pl

BUCKETS = ("guard", "wing", "big", "unk")
# Positional similarity between height buckets; unknown (most rookies) shares half with all.
_SIM = {("guard", "guard"): 1.0, ("wing", "wing"): 1.0, ("big", "big"): 1.0,
        ("guard", "wing"): 0.5, ("wing", "big"): 0.5, ("guard", "big"): 0.0}
LONG_ABSENCE = 15  # games: a missed run this long last season is an injury, not rest
HEALTHY_SHARE = 0.8  # reference games share when no earlier season shows a healthy one
TEAM_GAMES = 82.0
CHANNELS = ("opp_dep", "opp_arr", "opp_new", "opp_ret")


def position_similarity(a: str, b: str) -> float:
    if a == "unk" or b == "unk":
        return 0.5
    return _SIM.get((a, b), _SIM.get((b, a), 0.0))


SIM_MATRIX = np.array([[position_similarity(a, b) for b in BUCKETS] for a in BUCKETS])


def elasticity(mpg: np.ndarray) -> np.ndarray:
    """How readily a player's minutes move when a team's minutes do: most for rotation
    players in the middle of the range, little for 36-minute starters or deep reserves."""
    m = np.asarray(mpg, dtype=float)
    return np.maximum(m, 0.0) * np.maximum(1.0 - m / 40.0, 0.05)


def team_minutes_target(team_games: pl.DataFrame, seasons: int = 3) -> float:
    """Minutes per team game (48 x 5 plus overtime) over the last few seasons; about 241."""
    if team_games.height == 0 or "team_min" not in team_games.columns:
        return 241.0
    last = int(team_games["season"].max())
    tg = team_games.filter(pl.col("season") > last - seasons)
    return float(tg["team_min"].mean())


def returning_minutes() -> pl.Expr:
    """Extra minutes per team game a player returning from a long absence takes back: his
    usual minutes times the games share he lost last season."""
    long_absent = (pl.col("longest_absence").fill_null(0) >= LONG_ABSENCE) | (
        (pl.col("gp_1").fill_null(0) == 0) & (pl.col("gp_2").fill_null(0) > 0))
    ref = pl.mean_horizontal("share_2", "share_3").fill_null(HEALTHY_SHARE).clip(0.0, 0.9)
    gap = (ref - pl.col("share_1").fill_null(0.0)).clip(0.0, 1.0)
    return pl.when(long_absent).then(pl.col("mpg_w").fill_null(0.0) * gap).otherwise(0.0)


def opportunity(rows: pl.DataFrame, prev_games: pl.DataFrame, newcomers: pl.DataFrame | None) -> pl.DataFrame:
    """Add net minutes opportunity at the cut-off to the feature rows.

    ``rows`` needs player_id, team_now, hbucket, min_1, mpg_w, gp_1, gp_2, share_1-3 and
    longest_absence. ``prev_games`` are last season's player games (as-of). ``newcomers``
    (player_id, team, mpg, share) are rookies and other players with no recent history.

    Adds the four channels (``opp_dep``, ``opp_arr``, ``opp_new``, ``opp_ret``; position
    weighted, minutes per team game, each positive), the features ``opp_pos_mpg`` (their
    net) and ``opp_team_mpg`` (the unweighted net), and the newcomer sums the minutes
    budget's evaluation needs (``team_new_emin``, ``team_new_wt``).
    """
    n_games = prev_games.select("team", "game_id").unique().group_by("team").len()["len"]
    G = float(n_games.median()) if n_games.len() else TEAM_GAMES
    G = G if G > 0 else TEAM_GAMES
    stints = prev_games.group_by("player_id", "team").agg(pl.col("min").sum().alias("stint_min"))
    r = rows.with_columns(pl.col("hbucket").fill_null("unk"))
    at_team = r.select("player_id", "team_now").join(
        stints.rename({"team": "team_now"}), on=["player_id", "team_now"], how="left")
    r = r.join(at_team.select("player_id", pl.col("stint_min").fill_null(0.0).alias("min_here")), on="player_id",
               how="left")
    r = r.with_columns(
        ((pl.col("min_1").fill_null(0.0) - pl.col("min_here")).clip(0.0) / G).alias("_arr"),
        returning_minutes().alias("_ret"),
    )
    # Long table of contributions: (team, bucket, channel, minutes per team game).
    dep = stints.join(r.select("player_id", "team_now", "hbucket"), on="player_id", how="inner").filter(
        pl.col("team") != pl.col("team_now")).select("team", "hbucket", (pl.col("stint_min") / G).alias("v"),
                                                     pl.lit("opp_dep").alias("ch"))
    arr = r.select(pl.col("team_now").alias("team"), "hbucket", pl.col("_arr").alias("v"), pl.lit("opp_arr").alias("ch"))
    ret = r.select(pl.col("team_now").alias("team"), "hbucket", pl.col("_ret").alias("v"), pl.lit("opp_ret").alias("ch"))
    parts = [dep, arr, ret]
    if newcomers is not None and newcomers.height:
        parts.append(newcomers.select("team", pl.lit("unk").alias("hbucket"),
                                      (pl.col("mpg") * pl.col("share")).alias("v"), pl.lit("opp_new").alias("ch")))
    contrib = pl.concat([p.with_columns(pl.col("v").cast(pl.Float64)) for p in parts]).filter(pl.col("team").is_not_null())

    teams = sorted(set(contrib["team"].to_list()) | set(r["team_now"].drop_nulls().to_list()))
    t_idx = {t: i for i, t in enumerate(teams)}
    b_idx = {b: i for i, b in enumerate(BUCKETS)}
    c_idx = {c: i for i, c in enumerate(CHANNELS)}
    arr3 = np.zeros((len(teams), len(BUCKETS), len(CHANNELS)))
    for t, b, v, ch in contrib.iter_rows():
        arr3[t_idx[t], b_idx.get(b, 3), c_idx[ch]] += v

    # Relative to the league's average team this offseason: every team adds rookies and
    # loses retirees (which are not counted), so raw sums sit well below zero for everyone.
    arr3 = arr3 - arr3.mean(axis=0, keepdims=True)

    ti = np.array([t_idx.get(t, -1) for t in r["team_now"].to_list()])
    bi = np.array([b_idx.get(b, 3) for b in r["hbucket"].to_list()])
    weights = SIM_MATRIX[bi]  # (players, buckets)
    per = np.einsum("pb,pbc->pc", weights, arr3[np.maximum(ti, 0)])
    flat = arr3[np.maximum(ti, 0)].sum(axis=1)  # (players, channels), unweighted
    own = np.zeros((r.height, len(CHANNELS)))
    own[:, c_idx["opp_arr"]] = r["_arr"].to_numpy()
    own[:, c_idx["opp_ret"]] = r["_ret"].to_numpy()
    self_w = SIM_MATRIX[bi, bi][:, None]
    per = per - self_w * own
    flat = flat - own
    per[ti < 0] = 0.0
    flat[ti < 0] = 0.0
    sign = np.array([1.0, -1.0, -1.0, -1.0])
    out = r.drop("_arr", "_ret", "min_here").with_columns(
        *[pl.Series(c, per[:, c_idx[c]]) for c in CHANNELS],
        pl.Series("opp_pos_mpg", per @ sign),
        pl.Series("opp_team_mpg", flat @ sign),
    )
    return out.with_columns(*newcomer_budget_columns(newcomers))


def newcomer_budget_columns(newcomers: pl.DataFrame | None) -> list[pl.Expr]:
    """Per-row team sums over newcomers (expected minutes and their elasticity weight), so
    the training-time evaluation of the minutes budget sees the rookies it cannot score."""
    if newcomers is None or newcomers.height == 0:
        return [pl.lit(0.0).alias("team_new_emin"), pl.lit(0.0).alias("team_new_wt")]
    m, s = newcomers["mpg"].to_numpy().astype(float), newcomers["share"].to_numpy().astype(float)
    agg = pl.DataFrame({"team": newcomers["team"], "e": m * s, "w": s * elasticity(m)}).group_by("team").agg(
        pl.col("e").sum(), pl.col("w").sum())
    e = dict(zip(agg["team"].to_list(), agg["e"].to_list()))
    w = dict(zip(agg["team"].to_list(), agg["w"].to_list()))
    return [pl.col("team_now").replace_strict(e, default=0.0, return_dtype=pl.Float64).alias("team_new_emin"),
            pl.col("team_now").replace_strict(w, default=0.0, return_dtype=pl.Float64).alias("team_new_wt")]


def newcomer_minutes(newcomers: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    """(player_id, team, draft_pick) -> + mpg, share from the draft-slot baseline.
    Undrafted players get the second-round bucket, as their projections do."""
    from draftai.models.rookies import pick_bucket

    known = set(base["bucket"].to_list())
    return (
        newcomers.with_columns(pick_bucket(pl.col("draft_pick")).alias("bucket"))
        .with_columns(pl.when(pl.col("bucket").is_in(list(known))).then(pl.col("bucket"))
                      .otherwise(pl.lit("31-60")).alias("bucket"))
        .join(base.select("bucket", "mpg", "share"), on="bucket", how="left")
        .select("player_id", "team", pl.col("mpg").fill_null(10.0), pl.col("share").fill_null(0.3))
        .filter(pl.col("team").is_not_null())
    )
