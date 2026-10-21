"""Rookie projections (design section 9): players with no NBA history.

Two versions, compared draft class by draft class on rolling origin (``evaluate_rookies``):

* the **draft-slot baseline** (``rookie_baseline``): each pick bucket's historical
  first-year outcomes. It also supplies the rookies' expected minutes to the veterans'
  role features (``features.roles.newcomer_minutes``);
* the **rookie model** (``RookieModel``): small ridge regressions on what is known on draft
  day, plus the outcomes of the most similar past rookies (comparables) for the predictive
  range, the game-shape donors and the explanation ("players like this averaged X").

Inputs, in the design's order of value, as far as the data sources reach:

1. Minutes opportunity: veteran minutes per game his team opened at his positions since
   last season (departures minus arrivals, relative to the league's average team;
   ``features.roles.team_bucket_net``), and his team's record last season.
2. Draft position (log scale) and age. Later picks never project more games, minutes or
   per-minute production: the pick coefficients are constrained to be non-positive.
3. College or international/club background (the draft history's organisation type).
   College per-possession stats are not in nba_api, so they are not used.
4. Height and weight.

Games share is modelled for every pick, on inputs every pick has (slot, background, team).
Age and size enter only the models conditional on playing (minutes, per-minute rates,
bonuses): historically they come from the NBA's bio rows, which exist only for players who
reached the league, so in the games model their absence would announce the outcome. They
are static facts (birth date, height), read from a player's first bio row or, live, from
Sleeper's players file at the bio's reference date (30 June of the season's second year).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from draftai.features.build import COMPONENTS, availability_stats, height_bucket, season_stats
from draftai.features.roles import bucket_weighted, team_bucket_net
from draftai.scoring import ScoringSettings

PICK_BUCKETS = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 60)]
UNDRAFTED_PICK = 75  # as in the veterans' log_draft_pick
MIN_GP = 10  # FP/G outcomes and FP/G comparables need this many games
MIN_RATE_MINUTES = 100.0
INTL = "Other Team/Club"

SHARE_FEATURES = ["log_pick", "second_round", "intl", "team_win_1", "team_opp_mpg"]
PLAYED_FEATURES = SHARE_FEATURES + ["team_opp_pos", "age", "height_in", "weight_lb"]
NON_INCREASING = ("log_pick", "second_round")  # in the pick number
# Rates of the stats every scoring rewards may not rise with a later pick; turnovers may.
CONSTRAINED_RATES = ("pts", "reb", "ast", "stl", "blk", "fg3m")
COMP_WEIGHTS = {"log_pick": 2.0, "age": 1.0, "height_in": 1.0, "weight_lb": 0.5, "intl": 0.5}
N_DONORS = 12
N_NAMED = 5


# --- the draft-slot baseline -----------------------------------------------------------
def pick_bucket(pick: pl.Expr) -> pl.Expr:
    expr = pl.lit("undrafted")
    for lo, hi in reversed(PICK_BUCKETS):
        expr = pl.when(pick.is_between(lo, hi)).then(pl.lit(f"{lo}-{hi}")).otherwise(expr)
    return pl.when(pick.is_null()).then(pl.lit("undrafted")).otherwise(expr)


def rookie_baseline(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    draft: pl.DataFrame,
    settings: ScoringSettings,
    before_season: int,
) -> pl.DataFrame:
    """First-year outcomes by pick bucket for draft classes before ``before_season``."""
    classes = draft.filter(pl.col("draft_year") < before_season)
    seasons = classes["draft_year"].unique().to_list()
    pg = player_games.filter(pl.col("season").is_in(seasons))
    tg = team_games.filter(pl.col("season").is_in(seasons))
    ss = season_stats(pg, tg, settings).join(availability_stats(pg, tg), on=["player_id", "season"], how="left")
    team_gp = tg.group_by("season", "team").len().group_by("season").agg(pl.col("len").median().alias("team_gp"))
    rows = (
        classes.join(ss, left_on=["player_id", "draft_year"], right_on=["player_id", "season"], how="left")
        .join(team_gp, left_on="draft_year", right_on="season", how="left")
        .with_columns(
            pick_bucket(pl.col("overall_pick")).alias("bucket"),
            pl.col("gp").fill_null(0),
            pl.col("min").fill_null(0.0),
        )
        .with_columns((pl.col("gp") / pl.col("team_gp")).clip(0, 1).alias("share"))
    )
    played = pl.col("gp") >= 10
    return rows.group_by("bucket").agg(
        pl.len().alias("n"),
        pl.col("share").mean().alias("share"),
        pl.col("share").quantile(0.1).alias("share_p10"),
        pl.col("share").quantile(0.9).alias("share_p90"),
        (pl.col("min").filter(played).sum() / pl.col("gp").filter(played).sum()).alias("mpg"),
        pl.col("fp_pg").filter(played).mean().alias("fp_pg"),
        pl.col("fp_pg").filter(played).quantile(0.1).alias("fp_pg_p10"),
        pl.col("fp_pg").filter(played).quantile(0.5).alias("fp_pg_p50"),
        pl.col("fp_pg").filter(played).quantile(0.9).alias("fp_pg_p90"),
        (pl.col("fp_bonus").filter(played).sum() / pl.col("gp").filter(played).sum()).alias("bonus_pg"),
        *[
            (pl.col(c).filter(played).sum() / pl.col("min").filter(played).sum()).alias(f"rate_{c}")
            for c in COMPONENTS
        ],
        pl.col("player_id").filter(played).alias("donor_ids"),
    ).sort("bucket")


def baseline_bucket(pick: pl.Expr, known: set[str]) -> pl.Expr:
    """Undrafted signings have no draft-class history; the second round is the closest proxy."""
    b = pick_bucket(pick)
    return pl.when(b.is_in(list(known))).then(b).otherwise(pl.lit("31-60"))


# --- inputs on draft day ----------------------------------------------------------------
def team_records(team_games: pl.DataFrame, season: int) -> pl.DataFrame:
    """Each team's winning percentage in ``season`` (team, team_win_1)."""
    tg = team_games.filter(pl.col("season") == season).select("game_id", "team", "team_pts")
    if tg.height == 0:
        return pl.DataFrame(schema={"team": pl.Utf8, "team_win_1": pl.Float64})
    opp = tg.rename({"team": "opp_team", "team_pts": "opp_pts"})
    j = tg.join(opp, on="game_id").filter(pl.col("team") != pl.col("opp_team"))
    return j.group_by("team").agg((pl.col("team_pts") > pl.col("opp_pts")).cast(pl.Float64).mean().alias("team_win_1"))


@dataclass
class RookieContext:
    """The room each team offers a rookie, as of the cut-off."""

    net: pl.DataFrame  # team, hbucket, net veteran minutes per team game opened
    win: pl.DataFrame  # team, team_win_1


def rookie_context(player_games: pl.DataFrame, team_games: pl.DataFrame, roster: pl.DataFrame, bio: pl.DataFrame,
                   season: int) -> RookieContext:
    """From last season's games and bio rows before ``season`` (pass an as-of view).
    ``roster`` (player_id, team) is every player's team at the cut-off."""
    prev = player_games.filter(pl.col("season") == season - 1)
    heights = (bio.filter(pl.col("season") < season).sort("season").group_by("player_id")
               .agg(pl.col("height_in").drop_nulls().last()))
    buckets = heights.select("player_id", height_bucket(pl.col("height_in")).alias("hbucket"))
    return RookieContext(team_bucket_net(roster.select("player_id", "team"), prev, buckets),
                         team_records(team_games, season - 1))


def bio_attributes(bio: pl.DataFrame, season: int) -> pl.DataFrame:
    """Age (in ``season``), height and weight from each player's first bio row from
    ``season`` on: static facts, used only by the models conditional on playing."""
    first = bio.filter(pl.col("season") >= season).sort("season").group_by("player_id").agg(
        pl.col("season").first().alias("_s"), pl.col("age").first(), pl.col("height_in").first(),
        pl.col("weight_lb").first())
    return first.select("player_id", (pl.col("age") - (pl.col("_s") - season)).alias("age"), "height_in", "weight_lb")


def age_at_bio_date(birth: date | None, season: int) -> float | None:
    """Whole years on 30 June of the season's second year, the reference date of the
    NBA's bio ages (checked against several players' birth dates)."""
    if birth is None:
        return None
    ref = date(season + 1, 6, 30)
    return float(ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day)))


def sleeper_attributes(sleeper: pl.DataFrame, season: int) -> pl.DataFrame:
    """sleeper_id, age, height_in, weight_lb from Sleeper's players table (live rookies)."""
    cols = sleeper.columns
    return sleeper.select(
        "sleeper_id",
        pl.col("birth_date").map_elements(lambda d: age_at_bio_date(d, season), return_dtype=pl.Float64).alias("age")
        if "birth_date" in cols else pl.lit(None, dtype=pl.Float64).alias("age"),
        pl.col("height_in") if "height_in" in cols else pl.lit(None, dtype=pl.Float64).alias("height_in"),
        pl.col("weight_lb") if "weight_lb" in cols else pl.lit(None, dtype=pl.Float64).alias("weight_lb"),
    )


def rookie_features(rookies: pl.DataFrame, ctx: RookieContext) -> pl.DataFrame:
    """``rookies``: player_id, draft_pick (null = undrafted), org_type, team, age,
    height_in, weight_lb (any may be null). Adds the model inputs."""
    r = rookies
    for c, t in (("org_type", pl.Utf8), ("age", pl.Float64), ("height_in", pl.Float64), ("weight_lb", pl.Float64)):
        if c not in r.columns:
            r = r.with_columns(pl.lit(None, dtype=t).alias(c))
    pick = pl.col("draft_pick").cast(pl.Float64).fill_null(UNDRAFTED_PICK)
    r = r.with_columns(
        pick.log().alias("log_pick"),
        (pick > 30).cast(pl.Float64).alias("second_round"),
        (pl.col("org_type") == INTL).fill_null(False).cast(pl.Float64).alias("intl"),
        height_bucket(pl.col("height_in")).alias("hbucket"),
    ).join(ctx.win, on="team", how="left").with_columns(pl.col("team_win_1").fill_null(0.5))
    pos, flat = bucket_weighted(ctx.net, r["team"].to_list(), r["hbucket"].to_list())
    return r.with_columns(pl.Series("team_opp_pos", pos), pl.Series("team_opp_mpg", flat))


def _with_draft_columns(draft: pl.DataFrame) -> pl.DataFrame:
    if "org_type" not in draft.columns:
        draft = draft.with_columns(pl.lit(None, dtype=pl.Utf8).alias("org_type"))
    return draft


# --- history: one row per drafted player, features and first-year outcomes ----------------------
def class_outcomes(player_games: pl.DataFrame, team_games: pl.DataFrame, settings: ScoringSettings,
                   season: int) -> pl.DataFrame:
    pg = player_games.filter(pl.col("season") == season)
    tg = team_games.filter(pl.col("season") == season)
    ss = season_stats(pg, tg, settings).join(availability_stats(pg, tg), on=["player_id", "season"], how="left")
    return ss.select(
        "player_id",
        pl.col("gp").alias("y_gp"),
        pl.col("min").alias("y_min"),
        pl.col("mpg").alias("y_mpg"),
        pl.col("fp_pg").alias("y_fp_pg"),
        pl.col("fp").alias("y_fp"),
        (pl.col("fp_bonus") / pl.col("gp")).alias("y_bonus_pg"),
        (pl.col("gp") / pl.col("n_sched").clip(1)).clip(0, 1).alias("y_share"),
        *[(pl.col(c) / pl.col("min").clip(1e-9)).alias(f"y_rate_{c}") for c in COMPONENTS],
    )


def cutoff_roster(player_games: pl.DataFrame, draft: pl.DataFrame, season: int) -> pl.DataFrame:
    """Historical stand-in for the roster at the cut-off, as the backtest builds it: first
    team in the season, else last team in the two seasons before, else the draft team."""
    from draftai.features.build import roster_from_first_game

    prior = player_games.filter(pl.col("season").is_in([season - 1, season - 2])).sort("game_date")
    last_team = prior.group_by("player_id").agg(pl.col("team").last().alias("last_team"))
    first = roster_from_first_game(player_games, season).rename({"team": "first_team"})
    rookies = draft.filter(pl.col("draft_year") == season).select("player_id", pl.col("team").alias("draft_team"))
    ids = pl.concat([last_team.select("player_id"), rookies.select("player_id"), first.select("player_id")]).unique()
    return (ids.join(first, on="player_id", how="left").join(last_team, on="player_id", how="left")
            .join(rookies, on="player_id", how="left")
            .select("player_id", pl.coalesce("first_team", "last_team", "draft_team").alias("team"))
            .filter(pl.col("team").is_not_null()))


def class_rows(player_games: pl.DataFrame, team_games: pl.DataFrame, bio: pl.DataFrame, draft: pl.DataFrame,
               settings: ScoringSettings, years) -> pl.DataFrame:
    """Every drafted player of each class: draft-day features (as of opening night of his
    first season) and first-season outcomes (zero games for those who did not play)."""
    from draftai.features.asof import AsOf

    draft = _with_draft_columns(draft)
    seasons = set(team_games["season"].unique().to_list())
    frames = []
    for Y in years:
        cls = draft.filter(pl.col("draft_year") == Y)
        if cls.height == 0 or Y not in seasons:
            continue
        cut = team_games.filter(pl.col("season") == Y)["game_date"].min()
        view = AsOf.view(player_games, team_games, cut)
        roster = cutoff_roster(player_games, draft, Y)
        ctx = rookie_context(view.player_games, view.team_games, roster, bio, Y)
        base = cls.select("player_id", "player_name", pl.col("overall_pick").alias("draft_pick"), "org_type",
                          pl.col("team").alias("draft_team")).join(roster, on="player_id", how="left").with_columns(
            pl.coalesce("team", "draft_team").alias("team"))
        feats = rookie_features(base.join(bio_attributes(bio, Y), on="player_id", how="left"), ctx)
        team_gp = float(team_games.filter(pl.col("season") == Y).group_by("team").len()["len"].median())
        out = feats.join(class_outcomes(player_games, team_games, settings, Y), on="player_id", how="left").with_columns(
            pl.col("y_gp").fill_null(0), pl.col("y_min").fill_null(0.0), pl.col("y_share").fill_null(0.0),
            pl.col("y_fp").fill_null(0.0), pl.lit(team_gp).alias("y_team_gp"), pl.lit(Y, dtype=pl.Int32).alias("season"))
        frames.append(out)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("season", "draft_pick", "player_id")


# --- the model ------------------------------------------------------------------------------
@dataclass
class SignedRidge:
    """Ridge on standardised inputs, with some coefficients constrained to be <= 0.

    A violated constraint is pinned at zero (the feature is dropped and the rest refitted);
    for one constrained coefficient this is the exact constrained optimum, for several an
    active-set approximation."""

    features: list[str]
    alpha: float = 10.0
    non_positive: tuple[str, ...] = ()
    clip: tuple[float, float] = (0.0, np.inf)
    used: list[str] = field(default_factory=list)
    pipe: object | None = None

    def fit(self, df: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None) -> SignedRidge:
        used = list(self.features)
        while True:
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha))
            pipe.fit(df.select(used).to_numpy().astype(float), y, ridge__sample_weight=w)
            coef = dict(zip(used, pipe[-1].coef_))
            bad = [f for f in used if f in self.non_positive and coef[f] > 0]
            if not bad:
                break
            used.remove(max(bad, key=lambda f: coef[f]))
        self.used, self.pipe = used, pipe
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return np.clip(self.pipe.predict(df.select(self.used).to_numpy().astype(float)), *self.clip)

    def coefficients(self) -> dict[str, float]:
        """Per standard deviation of each input; dropped features at zero."""
        c = dict(zip(self.used, self.pipe[-1].coef_.tolist()))
        return {f: float(c.get(f, 0.0)) for f in self.features}


def _fp(rates: dict[str, np.ndarray], mpg: np.ndarray, bonus: np.ndarray, settings: ScoringSettings) -> np.ndarray:
    from draftai.models.components import linear_fp

    return linear_fp(rates, mpg, settings) + bonus


@dataclass
class RookieModel:
    alpha: float = 10.0
    k: int = 20
    adj_fp: tuple[float, float] = (0.0, 0.0)  # conformal corrections, from validation classes
    adj_share: tuple[float, float] = (0.0, 0.0)
    models: dict[str, SignedRidge] = field(default_factory=dict)
    # Comparables pool: the training rookies.
    pool_x: np.ndarray | None = None
    pool: dict = field(default_factory=dict)
    centre: np.ndarray | None = None
    scale: np.ndarray | None = None
    classes: list[int] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _clipped(self, df: pl.DataFrame) -> pl.DataFrame:
        """Inputs held to the training range: a linear model must not extrapolate (players
        returning from abroad after years away come through here too)."""
        return df.with_columns([pl.col(f).cast(pl.Float64).clip(lo, hi) for f, (lo, hi) in self.bounds.items()
                                if f in df.columns])

    def fit(self, rows: pl.DataFrame, settings: ScoringSettings) -> RookieModel:
        a = self.alpha
        self.bounds = {f: (float(rows[f].min()), float(rows[f].max())) for f in PLAYED_FEATURES
                       if rows[f].drop_nulls().len()}
        played = rows.filter(pl.col("y_gp") >= 1)
        rated = rows.filter(pl.col("y_min") >= MIN_RATE_MINUTES)
        regular = rows.filter(pl.col("y_gp") >= MIN_GP)
        m = {
            "share": SignedRidge(SHARE_FEATURES, a, NON_INCREASING, (0.0, 1.0)).fit(rows, rows["y_share"].to_numpy()),
            "mpg": SignedRidge(PLAYED_FEATURES, a, NON_INCREASING, (0.0, 40.0)).fit(
                played, played["y_mpg"].to_numpy(), played["y_gp"].to_numpy().astype(float)),
            "bonus": SignedRidge(PLAYED_FEATURES, a, NON_INCREASING, (0.0, 10.0)).fit(
                regular, regular["y_bonus_pg"].fill_null(0.0).to_numpy(), regular["y_gp"].to_numpy().astype(float)),
        }
        for c in COMPONENTS:
            m[f"rate_{c}"] = SignedRidge(PLAYED_FEATURES, a, NON_INCREASING if c in CONSTRAINED_RATES else (),
                                         (0.0, 2.0)).fit(rated, rated[f"y_rate_{c}"].to_numpy(),
                                                         rated["y_min"].to_numpy().astype(float))
        self.models = m
        self.classes = sorted(rows["season"].unique().to_list())
        # Comparables: standardised draft-day attributes, weighted by importance.
        feats = list(COMP_WEIGHTS)
        x = rows.select(feats).to_numpy().astype(float)
        self.centre = np.nanmedian(x, axis=0)
        self.scale = np.nanstd(x, axis=0) + 1e-9
        self.pool_x = self._comp_x(rows)
        pred = self._point(rows, settings)
        y_fp = rows["y_fp_pg"].fill_null(0.0).to_numpy()
        self.pool = {
            "player_id": rows["player_id"].to_numpy(),
            "name": rows["player_name"].to_list() if "player_name" in rows.columns else [None] * rows.height,
            "season": rows["season"].to_numpy(),
            "share": rows["y_share"].to_numpy(),
            "share_resid": rows["y_share"].to_numpy() - pred["share"],
            "regular": rows["y_gp"].to_numpy() >= MIN_GP,
            "fp_pg": y_fp,
            "fp_resid": y_fp - pred["fp_pg"],
        }
        return self

    def _comp_x(self, df: pl.DataFrame) -> np.ndarray:
        x = self._clipped(df).select(list(COMP_WEIGHTS)).to_numpy().astype(float)
        x = np.where(np.isnan(x), self.centre, x)
        return (x - self.centre) / self.scale * np.sqrt(np.array(list(COMP_WEIGHTS.values())))

    def _point(self, df: pl.DataFrame, settings: ScoringSettings) -> dict[str, np.ndarray]:
        df = self._clipped(df)
        p = {name: mdl.predict(df) for name, mdl in self.models.items()}
        rates = {c: p[f"rate_{c}"] for c in COMPONENTS}
        p["fp_pg"] = _fp(rates, p["mpg"], p["bonus"], settings)
        return p

    def comparables(self, df: pl.DataFrame) -> np.ndarray:
        """(rows, pool) squared distances to every training rookie."""
        q = self._comp_x(df)
        return ((q[:, None, :] - self.pool_x[None, :, :]) ** 2).sum(axis=2)

    def predict(self, df: pl.DataFrame, settings: ScoringSettings) -> pl.DataFrame:
        """Point projections, comparables-based ranges, donors and named comparables."""
        p = self._point(df, settings)
        d2 = self.comparables(df)
        P = self.pool
        reg = np.flatnonzero(P["regular"])
        fp_lo, fp_hi, sh_lo, sh_hi, comp_fp, comp_share, donors, names = [], [], [], [], [], [], [], []
        for i in range(df.height):
            near_all = np.argsort(d2[i], kind="stable")[: self.k]
            near_reg = reg[np.argsort(d2[i, reg], kind="stable")[: self.k]]
            r_fp = P["fp_resid"][near_reg]
            r_sh = P["share_resid"][near_all]
            fp_lo.append(float(np.quantile(r_fp, 0.1)) if r_fp.size else 0.0)
            fp_hi.append(float(np.quantile(r_fp, 0.9)) if r_fp.size else 0.0)
            sh_lo.append(float(np.quantile(r_sh, 0.1)))
            sh_hi.append(float(np.quantile(r_sh, 0.9)))
            comp_fp.append(float(P["fp_pg"][near_reg].mean()) if near_reg.size else None)
            comp_share.append(float(P["share"][near_all].mean()))
            donors.append([int(x) for x in P["player_id"][near_reg[:N_DONORS]]])
            names.append([f"{P['name'][j]} ({P['season'][j]}-{(P['season'][j] + 1) % 100:02d}: {P['fp_pg'][j]:.1f})"
                          for j in near_reg[:N_NAMED] if P["name"][j]])
        fp, share = p["fp_pg"], p["share"]
        return pl.DataFrame({
            "mpg_hat": p["mpg"], "share_hat": share,
            **{f"rate_{c}_hat": p[f"rate_{c}"] for c in COMPONENTS},
            "bonus_hat": p["bonus"], "fp_pg_hat": fp, "fp_direct_hat": fp,
            "fp_p10": np.clip(fp + np.array(fp_lo) - self.adj_fp[0], 0.0, None),
            "fp_p50": fp,
            "fp_p90": np.clip(fp + np.array(fp_hi) + self.adj_fp[1], 0.0, None),
            "share_p10": np.clip(share + np.array(sh_lo) - self.adj_share[0], 0.0, 1.0),
            "share_p50": share,
            "share_p90": np.clip(share + np.array(sh_hi) + self.adj_share[1], 0.0, 1.0),
            "comp_fp_pg": comp_fp, "comp_share": comp_share,
        }).with_columns(pl.Series("donor_ids", donors, dtype=pl.List(pl.Int64)),
                        pl.Series("comps", names, dtype=pl.List(pl.Utf8)))


def rookie_model_projections(feats: pl.DataFrame, model: RookieModel, q_league: float,
                             settings: ScoringSettings) -> pl.DataFrame:
    """The rookie model's rows in the projections table's shape (as ``rookie_projections``)."""
    pred = model.predict(feats, settings)
    team = feats["team"] if "team" in feats.columns else pl.Series("team", [None] * feats.height, dtype=pl.Utf8)
    return pl.concat([feats.select("player_id", "season"), pred], how="horizontal").with_columns(
        pl.lit(q_league).alias("q_short_hat"),
        pl.Series("team_now", team),
        feats["age"].cast(pl.Float64).alias("age"),
        pl.lit(None, dtype=pl.Float64).alias("fp_pg_1"),
        pl.lit(None, dtype=pl.Float64).alias("share_1"),
        pl.lit(None, dtype=pl.Float64).alias("gp_1"),
        pl.lit(None, dtype=pl.Float64).alias("mpg_1"),
        pl.lit(None, dtype=pl.Float64).alias("fp_pg_w"),
        pl.lit(None, dtype=pl.Float64).alias("share_w"),
        pl.lit(True).alias("rookie"),
        pl.lit(q_league).alias("q_league"),
        pl.lit("model").alias("rookie_source"),
        feats["draft_pick"].cast(pl.Float64).alias("draft_pick"),
        feats["team_opp_pos"].alias("rookie_team_opp"),
    )


def rookie_projections(
    rookies: pl.DataFrame, baseline: pl.DataFrame, q_league: float, settings: ScoringSettings
) -> pl.DataFrame:
    """Draft-slot baseline: each rookie gets his pick bucket's first-year outcomes."""
    from draftai.models.components import linear_fp

    known = set(baseline["bucket"].to_list())
    r = rookies.with_columns(baseline_bucket(pl.col("draft_pick"), known).alias("bucket")).join(
        baseline, on="bucket", how="left")
    rates = {c: r[f"rate_{c}"].fill_null(0).to_numpy() for c in COMPONENTS}
    mpg = r["mpg"].fill_null(10.0).to_numpy()
    lin = linear_fp(rates, mpg, settings) + r["bonus_pg"].fill_null(0).to_numpy()
    return r.select(
        "player_id", "season",
        pl.col("mpg").alias("mpg_hat"),
        pl.col("share").alias("share_hat"),
        *[pl.col(f"rate_{c}").alias(f"rate_{c}_hat") for c in COMPONENTS],
        pl.col("bonus_pg").alias("bonus_hat"),
        pl.Series("fp_pg_hat", lin),
        pl.Series("fp_direct_hat", lin),
        pl.col("fp_pg_p10").alias("fp_p10"), pl.col("fp_pg_p50").alias("fp_p50"), pl.col("fp_pg_p90").alias("fp_p90"),
        pl.col("share_p10"), pl.col("share").alias("share_p50"), pl.col("share_p90"),
        pl.lit(q_league).alias("q_short_hat"),
        pl.coalesce("team", "draft_team").alias("team_now") if "team" in r.columns else pl.col("draft_team").alias("team_now"),
        pl.lit(None, dtype=pl.Float64).alias("age"),
        pl.lit(None, dtype=pl.Float64).alias("fp_pg_1"),
        pl.lit(None, dtype=pl.Float64).alias("share_1"),
        pl.lit(None, dtype=pl.Float64).alias("gp_1"),
        pl.lit(None, dtype=pl.Float64).alias("mpg_1"),
        pl.lit(None, dtype=pl.Float64).alias("fp_pg_w"),
        pl.lit(None, dtype=pl.Float64).alias("share_w"),
        pl.lit(True).alias("rookie"),
        pl.lit(q_league).alias("q_league"),
        pl.col("donor_ids"),
        pl.col("bucket"),
        pl.lit("baseline").alias("rookie_source"),
        pl.col("draft_pick").cast(pl.Float64),
    )


def baseline_predictions(rows: pl.DataFrame, baseline: pl.DataFrame, settings: ScoringSettings) -> dict[str, np.ndarray]:
    """The draft-slot baseline's point predictions and ranges for class rows."""
    from draftai.models.components import linear_fp

    known = set(baseline["bucket"].to_list())
    r = rows.select("player_id", "draft_pick").with_columns(baseline_bucket(pl.col("draft_pick"), known).alias("bucket")) \
        .join(baseline, on="bucket", how="left", maintain_order="left")
    rates = {c: r[f"rate_{c}"].fill_null(0).to_numpy() for c in COMPONENTS}
    fp = linear_fp(rates, r["mpg"].fill_null(10.0).to_numpy(), settings) + r["bonus_pg"].fill_null(0).to_numpy()
    return {"fp": fp, "share": r["share"].fill_null(0.0).to_numpy(),
            "fp_iv": np.column_stack([r["fp_pg_p10"].fill_null(0).to_numpy(), r["fp_pg_p90"].fill_null(0).to_numpy()]),
            "share_iv": np.column_stack([r["share_p10"].fill_null(0).to_numpy(), r["share_p90"].fill_null(0).to_numpy()])}


# --- evaluation (rolling origin over draft classes) --------------------------------------------
@dataclass
class RookieEvaluation:
    metrics: pl.DataFrame  # split, season, model, metric columns
    summary: dict
    coefficients: dict[str, dict[str, float]]
    model_test: RookieModel
    model_final: RookieModel
    rows: pl.DataFrame


def _conformal(lo_hi: np.ndarray, y: np.ndarray, level: float = 0.8) -> tuple[float, float]:
    """Additive corrections to an interval's ends so it covers ``level`` (each tail half)."""
    n = len(y)
    if n == 0:
        return 0.0, 0.0
    tail = (1 - level) / 2
    q = min(1.0, (1 - tail) * (1 + 1 / n))
    return float(np.quantile(lo_hi[:, 0] - y, q)), float(np.quantile(y - lo_hi[:, 1], q))


def _metrics(split: str, season: int, model: str, d: pl.DataFrame, fp: np.ndarray, share: np.ndarray,
             fp_iv: np.ndarray | None, share_iv: np.ndarray | None) -> dict:
    from draftai.models.evaluate import mae, spearman

    reg = d["y_gp"].to_numpy() >= MIN_GP
    y_fp = d["y_fp_pg"].fill_null(0.0).to_numpy()
    y_sh = d["y_share"].to_numpy()
    y_tot = d["y_fp"].to_numpy()
    tot = fp * share * d["y_team_gp"].to_numpy()
    r1 = d["draft_pick"].fill_null(UNDRAFTED_PICK).to_numpy() <= 30

    def cov(iv, y, m):
        return float(np.mean((y[m] >= iv[m, 0]) & (y[m] <= iv[m, 1]))) if iv is not None and m.any() else None

    return {
        "split": split, "season": season, "model": model, "picks": d.height, "regular": int(reg.sum()),
        "fpg_mae": mae(fp[reg], y_fp[reg]) if reg.any() else None,
        "fpg_bias": float(np.mean(fp[reg] - y_fp[reg])) if reg.any() else None,
        "share_mae": mae(share, y_sh),
        "season_mae": mae(tot, y_tot),
        "season_mae_r1": mae(tot[r1], y_tot[r1]) if r1.any() else None,
        "season_spearman": spearman(tot, y_tot),
        "fpg_cov80": cov(fp_iv, y_fp, reg),
        "share_cov80": cov(share_iv, y_sh, np.ones(d.height, dtype=bool)),
    }


def evaluate_rookies(player_games: pl.DataFrame, team_games: pl.DataFrame, bio: pl.DataFrame, draft: pl.DataFrame,
                     settings: ScoringSettings, validation: list[int], test: int, alpha: float = 10.0, k: int = 20,
                     log=print) -> RookieEvaluation:
    """Rolling origin by draft class: for each validation class V, fit on the classes
    before V and project V; conformal corrections from those folds; the test class once;
    then the final model on every class through the test class."""
    from draftai.models.evaluate import paired_bootstrap

    first = int(draft["draft_year"].min())
    rows = class_rows(player_games, team_games, bio, draft, settings, range(first, test + 1))
    folds = []
    for V in validation:
        train, hold = rows.filter(pl.col("season") < V), rows.filter(pl.col("season") == V)
        m = RookieModel(alpha=alpha, k=k).fit(train, settings)
        folds.append((V, hold, m.predict(hold, settings), m))
    log(f"rookie model: validation classes {validation[0]}-{validation[-1]}")

    def ivs(pred: pl.DataFrame, kind: str) -> np.ndarray:
        return np.column_stack([pred[f"{kind}_p10"].to_numpy(), pred[f"{kind}_p90"].to_numpy()])

    def regular(h: pl.DataFrame) -> np.ndarray:
        return h["y_gp"].to_numpy() >= MIN_GP

    def pooled(idx: list[int]) -> tuple[tuple[float, float], tuple[float, float]]:
        fp_iv = np.vstack([ivs(folds[j][2], "fp")[regular(folds[j][1])] for j in idx])
        fp_y = np.concatenate([folds[j][1]["y_fp_pg"].to_numpy()[regular(folds[j][1])] for j in idx])
        sh_iv = np.vstack([ivs(folds[j][2], "share") for j in idx])
        sh_y = np.concatenate([folds[j][1]["y_share"].to_numpy() for j in idx])
        return _conformal(fp_iv, fp_y), _conformal(sh_iv, sh_y)

    def shifted(iv: np.ndarray, adj: tuple[float, float], hi: float) -> np.ndarray:
        return np.column_stack([np.clip(iv[:, 0] - adj[0], 0, hi), np.clip(iv[:, 1] + adj[1], 0, hi)])

    out = []
    for i, (V, hold, pred, _) in enumerate(folds):
        others = [j for j in range(len(folds)) if j != i] or [i]
        a_fp, a_sh = pooled(others)  # cross-conformal: corrections from the other classes
        base = baseline_predictions(hold, rookie_baseline(player_games, team_games, draft, settings, V), settings)
        out.append(_metrics("validation", V, "rookie_model", hold, pred["fp_pg_hat"].to_numpy(),
                            pred["share_hat"].to_numpy(), shifted(ivs(pred, "fp"), a_fp, 90.0),
                            shifted(ivs(pred, "share"), a_sh, 1.0)))
        out.append(_metrics("validation", V, "rookie_model_uncalibrated", hold, pred["fp_pg_hat"].to_numpy(),
                            pred["share_hat"].to_numpy(), ivs(pred, "fp"), ivs(pred, "share")))
        out.append(_metrics("validation", V, "comparables_mean", hold, pred["comp_fp_pg"].fill_null(0.0).to_numpy(),
                            pred["comp_share"].to_numpy(), None, None))
        out.append(_metrics("validation", V, "draft_slot_baseline", hold, base["fp"], base["share"], base["fp_iv"],
                            base["share_iv"]))
    adj_fp, adj_share = pooled(list(range(len(folds))))

    # Test class, once, with everything frozen.
    train, hold = rows.filter(pl.col("season") < test), rows.filter(pl.col("season") == test)
    m_test = RookieModel(alpha=alpha, k=k, adj_fp=adj_fp, adj_share=adj_share).fit(train, settings)
    pred = m_test.predict(hold, settings)
    base = baseline_predictions(hold, rookie_baseline(player_games, team_games, draft, settings, test), settings)
    out.append(_metrics("test", test, "rookie_model", hold, pred["fp_pg_hat"].to_numpy(), pred["share_hat"].to_numpy(),
                        ivs(pred, "fp"), ivs(pred, "share")))
    out.append(_metrics("test", test, "comparables_mean", hold, pred["comp_fp_pg"].fill_null(0.0).to_numpy(),
                        pred["comp_share"].to_numpy(), None, None))
    out.append(_metrics("test", test, "draft_slot_baseline", hold, base["fp"], base["share"], base["fp_iv"],
                        base["share_iv"]))
    m_final = RookieModel(alpha=alpha, k=k, adj_fp=adj_fp, adj_share=adj_share).fit(rows, settings)
    log(f"rookie model: test class {test} evaluated; final model on classes {first}-{test}")

    metrics = pl.DataFrame(out, infer_schema_length=None)

    def season_err(V: int, hold: pl.DataFrame, pred_fp: np.ndarray, pred_sh: np.ndarray) -> np.ndarray:
        return pred_fp * pred_sh * hold["y_team_gp"].to_numpy() - hold["y_fp"].to_numpy()

    gains = []
    for V, rows_v, pred_v, _ in [*folds, (test, hold, pred, m_test)]:
        base = baseline_predictions(rows_v, rookie_baseline(player_games, team_games, draft, settings, V), settings)
        e_b = season_err(V, rows_v, base["fp"], base["share"])
        e_m = season_err(V, rows_v, pred_v["fp_pg_hat"].to_numpy(), pred_v["share_hat"].to_numpy())
        diff, lo, hi = paired_bootstrap(e_b, e_m, seed=V)
        gains.append({"season": V, "split": "test" if V == test else "validation", "season_fp_gain": diff,
                      "lo": lo, "hi": hi})
    val_gains = [g for g in gains if g["split"] == "validation"]
    summary = {
        "validation_classes_better_than_baseline": sum(g["season_fp_gain"] > 0 for g in val_gains),
        "validation_classes": len(val_gains),
        "season_fp_gain_vs_baseline": gains,
        "conformal_adj_fp": list(adj_fp),
        "conformal_adj_share": list(adj_share),
        "classes": [first, test],
        "k": k, "alpha": alpha,
    }
    return RookieEvaluation(metrics, summary, {n: s.coefficients() for n, s in m_final.models.items()},
                            m_test, m_final, rows)


def write_rookie_report(ev: RookieEvaluation, out_dir) -> object:
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.metrics.write_parquet(out_dir / "rookie_metrics.parquet")
    cols = ["model", "picks", "regular", "fpg_mae", "fpg_bias", "share_mae", "season_mae", "season_mae_r1",
            "season_spearman", "fpg_cov80", "share_cov80"]
    val = (ev.metrics.filter(pl.col("split") == "validation").group_by("model")
           .agg(pl.col("picks", "regular").sum(), *[pl.col(c).mean() for c in cols[3:]]).sort("season_mae"))
    test = ev.metrics.filter(pl.col("split") == "test").select(cols).sort("season_mae")
    per = ev.metrics.filter(pl.col("split") == "validation").select("season", "model", "fpg_mae", "share_mae",
                                                                    "season_mae").sort("season", "model")
    gains = pl.DataFrame(ev.summary["season_fp_gain_vs_baseline"])
    coef = pl.DataFrame([{"target": t, **{f: round(v, 3) for f, v in c.items()}} for t, c in ev.coefficients.items()],
                        infer_schema_length=None)

    def md(df: pl.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        for row in df.iter_rows():
            lines.append("| " + " | ".join("" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
                                           for v in row) + " |")
        return "\n".join(lines)

    text = [
        "# Rookie projections",
        "",
        "Every drafted player of each class, projected as of opening night of his first season. `season_mae` is",
        "the error in first-season fantasy points (FP/G x games share x team games; zero for players who never",
        "played), over every pick; `season_mae_r1` over first-round picks. FP/G metrics use players with 10+",
        "games. Validation is rolling origin by draft class (fit on earlier classes only); the test class is",
        "evaluated once. Intervals are 80% (P10-P90); the rookie model's are conformalised on the validation",
        "classes (cross-conformal in the validation rows).",
        "",
        "## Validation classes (mean over classes)",
        md(val),
        "",
        "## Validation per class",
        md(per),
        "",
        "## Test class (evaluated once)",
        md(test),
        "",
        "## Season-points gain over the draft-slot baseline",
        "Baseline absolute error minus the model's, per pick, with a paired 95% bootstrap interval.",
        md(gains),
        "",
        "## Coefficients of the final model (per standard deviation of each input)",
        "Zero where a draft-slot constraint (later picks never project more) pinned a coefficient.",
        md(coef),
        "",
        "```json",
        json.dumps(ev.summary, indent=2),
        "```",
    ]
    path = out_dir / "rookie_report.md"
    path.write_text("\n".join(text))
    (out_dir / "rookie_summary.json").write_text(json.dumps(ev.summary, indent=2))
    return path
