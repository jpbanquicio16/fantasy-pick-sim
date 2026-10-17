"""Team minutes budget: a team's projections have to fit in the minutes that exist.

Each player is projected on his own, so nothing stops a team's projections summing to 265
minutes a game when only about 241 exist (48 x 5, plus overtime). After the minutes
model, each team's expected minutes (minutes per game times share of games played) are
pulled toward the league's minutes per team game:

    S_T   = sum_i share_i * m_i                                 (rookies included)
    E_T   = strength * (S_T - target)                           minutes to remove (< 0: add)
    dm_i  = -E_T * e(m_i) / sum_j share_j * e(m_j)             e(m) = m * max(1 - m/40, 0.05)

A new arrival or a teammate returning from injury raises S_T, so everyone else's minutes
fall: that is how a reduced role reaches the projection.

Two choices differ from a plain proportional rescale to exactly 241, and both come from
rolling validation (train.py reports the grid):

* **Elasticity.** The minutes come mostly out of rotation players in the middle of the
  range (``features.roles.elasticity``); a 36-minute starter barely moves. Proportional
  rescaling cut stars' minutes for the noise in their teams' bench projections.
* **Strength.** Team sums of separately projected players are noisy (the spread across
  teams is about 15 minutes, and almost uncorrelated with the minutes those players went
  on to play), so pulling all the way to the target over-corrects. The strength is chosen
  on validation seasons and saved with the models.

Known changes are different: a minutes override (preseason news entered by hand) is
absorbed in full by the player's teammates, weighted toward his positions
(``apply_overrides``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from draftai.features.build import COMPONENTS
from draftai.features.roles import elasticity
from draftai.models.components import linear_fp
from draftai.scoring import ScoringSettings

STRENGTH_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CAP = 38.0  # nobody is pushed above this (or above his own projection, if higher)
OTHER_POSITION = 0.3  # an override's minutes come 0.3x as much from teammates at other positions


def allocate(team: np.ndarray, mpg: np.ndarray, share: np.ndarray, target: float, strength: float,
             extra_e: dict | None = None, extra_w: dict | None = None) -> np.ndarray:
    """Budgeted minutes per game for every row, team by team.

    ``extra_e`` and ``extra_w`` (team -> value) add players who share the budget but are
    not rows here (the rookies, when the training evaluation scores veterans only): their
    expected minutes and their elasticity weight.
    """
    m = np.asarray(mpg, dtype=float)
    s = np.clip(np.nan_to_num(np.asarray(share, dtype=float)), 0.0, 1.0)
    out = m.copy()
    if strength <= 0.0:
        return out
    w = s * elasticity(m)
    team = np.asarray(team, dtype=object)
    for t in np.unique(team[np.array([x is not None for x in team])]):
        i = team == t
        S = float((s[i] * m[i]).sum() + (extra_e or {}).get(t, 0.0))
        W = float(w[i].sum() + (extra_w or {}).get(t, 0.0))
        if W <= 1e-9:
            continue
        E = strength * (S - target)
        out[i] = m[i] - E * elasticity(m[i]) / W
    return np.clip(out, 0.0, np.maximum(CAP, m))


def reprice(proj: pl.DataFrame, new_mpg: np.ndarray, settings: ScoringSettings) -> pl.DataFrame:
    """Carry a change in minutes through to every derived projection column.

    Per-minute rates stay; linear FP moves with minutes, the bonus expectation and the
    direct FP/G cross-check move in proportion, and the season FP/G quantiles keep their
    shape around the new point projection.
    """
    old = proj["mpg_hat"].to_numpy().astype(float)
    ratio = np.where(old > 1e-6, new_mpg / np.maximum(old, 1e-6), 1.0)
    rates = {c: proj[f"rate_{c}_hat"].fill_null(0.0).to_numpy() for c in COMPONENTS}
    bonus = proj["bonus_hat"].fill_null(0.0).to_numpy() * ratio
    fp = linear_fp(rates, new_mpg, settings) + bonus
    fp_old = proj["fp_pg_hat"].fill_null(0.0).to_numpy()
    fp_ratio = np.where(fp_old > 1e-6, fp / np.maximum(fp_old, 1e-6), 1.0)
    cols = [pl.Series("mpg_hat", new_mpg), pl.Series("bonus_hat", bonus), pl.Series("fp_pg_hat", fp)]
    if "fp_direct_hat" in proj.columns:
        cols.append(pl.Series("fp_direct_hat", proj["fp_direct_hat"].fill_null(0.0).to_numpy() * ratio))
    for q in ("fp_p10", "fp_p50", "fp_p90"):
        if q in proj.columns:
            cols.append(pl.Series(q, proj[q].fill_null(0.0).to_numpy() * fp_ratio))
    return proj.with_columns(cols)


def apply_budget(proj: pl.DataFrame, target: float, strength: float, settings: ScoringSettings) -> pl.DataFrame:
    """The team minutes budget on a projections table (veterans and rookies together).

    Keeps the model's minutes as ``mpg_model`` and adds ``team_minutes`` (the team's
    expected minutes per game before the budget), ``team_minutes_target`` and
    ``budget_delta`` (budgeted minus model minutes per game).
    """
    mpg = proj["mpg_hat"].fill_null(0.0).to_numpy().astype(float)
    share = proj["share_hat"].fill_null(0.0).to_numpy().astype(float)
    team = np.array(proj["team_now"].to_list(), dtype=object)
    new = allocate(team, mpg, share, target, strength)
    out = proj.with_columns(
        pl.Series("mpg_model", mpg),
        (pl.col("mpg_hat").fill_null(0.0) * pl.col("share_hat").fill_null(0.0)).sum().over("team_now")
        .alias("team_minutes"),
        pl.lit(target).alias("team_minutes_target"),
        pl.lit(strength).alias("budget_strength"),
    )
    return reprice(out, new, settings).with_columns((pl.col("mpg_hat") - pl.col("mpg_model")).alias("budget_delta"))


def _same_position(a: list[str] | None, b: list[str] | None) -> bool:
    return bool(a) and bool(b) and bool(set(a) & set(b))


def apply_overrides(proj: pl.DataFrame, overrides: pl.DataFrame, settings: ScoringSettings,
                    id_col: str = "sleeper_id") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Minutes overrides for preseason news (a starting job, a move to the bench).

    ``overrides`` has ``id_col`` and ``mpg`` (and optionally ``note``). Each overridden
    player is pinned to his minutes; the change in his expected minutes (times his games
    share) is taken from or given to his teammates in full, by elasticity, weighted toward
    teammates who share a position with him (``positions``; others count 0.3x). A team's
    expected minutes are therefore the same before and after.

    Returns the repriced projections (``minutes_override``, ``minutes_note``,
    ``override_delta`` added) and a log of what changed.
    """
    o = overrides.select(pl.col(id_col).cast(pl.Utf8), pl.col("mpg").cast(pl.Float64),
                         pl.col("note").cast(pl.Utf8) if "note" in overrides.columns else pl.lit(None, pl.Utf8).alias("note"))
    o = o.filter(pl.col("mpg").is_not_null()).unique(id_col, keep="last")
    p = proj.with_columns(pl.col(id_col).cast(pl.Utf8))
    pinned_mpg = dict(zip(o[id_col].to_list(), o["mpg"].to_list()))
    notes = dict(zip(o[id_col].to_list(), o["note"].to_list()))
    ids = p[id_col].to_list()
    mpg = p["mpg_hat"].fill_null(0.0).to_numpy().astype(float)
    share = np.clip(p["share_hat"].fill_null(0.0).to_numpy().astype(float), 0.0, 1.0)
    team = p["team_now"].to_list()
    positions = p["positions"].to_list() if "positions" in p.columns else [None] * p.height
    pinned = np.array([i in pinned_mpg for i in ids])
    new = mpg.copy()
    log = []
    by_id = {i: k for k, i in enumerate(ids)}
    for sid, target_mpg in pinned_mpg.items():
        k = by_id.get(sid)
        if k is None:
            log.append({"sleeper_id": sid, "name": None, "team": None, "model_mpg": None, "before_mpg": None,
                        "override_mpg": target_mpg, "teammates": 0, "note": notes.get(sid), "status": "not on the board"})
            continue
        delta_e = share[k] * (target_mpg - new[k])
        new[k] = target_mpg
        mates = np.array([team[j] == team[k] and not pinned[j] and j != k for j in range(len(ids))])
        if mates.any() and abs(delta_e) > 1e-9:
            sim = np.array([1.0 if _same_position(positions[j], positions[k]) else OTHER_POSITION for j in range(len(ids))])
            w = np.where(mates, share * elasticity(new) * sim, 0.0)
            if w.sum() > 1e-9:
                # Expected minutes: share_j * dm_j = -delta_e * w_j / sum(w).
                dm = -delta_e * w / w.sum() / np.maximum(share, 1e-6)
                new = np.where(mates, np.clip(new + dm, 0.0, np.maximum(CAP, new)), new)
        log.append({"sleeper_id": sid, "name": p["name"][k] if "name" in p.columns else None, "team": team[k],
                    "model_mpg": float(p["mpg_model"][k]) if "mpg_model" in p.columns else float(mpg[k]),
                    "before_mpg": float(mpg[k]), "override_mpg": target_mpg, "teammates": int(mates.sum()),
                    "note": notes.get(sid), "status": "applied"})
    out = reprice(p, new, settings).with_columns(
        pl.Series("minutes_override", pinned),
        pl.Series("minutes_note", [notes.get(i) for i in ids], dtype=pl.Utf8),
        pl.Series("override_delta", new - mpg),
    )
    schema = {"sleeper_id": pl.Utf8, "name": pl.Utf8, "team": pl.Utf8, "model_mpg": pl.Float64,
              "before_mpg": pl.Float64, "override_mpg": pl.Float64, "teammates": pl.Int64, "note": pl.Utf8,
              "status": pl.Utf8}
    return out, pl.DataFrame(log, schema=schema)
