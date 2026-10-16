"""The Lock-In headline experiment (V2 gate).

On a held-out season, with projections made before it started (models trained only on
earlier seasons), run the optimal lock rule on every player-week of real game logs and
test whether Lock-In value predicts the realised locked score better than mean FP/G.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from draftai.features.asof import season_cutoff
from draftai.features.build import availability_stats, league_b2b_multiplier, roster_from_first_game
from draftai.models.evaluate import draftable_mask
from draftai.models.project import season_projections
from draftai.scoring import ScoringSettings, score_frame
from draftai.value.games import DonorBank, GameProjection, GameSimulator, simulated_stats
from draftai.value.lockin import EmpiricalScores, kappa, lockin_value, realised_lock, thresholds
from draftai.value.schedule import add_weeks
from draftai.value.weekly import Availability


def run(player_games: pl.DataFrame, team_games: pl.DataFrame, bio: pl.DataFrame, draft: pl.DataFrame, models,
        settings: ScoringSettings, season: int, game_samples: int = 4000, top_n: int = 200, seed: int = 0) -> dict:
    cut = season_cutoff(team_games, season)
    roster = roster_from_first_game(player_games, season)
    proj = season_projections(player_games, team_games, bio, draft, models, settings, season, cut, roster)
    proj = proj.filter(~pl.col("rookie"))
    # Draftable: top N by the same pre-season proxy as the projection report.
    from draftai.features.asof import AsOf
    from draftai.features.build import build_features

    feats = build_features(AsOf.view(player_games, team_games, cut), season, roster, bio, draft, settings)
    top_ids = feats.filter(pl.Series(draftable_mask(feats, top_n))).select("player_id")
    proj = proj.join(top_ids, on="player_id", how="semi")

    hist = player_games.filter(pl.col("game_date") < cut)
    b2b = league_b2b_multiplier(availability_stats(hist.filter(pl.col("season") >= season - 3),
                                                   team_games.filter((pl.col("season") >= season - 3) & (pl.col("game_date") < cut))))
    sim = GameSimulator(DonorBank.build(hist, season, simulated_stats(settings)), settings, seed=seed)

    sched = add_weeks(team_games.filter(pl.col("season") == season).select("game_id", "game_date", "team", "opp", "home", "season"))
    actual = score_frame(player_games.filter(pl.col("season") == season), settings).select("player_id", "game_id", "fp")
    rows = []
    for r in proj.iter_rows(named=True):
        pid = r["player_id"]
        rates = {c: r[f"rate_{c}_hat"] for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")}
        dist = EmpiricalScores(sim.fp_samples(GameProjection(pid, r["mpg_hat"], rates), game_samples))
        avail = Availability.from_share(r["share_hat"], r["q_short_hat"], b2b)
        mine = actual.filter(pl.col("player_id") == pid)
        teams = player_games.filter((pl.col("season") == season) & (pl.col("player_id") == pid)).select("game_id", "team")
        # The player's team schedule for the weeks he was on each team.
        team_sched = sched.join(teams.select("team").unique(), on="team", how="semi")
        played = dict(zip(mine["game_id"].to_list(), mine["fp"].to_list()))
        stints = teams.join(sched.select("game_id", "game_date"), on="game_id").group_by("team").agg(
            pl.col("game_date").min().alias("a"), pl.col("game_date").max().alias("b"))
        team_sched = team_sched.join(stints, on="team").filter(pl.col("game_date").is_between("a", "b"))
        for (week,), g in team_sched.sort("game_date").group_by(["week"], maintain_order=True):
            scores = [played.get(gid) for gid in g["game_id"].to_list()]
            if all(s is None for s in scores):
                continue  # out the whole week: a lineup decision, not a lock decision
            pattern = tuple(g["b2b2"].to_list())
            q = avail.game_q(pattern)
            li = lockin_value(dist, q)
            thr = thresholds(dist, q)
            rows.append({
                "player_id": pid, "week": int(week), "n_games": len(scores), "n_played": sum(s is not None for s in scores),
                "pred_mean": dist.mean, "pred_lockin": li, "pred_normal": dist.mean + kappa(len(scores)) * dist.sd,
                "realised": realised_lock(scores, thr),
                "hindsight": max(s for s in scores if s is not None),
            })
    df = pl.DataFrame(rows)
    return summarise(df, season)


def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return {"mae": float(np.mean(np.abs(y - x))), "r2": float(1 - resid.var() / y.var()),
            "corr": float(np.corrcoef(x, y)[0, 1]), "bias": float(np.mean(x - y))}


def summarise(df: pl.DataFrame, season: int, reps: int = 1000, seed: int = 0) -> dict:
    y = df["realised"].to_numpy()
    out = {"season": season, "player_weeks": df.height, "players": df["player_id"].n_unique()}
    for name in ("pred_mean", "pred_lockin", "pred_normal"):
        out[name] = _fit(df[name].to_numpy(), y)
    # Paired bootstrap over players: MAE(mean) - MAE(lockin), positive = Lock-In better.
    rng = np.random.default_rng(seed)
    players = df["player_id"].unique().to_numpy()
    per = df.group_by("player_id").agg(
        (pl.col("pred_mean") - pl.col("realised")).abs().sum().alias("e_mean"),
        (pl.col("pred_lockin") - pl.col("realised")).abs().sum().alias("e_lock"),
        pl.len().alias("n"),
    )
    e_m, e_l, n = per["e_mean"].to_numpy(), per["e_lock"].to_numpy(), per["n"].to_numpy()
    boots = []
    for _ in range(reps):
        i = rng.integers(0, len(players), len(players))
        boots.append((e_m[i].sum() - e_l[i].sum()) / n[i].sum())
    out["mae_gain_lockin_vs_mean"] = [float((e_m.sum() - e_l.sum()) / n.sum()), float(np.quantile(boots, 0.025)),
                                      float(np.quantile(boots, 0.975))]
    by_n = (
        df.group_by("n_games").agg(
            pl.len().alias("weeks"), pl.col("realised").mean(), pl.col("pred_lockin").mean(),
            pl.col("pred_mean").mean(), pl.col("hindsight").mean(),
        ).sort("n_games")
    )
    out["by_games"] = by_n.to_dicts()
    # Weeks he played every game isolate the lock decision from mid-week injuries.
    full = df.filter(pl.col("n_played") == pl.col("n_games"))
    out["full_weeks"] = {
        "player_weeks": full.height,
        "bias_lockin": float((full["pred_lockin"] - full["realised"]).mean()),
        "bias_mean": float((full["pred_mean"] - full["realised"]).mean()),
        "mae_lockin": float((full["pred_lockin"] - full["realised"]).abs().mean()),
        "mae_mean": float((full["pred_mean"] - full["realised"]).abs().mean()),
    }
    part = df.filter(pl.col("n_played") < pl.col("n_games"))
    out["partial_weeks"] = {"player_weeks": part.height,
                            "bias_lockin": float((part["pred_lockin"] - part["realised"]).mean()) if part.height else None}
    out["gate_v2_lockin_passed"] = out["mae_gain_lockin_vs_mean"][1] > 0 and out["pred_lockin"]["r2"] >= out["pred_mean"]["r2"]
    return out


def write(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lockin_experiment.json").write_text(json.dumps(result, indent=2))
    lines = [
        f"# Lock-In headline experiment: {result['season']} (held out)",
        "",
        f"{result['player_weeks']} player-weeks, {result['players']} draftable players. Projections were made",
        "before the season from models trained on earlier seasons. Each week the optimal lock rule",
        "(lock game k iff its score beats the continuation value) was applied to the real games.",
        "",
        "| predictor | MAE vs realised | bias | corr | R2 (linear fit) |",
        "|---|---|---|---|---|",
    ]
    for k, label in (("pred_mean", "mean FP/G"), ("pred_lockin", "Lock-In value (simulated games)"),
                     ("pred_normal", "Lock-In value (normal approx.)")):
        m = result[k]
        lines.append(f"| {label} | {m['mae']:.2f} | {m['bias']:+.2f} | {m['corr']:.3f} | {m['r2']:.3f} |")
    g, lo, hi = result["mae_gain_lockin_vs_mean"]
    lines += ["", f"MAE gain of Lock-In value over mean FP/G: {g:+.2f} FP per slot-week (95% CI {lo:+.2f} to {hi:+.2f}).",
              "", "| games in week | weeks | realised locked | predicted Lock-In | predicted mean | hindsight best |",
              "|---|---|---|---|---|---|"]
    for r in result["by_games"]:
        lines.append(f"| {r['n_games']} | {r['weeks']} | {r['realised']:.2f} | {r['pred_lockin']:.2f} | {r['pred_mean']:.2f} | {r['hindsight']:.2f} |")
    f, pw = result["full_weeks"], result["partial_weeks"]
    lines += [
        "",
        f"Weeks in which he played every game ({f['player_weeks']}): Lock-In bias {f['bias_lockin']:+.2f}, "
        f"mean-FP/G bias {f['bias_mean']:+.2f}; MAE {f['mae_lockin']:.2f} vs {f['mae_mean']:.2f}.",
        f"Weeks with a missed game ({pw['player_weeks']}): Lock-In bias {pw['bias_lockin']:+.2f}. These are mostly",
        "injury onsets and returns; the value layer treats availability week by week, so it is",
        "optimistic for them. That is the main known limitation of the weekly value.",
        "",
        f"Gate passed: {result['gate_v2_lockin_passed']}",
    ]
    p = out_dir / "lockin_experiment.md"
    p.write_text("\n".join(lines))
    return p
