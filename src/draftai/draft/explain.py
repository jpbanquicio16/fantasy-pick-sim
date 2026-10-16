"""Explanation cards: the decision's own components, not a post-hoc story.

With Monte Carlo rollouts the card leads with confidence, P(A's roster beats B's) across
the shared simulated worlds, next to the expected edge in fantasy points per week; a 55%
edge worth 0.2 points and one worth 4 points are different situations. Without rollouts
it reports the greedy recommender's expected edge only.
"""

from __future__ import annotations

import polars as pl

from draftai.draft.recommender import Recommendation
from draftai.scoring import ScoringSettings
from draftai.scoring.engine import component_contributions


def stocks_share(row: dict, settings: ScoringSettings) -> float:
    """Share of linear fantasy points from steals and blocks."""
    exp = {c: (row.get(f"rate_{c}_hat") or 0.0) * (row.get("mpg_hat") or 0.0)
           for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")}
    contrib = component_contributions(exp, settings)
    pos = sum(v for v in contrib.values() if v > 0)
    return (contrib.get("stl", 0.0) + contrib.get("blk", 0.0)) / pos if pos else 0.0


def card(rec: Recommendation, board_players: pl.DataFrame, settings: ScoringSettings, rollout=None) -> dict:
    """``rollout`` (a RolloutResult for the same state) reorders the candidates and adds
    confidence; the greedy recommender still supplies survival and value components."""
    by_row = {c.row: c for c in rec.candidates}
    confidence = edge_se = None
    if rollout is not None and rollout.best is not None and rollout.best.row in by_row:
        a = by_row[rollout.best.row]
        b = by_row.get(rollout.candidates[1].row) if len(rollout.candidates) > 1 else None
        edge = rollout.per_week(rollout.edge) if rollout.edge is not None else None
        edge_se = rollout.per_week(rollout.edge_se) if rollout.edge_se is not None else None
        confidence = rollout.confidence
    else:
        rollout = None
        a = rec.best
        if a is None:
            return {}
        b = rec.candidates[1] if len(rec.candidates) > 1 else None
        edge = rec.per_week(a.score - b.score) if b else 0.0
    ra = board_players.row(a.row, named=True)
    rb = board_players.row(b.row, named=True) if b else None
    sim_surv = rollout.survival if rollout is not None else {}
    a_surv = sim_surv.get(a.sleeper_id, a.survival)
    b_surv = sim_surv.get(b.sleeper_id, b.survival) if b else None
    why, risk = [], []
    if b and rec.next_pick:
        if b_surv - a_surv > 0.25:
            why.append(f"{b.name} is likely to be there at {rec.next_pick} ({b_surv:.0%}); {a.name} is not ({a_surv:.0%}).")
        elif a.delta > b.delta:
            why.append(f"{a.name} adds more to your lineup now (+{rec.per_week(a.delta - b.delta):.1f} FP/week).")
    elif b and a.delta >= b.delta:
        why.append(f"{a.name} adds the most to your lineup.")
    sa = stocks_share(ra, settings)
    if sa >= 0.2:
        why.append(f"{sa:.0%} of {a.name}'s value comes from steals and blocks, which this scoring doubles.")
    if ra["lockin_premium_3"] > (rb["lockin_premium_3"] if rb else 0) + 0.5:
        why.append(f"Spikier games: +{ra['lockin_premium_3']:.1f} lockable points in a 3-game week.")
    if rb:
        if ra["season_sd"] > 1.15 * rb["season_sd"]:
            risk.append(f"wider season range than {b.name}")
        if ra["share_hat"] + 0.03 < rb["share_hat"]:
            risk.append(f"availability {ra['share_hat']:.0%} of games ({b.name}: {rb['share_hat']:.0%})")
        if ra.get("rookie"):
            risk.append("rookie: draft-slot baseline, widest intervals")
        if ra.get("injury_status"):
            risk.append(f"injury status: {ra['injury_status']}")
    from draftai.models.explain import top_drivers

    drivers = [(g, v) for g, v in top_drivers(ra, "fp") if abs(v) >= 0.3]
    return {
        "pick": rec.pick,
        "reach": a.reach,
        "drivers": drivers,
        "player": a.name,
        "runner_up": b.name if b else None,
        "edge_per_week": edge,
        "fp_pg": ra["fp_pg"],
        "fp_p10": ra["fp_p10"],
        "fp_p90": ra["fp_p90"],
        "lockin_3": ra["lockin_3"],
        "lockin_premium_3": ra["lockin_premium_3"],
        "vor_per_week": ra["vor_per_week"],
        "survival": a_surv,
        "runner_up_survival": b_surv,
        "next_pick": rec.next_pick,
        "why": why,
        "risk": risk,
        "confidence": confidence,
        "edge_se": edge_se,
        "rollout_worlds": rollout.worlds if rollout is not None else None,
    }


def render(c: dict) -> str:
    if not c:
        return "No recommendation."
    if c.get("confidence") is not None and c["runner_up"]:
        head = f"Pick {c['pick']}: draft {c['player']}        confidence {c['confidence']:.0%} vs {c['runner_up']}"
    else:
        head = f"Pick {c['pick']}: draft {c['player']}" + (
            f"        edge vs {c['runner_up']}: {c['edge_per_week']:+.1f} FP/week"
            if c["runner_up"] and c["edge_per_week"] is not None else "")
    lines = [head]
    if c.get("reach", 1.0) < 0.999:
        lines.append(f"(planning ahead: {c['reach']:.0%} chance {c['player']} is still there at pick {c['pick']})")
    lines += [
        "",
        f"Projected            {c['fp_pg']:.1f} FP/G (season P10-P90: {c['fp_p10']:.0f} to {c['fp_p90']:.0f})",
        f"Lock-In value        {c['lockin_3']:.1f} per slot-week with 3 games (+{c['lockin_premium_3']:.1f} from game-to-game spread)",
        f"Value over repl.     {c['vor_per_week']:+.1f} per week",
    ]
    if c["next_pick"]:
        rs = f"   ({c['runner_up']}: {c['runner_up_survival']:.0%})" if c["runner_up"] else ""
        lines.append(f"Survival to pick {c['next_pick']:<4} {c['survival']:.0%}{rs}")
    if c.get("confidence") is not None and c["runner_up"]:
        se = f" (+/- {c['edge_se']:.1f})" if c.get("edge_se") is not None else ""
        lines.append(f"Edge over {c['runner_up']:<9} {c['edge_per_week']:+.1f} FP per week{se}, "
                     f"{c['rollout_worlds']} simulated drafts and seasons")
    if c.get("drivers"):
        lines.append("Projection drivers   " + ", ".join(f"{g} {v:+.1f}" for g, v in c["drivers"]))
    if c["why"]:
        lines += ["", "Why:  " + "\n      ".join(c["why"])]
    if c["risk"]:
        lines.append("Risk: " + "; ".join(c["risk"]) + ".")
    return "\n".join(lines)
