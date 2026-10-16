"""The paired draft backtest (V4 gate).

For each past season: freeze information at the draft date (``history.py``), draft many
simulated leagues, give one seat to the strategy under test, and play the season with
the real box scores under Lock-In rules (``league.py``). Strategies are compared in a
paired design: the same leagues, the same seat, the same opponent random numbers and the
same head-to-head schedules; only the strategy changes. The seat rotates through every
slot so no strategy benefits from a lucky one.

Also measured here:

* confidence calibration: at each rollout decision, the A-branch and the B-branch are
  both continued with the base policy against the same opponents, and the A-roster's
  realised season is compared with the B-roster's. Among decisions shown at about 70%,
  A should win about 70% of the time;
* the market tests: does the projection rank realised value better than the market,
  and does it carry information the market does not (regression on both)?
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from draftai.backtest.history import LeagueRules, SeasonData, season_data
from draftai.backtest.league import play, team_weeks
from draftai.backtest.strategies import KeyGreedy, Rollout, Strategy, build
from draftai.config import BacktestConfig, Paths, SimConfig, TrainConfig
from draftai.draft.needs import NeedModel
from draftai.draft.opponents import Market, OpponentModel, calibrate, choose
from draftai.draft.order import pick_owners
from draftai.draft.recommender import RosterEvaluator
from draftai.draft.rollout import RolloutConfig
from draftai.draft.state import DraftState
from draftai.value.lineup import eligibility_matrix

METRICS = ["p_playoffs", "p_title", "realised_per_week", "win_rate", "proj_per_week"]


@dataclass
class Room:
    """Everything a season's drafts share."""

    sd: SeasonData
    rules: LeagueRules
    needs: NeedModel
    market: Market
    model: OpponentModel
    evaluator: RosterEvaluator
    elig: np.ndarray
    ids: list[str]

    @classmethod
    def build(cls, sd: SeasonData, rules: LeagueRules, model: OpponentModel) -> "Room":
        b = sd.board
        needs = NeedModel(b.positions(), b.slots)
        return cls(sd, rules, needs, Market(b.players["adp"].to_numpy(), needs), model, RosterEvaluator(b),
                   eligibility_matrix(b.positions(), b.slots), b.players["sleeper_id"].to_list())

    def new_state(self, slot: int) -> DraftState:
        r = self.rules
        return DraftState.new(r.teams, r.rounds, slot, r.draft_type, r.reversal_round)


def opponent_model(room_board, rules: LeagueRules, sim: SimConfig, paths: Paths, room: str = "calibrated") -> OpponentModel:
    """The simulated room: calibrated to the configured ADP spread on this season's market,
    or (``room="fitted"``) the model fitted on your real drafts by `draftai opponents`."""
    path = paths.artifacts / "opponents" / "history.json"
    if room == "fitted" and path.exists():
        return OpponentModel.load(path)
    needs = NeedModel(room_board.positions(), room_board.slots)
    owners = pick_owners(rules.teams, rules.rounds, rules.draft_type, rules.reversal_round)
    return calibrate(Market(room_board.players["adp"].to_numpy(), needs), owners, sim.adp_sigma_a, sim.adp_sigma_b,
                     window=sim.opponent_window)


def _arrays(room: Room, state: DraftState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_of = {s: i for i, s in enumerate(room.ids)}
    avail = np.ones(len(room.ids), dtype=bool)
    counts = np.zeros((state.teams + 1, room.needs.n_types), dtype=np.int64)
    for sid, slot in zip(state.picks, state.pick_slots):
        r = row_of.get(sid)
        if r is not None:
            avail[r] = False
            counts[slot, room.needs.type_of[r]] += 1
    cols = np.ones(room.market.size, dtype=bool)
    cols[room.market.col_of[~avail]] = False
    return avail, cols, counts


def draft(room: Room, chooser, state: DraftState, uniforms: np.ndarray, stop_after: int | None = None) -> DraftState:
    """Run a draft from ``state``: your seat via ``chooser(state, avail, counts)``, every
    other seat via the opponent model with this league's random numbers."""
    avail, cols, counts = _arrays(room, state)
    m = room.market
    while not state.complete and (stop_after is None or state.current_pick <= stop_after):
        t, owner = state.current_pick, state.on_the_clock
        if owner == state.my_slot:
            row = chooser(state, avail, counts[owner])
        else:
            col = choose(room.model, m, cols, counts[owner], t, float(uniforms[t]))
            row = int(m.order[col])
        state.add_pick(room.ids[row])
        avail[row] = False
        cols[m.col_of[row]] = False
        counts[owner, room.needs.type_of[row]] += 1
    return state


def rosters_of(room: Room, state: DraftState) -> list[list[int]]:
    row_of = {s: i for i, s in enumerate(room.ids)}
    return [[row_of[s] for s in state.roster(slot) if s in row_of] for slot in range(1, state.teams + 1)]


def market_tests(sd: SeasonData, top: int = 200, reps: int = 1000, seed: int = 0) -> dict:
    """Design section 10, tests 2 and 3, against this season's market."""
    p = sd.board.players
    real, adp, proj = sd.realised.season_value, p["adp"].to_numpy(), p["season_value"].to_numpy()
    m = np.argsort(adp)[:top]
    y, a, x = real[m], -adp[m], proj[m]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    X = np.column_stack([np.ones(len(m)), z(a), z(x)])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        i = rng.integers(0, len(m), len(m))
        boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][2])
    return {
        "spearman_projection": float(spearmanr(x, y).statistic),
        "spearman_market": float(spearmanr(a, y).statistic),
        "coef_projection": float(coef[2]),
        "coef_projection_ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "coef_market": float(coef[1]),
        "players": int(len(m)),
    }


def calibration_rows(room: Room, decisions, uniforms: np.ndarray, base: KeyGreedy) -> list[dict]:
    """Realised outcome of each rollout decision's A-branch vs B-branch (base policy after)."""
    out = []
    w = room.sd.realised.weights
    for d in decisions:
        if d.b is None or d.confidence is None:
            continue
        val = {}
        for label, first in (("a", d.a), ("b", d.b)):
            st = d.state.copy()
            last = st.my_upcoming()[-1]
            done = {"first": False}

            def chooser(state, avail, counts, first=first, done=done):
                if not done["first"]:
                    done["first"] = True
                    return first
                return base.choose(state, avail, counts)

            st = draft(room, chooser, st, uniforms, stop_after=last)
            rows = rosters_of(room, st)[st.my_slot - 1]
            val[label] = float(team_weeks(rows, room.sd.realised, room.elig) @ w)
        out.append({"pick": d.pick, "confidence": d.confidence, "edge": d.edge, "a_won": val["a"] > val["b"],
                    "realised_diff": val["a"] - val["b"]})
    return out


def run_season(season: int, rules: LeagueRules, sim: SimConfig, train_cfg: TrainConfig, bt: BacktestConfig,
               paths: Paths) -> dict:
    t0 = time.time()
    sd = season_data(season, rules, sim, train_cfg, paths, bt.game_samples, log=lambda *_: None)
    model = opponent_model(sd.board, rules, sim, paths, bt.room)
    room = Room.build(sd, rules, model)
    rcfg = RolloutConfig(worlds=bt.rollout_worlds, shortlist=bt.rollout_shortlist, race_after=bt.rollout_race_after,
                         waiver_slots=sim.waiver_slots, seed=bt.seed + season)
    strategies: list[Strategy] = build(bt.strategies)
    for s in strategies:
        s.prepare(sd, room.needs, room.evaluator, model, sim, rcfg)
    base = KeyGreedy("vor", "vor")
    base.prepare(sd, room.needs, room.evaluator, model, sim, rcfg)
    weeks_weight = float(sd.board.weights.sum())
    rows, calib = [], []
    for league in range(bt.leagues):
        slot = league % rules.teams + 1
        uniforms = np.random.default_rng([bt.seed, season, league]).random(rules.teams * rules.rounds + 2)
        for s in strategies:
            if s.expensive and league >= bt.expensive_leagues:
                continue
            s.reset()
            state = draft(room, s.choose, room.new_state(slot), uniforms)
            rosters = rosters_of(room, state)
            res = play(rosters, sd.realised, room.elig, rules, np.random.default_rng([bt.seed, season, league, 1]),
                       bt.schedules, bt.waiver_claims)
            i = slot - 1
            rows.append({
                "season": season, "league": league, "slot": slot, "strategy": s.name,
                "proj_per_week": room.evaluator.value(rosters[i]) / weeks_weight,
                "realised_per_week": float(res.regular_per_week(sd.realised.playoff)[i]),
                "win_rate": float(res.outcome.win_rate[i]), "p_playoffs": float(res.outcome.p_playoffs[i]),
                "p_title": float(res.outcome.p_title[i]), "mean_seed": float(res.outcome.mean_seed[i]),
                "first_picks": ", ".join(sd.board.players["name"][r] for r in rosters[i][:3]),
            })
            if isinstance(s, Rollout):
                for c in calibration_rows(room, s.decisions, uniforms, base):
                    calib.append({"season": season, "league": league, **c})
    meta = {"season": season, **sd.meta, "market_tests": market_tests(sd), "opponents": model.describe(),
            "seconds": round(time.time() - t0, 1)}
    return {"rows": rows, "calibration": calib, "meta": meta}


@dataclass
class BacktestResult:
    rows: pl.DataFrame
    calibration: pl.DataFrame
    meta: list[dict]
    replay: pl.DataFrame | None = None
    replay_meta: dict | None = None


def run(rules: LeagueRules, sim: SimConfig, train_cfg: TrainConfig, bt: BacktestConfig, paths: Paths,
        log=print) -> BacktestResult:
    # Build (and cache) seasons and fold models once, serially, so workers only load them.
    for season in bt.seasons:
        season_data(season, rules, sim, train_cfg, paths, bt.game_samples, log=log)
    results = []
    jobs = max(1, min(bt.jobs, len(bt.seasons)))
    if jobs == 1:
        for season in bt.seasons:
            results.append(run_season(season, rules, sim, train_cfg, bt, paths))
            log(f"season {season}: done in {results[-1]['meta']['seconds']} s")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_season, s, rules, sim, train_cfg, bt, paths): s for s in bt.seasons}
            for f, s in futs.items():
                results.append(f.result())
                log(f"season {s}: done in {results[-1]['meta']['seconds']} s")
    rows = pl.DataFrame([r for res in results for r in res["rows"]])
    calib = [c for res in results for c in res["calibration"]]
    calibration = pl.DataFrame(calib) if calib else pl.DataFrame(
        schema={"season": pl.Int64, "confidence": pl.Float64, "a_won": pl.Boolean})
    return BacktestResult(rows, calibration, [res["meta"] for res in results])


# --- summaries ------------------------------------------------------------------------------
def summary(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.group_by("strategy").agg(
        pl.len().alias("leagues"), *[pl.col(m).mean().round(3) for m in METRICS]
    ).sort("p_playoffs", descending=True)


def per_season(rows: pl.DataFrame, metric: str = "p_playoffs") -> pl.DataFrame:
    return rows.group_by("season", "strategy").agg(pl.col(metric).mean().round(3)).pivot(
        "season", index="strategy", values=metric, sort_columns=True).sort("strategy")


def paired(rows: pl.DataFrame, strategy: str, baseline: str, metric: str, reps: int = 2000, seed: int = 0,
           season: int | None = None) -> dict | None:
    """Mean paired difference over shared leagues, with a 95% bootstrap interval
    (resampling leagues within each season)."""
    r = rows if season is None else rows.filter(pl.col("season") == season)
    a = r.filter(pl.col("strategy") == strategy).select("season", "league", pl.col(metric).alias("a"))
    b = r.filter(pl.col("strategy") == baseline).select("season", "league", pl.col(metric).alias("b"))
    j = a.join(b, on=["season", "league"])
    if j.height == 0:
        return None
    d = (j["a"] - j["b"]).to_numpy()
    groups = [np.flatnonzero((j["season"] == s).to_numpy()) for s in j["season"].unique().to_list()]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        idx = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])
        boots.append(d[idx].mean())
    return {"strategy": strategy, "baseline": baseline, "metric": metric, "diff": float(d.mean()),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)), "leagues": int(d.size)}


def gate(rows: pl.DataFrame) -> dict:
    """Design section 25: the rollout's P(top 6) beats draft-by-ADP and greedy VOR in most
    test seasons, with a paired interval that excludes zero."""
    seasons = sorted(rows["season"].unique().to_list())
    out = {}
    for base in ("adp", "vor"):
        if base not in rows["strategy"].unique().to_list() or "rollout" not in rows["strategy"].unique().to_list():
            continue
        per = [paired(rows, "rollout", base, "p_playoffs", season=s) for s in seasons]
        pooled = paired(rows, "rollout", base, "p_playoffs")
        wins = sum(1 for p in per if p and p["diff"] > 0)
        out[base] = {"seasons_better": wins, "seasons": len(seasons), "pooled": pooled,
                     "passed": wins > len(seasons) / 2 and pooled is not None and pooled["lo"] > 0}
    out["passed"] = bool(out) and all(v["passed"] for k, v in out.items() if k in ("adp", "vor"))
    return out


def calibration_table(cal: pl.DataFrame) -> pl.DataFrame:
    if cal.height == 0:
        return cal
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    c = cal.with_columns(pl.max_horizontal(pl.col("confidence"), 0.5).alias("c"))
    rows = []
    for lo, hi, lab in zip(bins, bins[1:], labels):
        g = c.filter((pl.col("c") >= lo) & (pl.col("c") < hi))
        if g.height:
            rows.append({"shown": lab, "decisions": g.height, "mean_confidence": round(float(g["confidence"].mean()), 3),
                         "a_won": round(float(g["a_won"].mean()), 3)})
    return pl.DataFrame(rows)


def _md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(res: BacktestResult, rules: LeagueRules, bt: BacktestConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res.rows.write_parquet(out_dir / "backtest_rows.parquet")
    res.calibration.write_parquet(out_dir / "backtest_calibration.parquet")
    if res.replay is not None:
        res.replay.write_parquet(out_dir / "backtest_replay.parquet")
    g = gate(res.rows)
    strategies = res.rows["strategy"].unique().to_list()
    pairs = []
    for s in [x for x in ("rollout", "greedy_survival", "vor", "points", "oracle") if x in strategies]:
        for base in ("adp", "vor"):
            if base == s or base not in strategies:
                continue
            for metric in ("p_playoffs", "realised_per_week", "p_title"):
                p = paired(res.rows, s, base, metric)
                if p:
                    pairs.append(p)
    pair_df = pl.DataFrame(pairs).with_columns(pl.col("diff", "lo", "hi").round(3)) if pairs else pl.DataFrame()
    mt = pl.DataFrame([{"season": m["season"], **{k: v for k, v in m["market_tests"].items() if k != "coef_projection_ci"},
                        "coef_proj_lo": m["market_tests"]["coef_projection_ci"][0],
                        "coef_proj_hi": m["market_tests"]["coef_projection_ci"][1]} for m in res.meta]).sort("season")
    text = [
        "# Draft backtest",
        "",
        f"League: {rules.teams} teams, {rules.rounds} rounds, {rules.draft_type}"
        + (f" with reversal from round {rules.reversal_round}" if rules.reversal_round else "")
        + f"; lineup {', '.join(rules.roster_positions)}; top {rules.playoff_teams} make the playoffs.",
        f"Seasons {', '.join(str(s) for s in sorted(res.rows['season'].unique().to_list()))}; {bt.leagues} simulated leagues "
        f"per season (seat rotated through every slot); greedy_survival and rollout on the first "
        f"{bt.expensive_leagues}. Paired: same leagues, seat, opponent random numbers and schedules.",
        "",
        "Projections, board and market are frozen at each season's opening night (models trained only on",
        "earlier seasons). Opponents draft with the conditional-logit opponent model around the market"
        + (" (calibrated room: the configured ADP spread)." if bt.room == "calibrated" else
           " (fitted room: the model fitted on your real drafts)."),
        "**The market is a naive proxy** (last season's FP/G, rookies from the draft-slot projection,",
        "pre-season news from configs/backtest_news.csv half-priced) because historical ADP snapshots are",
        "not available. Real ADP is sharper, so gains against `adp` here overstate gains against a real",
        "draft room; the replay of real Sleeper drafts below is the honest comparison.",
        "",
        "`points` and `vor` draft identically in a league with UTIL slots: VOR measures every player",
        "against his cheapest eligible slot, which is always UTIL, so it is projected value minus a",
        "constant. Positional scarcity reaches the decision through marginal roster value instead.",
        "",
        "## All seasons",
        _md(summary(res.rows)),
        "",
        "`p_playoffs` and `p_title` are averaged over random head-to-head schedules; `realised_per_week` is",
        "the regular-season weekly score with real box scores; `proj_per_week` is the pre-season projection.",
        "",
        "## P(top 6) by season",
        _md(per_season(res.rows)),
        "",
        "## Paired differences (95% bootstrap interval over leagues)",
        _md(pair_df) if pairs else "(not enough strategies)",
        "",
        "## Gate (design section 25)",
        "Rollout P(top 6) above draft-by-ADP and greedy VOR in most seasons, pooled interval above zero:",
        "```json",
        json.dumps(g, indent=2),
        "```",
        "",
        "## Confidence calibration",
        "At each rollout decision both branches (recommendation A, runner-up B) were continued with the base",
        "policy against the same opponents and scored with the real season. Seasons share one realisation, so",
        "decisions within a season are correlated.",
        "",
        _md(calibration_table(res.calibration)) if res.calibration.height else "(no rollout decisions)",
        "",
        "## Market tests (design section 10, against this market)",
        _md(mt),
        "",
        "Spearman is over the market's top 200; the coefficients come from regressing realised season value",
        "on standardised market rank and projection together (bootstrap interval for the projection's).",
        "",
        "## Seasons",
        _md(pl.DataFrame([{"season": m["season"], "cutoff": m["cutoff"], "players": m["players"],
                           "sleeper_positions": round(m["drafted_with_sleeper_positions"], 3),
                           "seconds": m["seconds"]} for m in res.meta]).sort("season")),
        "",
        f"Opponents: {res.meta[0]['opponents'] if res.meta else ''}",
    ]
    if res.replay is not None and res.replay_meta:
        from draftai.backtest.replay import replay_section

        text += ["", *replay_section(res.replay, res.replay_meta)]
    path = out_dir / "backtest_report.md"
    path.write_text("\n".join(text))
    (out_dir / "backtest_summary.json").write_text(json.dumps({"gate": g, "meta": res.meta, "pairs": pairs,
                                                               "replay": res.replay_meta}, indent=2, default=str))
    return path
