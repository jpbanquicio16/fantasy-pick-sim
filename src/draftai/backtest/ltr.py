"""The learning-to-rank experiment (design section 17, V5).

The question: can a ranker trained on realised outcomes correct the simulator's systematic
biases? If the backtest says no, report that and keep the simulator.

1. **Collect.** In simulated leagues of each backtest season (your seat rotating through the
   slots, opponents from the backtest's room, on random numbers of their own), every pick
   of yours is a query and its documents are the greedy recommender's top candidates. Your
   seat follows the simulator's choice, with some exploration so that the states are not
   all the simulator's own.
2. **Label** every document twice:

   * simulator label: the rollout's expected final-roster value if you take him (every
     document a rollout candidate, no racing);
   * realised label: take him, let the base policy (greedy VOR, the rollouts' own) finish
     your draft against the same opponents, and score that roster with the real season.
     One realisation per document, so it is noisy, and within a season every label shares
     the same season.

3. **Train**, leaving one season out: LambdaRank on the simulator's labels (``ltr_sim``, a
   distilled, fast surrogate of the simulator), then more boosting rounds on the realised
   labels (``ltr``, the fine-tuned ranker); ``ltr_real_only`` (realised labels from scratch)
   is the ablation that says whether the distillation helps.
4. **Judge** it on the held-out season: offline (NDCG@5 and regret against realised labels,
   next to the simulator's own ranking), and, as the design asks, in the paired draft
   backtest against the simulator itself (strategies ``ltr_sim`` and ``ltr``).
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from draftai.backtest.history import LeagueRules
from draftai.config import BacktestConfig, LTRConfig, Paths, SimConfig, TrainConfig
from draftai.draft.ranker import FEATURES, Ranker, fit, grades, ndcg_at, query

STAGES = ("sim", "real", "real_only")
METHODS = ("simulator", "ltr", "ltr_sim", "ltr_real_only", "greedy_survival", "vor", "adp")


def query_id(season: int, league: int, pick: int) -> int:
    return int(season * 10_000_000 + league * 1_000 + pick)


def ltr_dir(paths: Paths) -> Path:
    return paths.artifacts / "ltr"


def ranker_path(paths: Paths, season: int, stage: str) -> Path:
    return ltr_dir(paths) / "rankers" / f"season_{season}" / f"{stage}.txt"


def _cfg_hash(cfg: LTRConfig) -> str:
    keep = {k: v for k, v in cfg.model_dump().items() if k in ("leagues", "docs", "worlds", "epsilon", "seed")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:8]


# --- 1-2. collect and label ----------------------------------------------------------------------
def realised_branch(room, state, first_row: int, uniforms: np.ndarray, base) -> float:
    """Take ``first_row`` now, finish your draft with the base policy against the same
    opponents, and score your roster with the real season (playoff weeks weighted)."""
    from draftai.backtest.league import team_weeks
    from draftai.backtest.run import draft, rosters_of

    st = state.copy()
    last = st.my_upcoming()[-1]
    done = {"first": False}

    def chooser(s, avail, counts):
        if not done["first"]:
            done["first"] = True
            return first_row
        return base.choose(s, avail, counts)

    st = draft(room, chooser, st, uniforms, stop_after=last)
    rows = rosters_of(room, st)[st.my_slot - 1]
    return float(team_weeks(rows, room.sd.realised, room.elig) @ room.sd.realised.weights)


def collect_league(room, engine, base, sim: SimConfig, cfg: LTRConfig, season: int, league: int,
                   rng: np.random.Generator) -> list[dict]:
    """Queries and both labels for every pick of yours in one simulated league."""
    from draftai.backtest.run import draft

    sd = room.sd
    names, ids = sd.board.players["name"].to_list(), room.ids
    W = float(sd.board.weights.sum())
    teams, rounds = room.rules.teams, room.rules.rounds
    slot = league % teams + 1
    uniforms = np.random.default_rng([cfg.seed, season, league, 1]).random(teams * rounds + 2)
    out: list[dict] = []

    def chooser(state, avail, counts):
        rec, cands, X = query(sd.board, state, sim, room.evaluator, room.needs, counts, cfg.docs)
        ro = engine.recommend(state, rec)
        simv = {c.row: c.mean for c in ro.candidates}
        sim_label = np.array([simv.get(c.row, np.nan) for c in cands])
        real = np.array([realised_branch(room, state, c.row, uniforms, base) for c in cands])
        order = np.argsort(-np.nan_to_num(sim_label, nan=-np.inf), kind="stable")
        pick = int(order[0]) if rng.random() >= cfg.epsilon else int(rng.choice(order[: min(5, order.size)]))
        q = query_id(season, league, state.current_pick)
        for j, c in enumerate(cands):
            out.append({"season": season, "league": league, "pick": state.current_pick, "query": q, "doc": j,
                        "row": c.row, "sleeper_id": ids[c.row], "name": names[c.row],
                        **{f: float(X[j, k]) for k, f in enumerate(FEATURES)},
                        "sim_label": float(sim_label[j]) / W, "real_label": float(real[j]) / W,
                        "chosen": j == pick})
        return cands[pick].row

    draft(room, chooser, room.new_state(slot), uniforms)
    return out


def collect_season(season: int, rules: LeagueRules, sim: SimConfig, train_cfg: TrainConfig, bt: BacktestConfig,
                   paths: Paths) -> pl.DataFrame:
    from draftai.backtest.history import season_data
    from draftai.backtest.run import Room, opponent_model
    from draftai.backtest.strategies import KeyGreedy
    from draftai.draft.rollout import RolloutConfig, RolloutEngine

    cfg = bt.ltr
    sd = season_data(season, rules, sim, train_cfg, paths, bt.game_samples, log=lambda *_: None)
    model = opponent_model(sd.board, rules, sim, paths, bt.room)
    room = Room.build(sd, rules, model)
    rcfg = RolloutConfig(worlds=cfg.worlds, shortlist=cfg.docs, race_after=cfg.worlds, waiver_slots=sim.waiver_slots,
                         seed=cfg.seed + season)
    engine = RolloutEngine(sd.board, model, rcfg, sim, evaluator=room.evaluator)
    base = KeyGreedy("vor", "vor")
    base.prepare(sd, room.needs, room.evaluator, model, sim, rcfg)
    rng = np.random.default_rng([cfg.seed, season])
    rows = []
    for league in range(cfg.leagues):
        rows += collect_league(room, engine, base, sim, cfg, season, league, rng)
    return pl.DataFrame(rows)


def collect(rules: LeagueRules, sim: SimConfig, train_cfg: TrainConfig, bt: BacktestConfig, paths: Paths,
            log=print) -> pl.DataFrame:
    """Queries for every backtest season, cached per season board and collection settings."""
    from draftai.backtest.history import season_data, season_key

    cache = ltr_dir(paths) / "queries"
    cache.mkdir(parents=True, exist_ok=True)
    files = {s: cache / f"{season_key(s, rules, sim, bt.game_samples, train_cfg)}_{_cfg_hash(bt.ltr)}.parquet"
             for s in bt.seasons}
    todo = [s for s, f in files.items() if not f.exists()]
    for s in todo:  # build (and cache) seasons serially so workers only load them
        season_data(s, rules, sim, train_cfg, paths, bt.game_samples, log=log)
    if todo:
        t0 = time.time()
        jobs = max(1, min(bt.jobs, len(todo)))
        log(f"collecting LTR queries for {todo} ({bt.ltr.leagues} leagues each, {jobs} processes)...")
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {s: ex.submit(collect_season, s, rules, sim, train_cfg, bt, paths) for s in todo}
            for s, f in futs.items():
                f.result().write_parquet(files[s])
                log(f"season {s}: queries collected ({time.time() - t0:.0f} s)")
    return pl.concat([pl.read_parquet(f) for f in files.values()], how="vertical_relaxed")


# --- 3. train ---------------------------------------------------------------------------------------
def _arrays(q: pl.DataFrame, label: str, levels: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    q = q.sort("query", "doc")
    X = q.select(FEATURES).to_numpy().astype(np.float32)
    g = q["query"].to_numpy()
    y = grades(q[label].fill_nan(None).fill_null(q[label].min()).to_numpy().astype(float), g, levels)
    sizes = q.group_by("query", maintain_order=True).len()["len"].to_list()
    return X, y, sizes


def train(queries: pl.DataFrame, cfg: LTRConfig, paths: Paths, season_keys: dict[int, str], log=print) -> None:
    """Leave one season out: for each season, rankers fitted on the other seasons only."""
    seasons = sorted(queries["season"].unique().to_list())
    for s in seasons:
        tr = queries.filter(pl.col("season") != s)
        X, y_sim, sizes = _arrays(tr, "sim_label", cfg.levels)
        _, y_real, _ = _arrays(tr, "real_label", cfg.levels)
        meta = {"season": s, "trained_on": [x for x in seasons if x != s], "queries": len(sizes),
                "season_key": season_keys.get(s), "features": FEATURES}
        b_sim = fit(X, y_sim, sizes, cfg.sim_rounds, cfg.levels, seed=cfg.seed)
        p_sim = ranker_path(paths, s, "sim")
        Ranker(b_sim, {**meta, "stage": "sim"}).save(p_sim)
        # Fine-tune: continue boosting from the distilled ranker on the realised labels.
        b_real = fit(X, y_real, sizes, cfg.real_rounds, cfg.levels, init_model=str(p_sim),
                     learning_rate=cfg.real_learning_rate, seed=cfg.seed)
        Ranker(b_real, {**meta, "stage": "real"}).save(ranker_path(paths, s, "real"))
        b_only = fit(X, y_real, sizes, cfg.sim_rounds, cfg.levels, seed=cfg.seed)
        Ranker(b_only, {**meta, "stage": "real_only"}).save(ranker_path(paths, s, "real_only"))
        log(f"rankers for {s}: trained on {meta['trained_on']} ({len(sizes)} queries)")


# --- 4. offline evaluation ----------------------------------------------------------------------------
def offline(queries: pl.DataFrame, paths: Paths, levels: int = 10, k: int = 5) -> pl.DataFrame:
    """Per held-out query and method: NDCG@5 against realised and simulator grades, regret
    (FP per week lost against the best document in hindsight) and agreement with the
    simulator's top choice."""
    rows = []
    for s in sorted(queries["season"].unique().to_list()):
        q = queries.filter(pl.col("season") == s).sort("query", "doc")
        X = q.select(FEATURES).to_numpy().astype(np.float32)
        model_scores = {f"ltr{'' if st == 'real' else '_' + st}": Ranker.load(ranker_path(paths, s, st)).scores(X)
                        for st in STAGES}
        scores = {
            "simulator": q["sim_label"].fill_nan(-1e9).to_numpy(),
            "greedy_survival": -q["greedy_rank"].to_numpy(),
            "vor": q["vor_pw"].to_numpy(),
            "adp": -q["adp"].to_numpy(),
            **model_scores,
        }
        g = q["query"].to_numpy()
        real, simv = q["real_label"].to_numpy(), q["sim_label"].fill_nan(None).fill_null(0.0).to_numpy()
        g_real, g_sim = grades(real, g, levels), grades(simv, g, levels)
        picks = q["pick"].to_numpy()
        bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
        for a, b in pairwise(bounds):
            sim_top = a + int(np.argmax(scores["simulator"][a:b]))
            for m in METHODS:
                sc = scores[m][a:b]
                top = a + int(np.argmax(sc))
                rows.append({"season": s, "query": int(g[a]), "pick": int(picks[a]), "method": m,
                             "ndcg5_real": ndcg_at(sc, g_real[a:b].astype(float), k),
                             "ndcg5_sim": ndcg_at(sc, g_sim[a:b].astype(float), k),
                             "regret_real": float(real[a:b].max() - real[top]),
                             "regret_sim": float(simv[a:b].max() - simv[top]),
                             "top1_is_simulators": bool(top == sim_top)})
    return pl.DataFrame(rows)


def label_agreement(queries: pl.DataFrame) -> pl.DataFrame:
    """How well the simulator's labels predict the realised ones, within queries."""
    from scipy.stats import spearmanr

    out = []
    for (s,), d in queries.group_by(["season"], maintain_order=True):
        rhos = []
        for (_,), qd in d.group_by(["query"]):
            a, b = qd["sim_label"].to_numpy(), qd["real_label"].to_numpy()
            if np.std(a) > 0 and np.std(b) > 0:
                rhos.append(spearmanr(a, b).statistic)
        out.append({"season": s, "queries": d["query"].n_unique(), "docs": d.height,
                    "spearman_sim_vs_real": float(np.mean(rhos)) if rhos else None})
    return pl.DataFrame(out).sort("season")


def importances(paths: Paths, seasons: list[int]) -> pl.DataFrame:
    rows = []
    for st in STAGES:
        imp = np.zeros(len(FEATURES))
        for s in seasons:
            r = Ranker.load(ranker_path(paths, s, st))
            imp += r.booster.feature_importance("gain")
        imp = imp / max(imp.sum(), 1e-9)
        rows += [{"stage": st, "feature": f, "gain_share": float(v)} for f, v in zip(FEATURES, imp)]
    return pl.DataFrame(rows).pivot("stage", index="feature", values="gain_share").sort("real", descending=True)


def paired_query_diff(off: pl.DataFrame, a: str, b: str, metric: str, reps: int = 2000, seed: int = 0) -> dict:
    """Mean paired difference a - b over held-out queries, with a bootstrap interval that
    resamples queries within seasons. Queries of one season share its realisation, so the
    interval understates the uncertainty."""
    x = off.filter(pl.col("method") == a).select("season", "query", pl.col(metric).alias("a"))
    y = off.filter(pl.col("method") == b).select("season", "query", pl.col(metric).alias("b"))
    j = x.join(y, on=["season", "query"])
    d = (j["a"] - j["b"]).to_numpy()
    groups = [np.flatnonzero((j["season"] == s).to_numpy()) for s in j["season"].unique().to_list()]
    rng = np.random.default_rng(seed)
    boots = [d[np.concatenate([gi[rng.integers(0, gi.size, gi.size)] for gi in groups])].mean() for _ in range(reps)]
    return {"a": a, "b": b, "metric": metric, "diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "queries": int(d.size)}


def write_report(queries: pl.DataFrame, off: pl.DataFrame, paths: Paths, backtest_rows: pl.DataFrame | None,
                 cfg: LTRConfig) -> Path:
    from draftai.backtest.run import paired

    def md(df: pl.DataFrame) -> str:
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        for row in df.iter_rows():
            lines.append("| " + " | ".join("" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
                                           for v in row) + " |")
        return "\n".join(lines)

    seasons = sorted(queries["season"].unique().to_list())
    pooled = off.group_by("method").agg(
        pl.col("ndcg5_real").mean(), pl.col("ndcg5_sim").mean(), pl.col("regret_real").mean(),
        pl.col("regret_sim").mean(), pl.col("top1_is_simulators").mean()).sort("regret_real")
    per = off.group_by("season", "method").agg(pl.col("regret_real").mean()).pivot(
        "season", index="method", values="regret_real", sort_columns=True).sort("method")
    diffs = pl.DataFrame([paired_query_diff(off, a, "simulator", m) for a in ("ltr", "ltr_sim", "ltr_real_only")
                          for m in ("regret_real", "ndcg5_real")])
    text = [
        "# Learning-to-rank (design section 17)",
        "",
        "Can a ranker trained on realised outcomes correct the simulator's systematic biases? Queries are your",
        f"picks in {cfg.leagues} simulated leagues per season (seat rotated through every slot); documents are the greedy",
        f"recommender's top {cfg.docs} candidates. Simulator label: the rollout's expected final-roster value",
        f"({cfg.worlds} worlds, every document a candidate). Realised label: take him, let greedy VOR finish your",
        "draft against the same opponents, and score the roster with the real season (one realisation). Rankers",
        "are LambdaRank (NDCG@5) trained leaving one season out: `ltr_sim` on simulator labels (the distilled",
        "surrogate), `ltr` continues from it on realised labels (the design's plan), `ltr_real_only` uses realised",
        "labels alone. Values are FP per week. Your seat followed the simulator's choice, with a random top-5",
        f"candidate {cfg.epsilon:.0%} of the time.",
        "",
        "## The labels",
        "Within-query rank correlation between the simulator's labels and the realised ones.",
        md(label_agreement(queries)),
        "",
        "## Offline, on the held-out season",
        "`regret_real`: realised FP/week lost against the best document in hindsight (lower is better; the best",
        "document is chosen with hindsight, so even a perfect forecaster has regret). `top1_is_simulators`: share of",
        "queries where the method's first choice is the simulator's.",
        md(pooled),
        "",
        "### Realised regret by held-out season",
        md(per),
        "",
        "### Paired against the simulator (95% interval, queries resampled within seasons)",
        "Negative regret differences and positive NDCG differences favour the ranker.",
        md(diffs.with_columns(pl.col("diff", "lo", "hi").round(3))),
        "",
        "## What the rankers use (share of split gain, mean over folds)",
        md(importances(paths, seasons).head(15)),
    ]
    if backtest_rows is not None and backtest_rows.height:
        from draftai.backtest.run import per_season, summary

        strategies = backtest_rows["strategy"].unique().to_list()
        pairs = []
        for a in ("ltr", "ltr_sim"):
            for b in ("rollout", "greedy_survival", "vor", "adp", "ltr_sim"):
                if a == b or a not in strategies or b not in strategies:
                    continue
                for metric in ("p_playoffs", "realised_per_week", "p_title"):
                    p = paired(backtest_rows, a, b, metric)
                    if p:
                        pairs.append(p)
        text += [
            "",
            "## The draft backtest (the design's judge)",
            "Same leagues, seats, opponents and schedules as `draftai backtest`; each season's rankers never saw it.",
            md(summary(backtest_rows)),
            "",
            "### P(top 6) by season",
            md(per_season(backtest_rows)),
            "",
            "### Paired differences (95% bootstrap interval over leagues)",
            md(pl.DataFrame(pairs).with_columns(pl.col("diff", "lo", "hi").round(3))) if pairs else "(none)",
        ]
    path = paths.artifacts / "reports" / "ltr_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text))
    off.write_parquet(paths.artifacts / "reports" / "ltr_offline.parquet")
    return path
