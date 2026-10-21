"""Learning-to-rank over draft candidates (design section 17; a V5 experiment).

A query is a draft state at your pick; its documents are the greedy recommender's top
candidates. Each (state, player) pair is described by his value (Lock-In season value, VOR,
marginal roster value and the survival lookahead), the market (ADP against the pick), his
spread, roster need and the round. Values are per week and also relative to the best
candidate in the query, so one model serves every board and every round. LightGBM's
LambdaRank orders the documents, optimising NDCG@5.

The labels, the training and the evaluation live in ``backtest/ltr.py``; this module is
the pure part a strategy needs at a pick: build the query, score it, choose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from draftai.config import SimConfig
from draftai.draft.needs import NeedModel
from draftai.draft.recommender import Candidate, Recommendation, RosterEvaluator, recommend
from draftai.draft.state import DraftState
from draftai.value.board import Board

N_DOCS = 24
FEATURES = [
    "value_pw", "lockin_3", "vor_pw", "fp_pg", "share_hat", "spread", "lockin_premium_3", "games_proj",
    "adp", "adp_minus_pick", "adp_rank", "delta_pw", "delta_rel", "delta_gap_pw", "survival", "e_next_pw",
    "score_gap_pw", "greedy_rank", "vor_rank", "round", "picks_to_next", "fills_need", "open_slots",
    "roster_size", "is_g", "is_f", "is_c", "rookie", "age",
]


def _rank(v: np.ndarray, descending: bool = True) -> np.ndarray:
    order = np.argsort(-v if descending else v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    return r


def features(board: Board, state: DraftState, rec: Recommendation, cands: list[Candidate], needs: NeedModel,
             counts: np.ndarray) -> np.ndarray:
    """(documents, FEATURES) for one query."""
    p = board.players
    rows = np.array([c.row for c in cands], dtype=int)
    W = max(float(board.weights.sum()), 1e-9)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in p.columns:
            return np.full(rows.size, default)
        v = p[name].to_numpy()[rows].astype(float)
        return np.where(np.isfinite(v), v, default)

    value, vor = col("season_value"), col("vor")
    delta = np.array([c.delta for c in cands])
    score = np.array([c.score for c in cands])
    adp = np.array([c.adp for c in cands])
    fills = needs.fills(counts)[needs.type_of[rows]].astype(float)
    open_slots = float(len(needs.slots) - needs.matched(counts))
    t = state.current_pick
    t_next = rec.next_pick
    pos = [set(x) for x in p["positions"].to_list()]
    rookie = p["rookie"].to_numpy()[rows].astype(float) if "rookie" in p.columns else np.zeros(rows.size)
    best_delta = max(float(delta.max()), 1e-9)
    X = np.column_stack([
        value / W, col("lockin_3"), vor / W, col("fp_pg"), col("share_hat"),
        col("season_sd") / np.maximum(value, 1.0), col("lockin_premium_3"), col("games_proj"),
        adp, adp - t, _rank(adp, descending=False), delta / W, delta / best_delta, (delta.max() - delta) / W,
        np.array([c.survival for c in cands]), np.array([c.e_next for c in cands]) / W, (score.max() - score) / W,
        _rank(score), _rank(vor), np.full(rows.size, (t - 1) // state.teams + 1),
        np.full(rows.size, (t_next - t) if t_next else 0), fills, np.full(rows.size, open_slots),
        np.full(rows.size, len(state.roster())),
        np.array([float(bool(pos[r] & {"PG", "SG"})) for r in rows]),
        np.array([float(bool(pos[r] & {"SF", "PF"})) for r in rows]),
        np.array([float("C" in pos[r]) for r in rows]),
        rookie, col("age", 26.0),
    ])
    return X.astype(np.float32)


def query(board: Board, state: DraftState, sim: SimConfig, evaluator: RosterEvaluator, needs: NeedModel,
          counts: np.ndarray, n_docs: int = N_DOCS) -> tuple[Recommendation, list[Candidate], np.ndarray]:
    """The query at your pick: the greedy recommendation, its top ``n_docs`` candidates and
    their features."""
    rec = recommend(board, state, sim, evaluator)
    cands = rec.candidates[:n_docs]
    return rec, cands, features(board, state, rec, cands, needs, counts) if cands else np.zeros((0, len(FEATURES)))


def grades(values: np.ndarray, groups: np.ndarray, levels: int = 10) -> np.ndarray:
    """Relevance grades 0..levels within each query: the value's position between the
    query's worst and best document (ties and a flat query get the same grade)."""
    out = np.zeros(values.size, dtype=np.int32)
    for g in np.unique(groups):
        m = groups == g
        v = values[m]
        lo, hi = float(v.min()), float(v.max())
        out[m] = 0 if hi - lo < 1e-9 else np.rint(levels * (v - lo) / (hi - lo)).astype(np.int32)
    return out


def ndcg_at(scores: np.ndarray, gains: np.ndarray, k: int = 5) -> float:
    """NDCG@k of one query, with linear gains."""
    order = np.argsort(-scores, kind="stable")[:k]
    ideal = np.sort(gains)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains[order] * disc[: order.size]).sum())
    idcg = float((ideal * disc[: ideal.size]).sum())
    return dcg / idcg if idcg > 0 else 1.0


DEFAULT_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [5], "learning_rate": 0.05, "num_leaves": 15,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "deterministic": True, "force_row_wise": True,
}


def fit(X: np.ndarray, labels: np.ndarray, group_sizes: list[int], rounds: int, levels: int = 10,
        init_model: lgb.Booster | str | None = None, learning_rate: float | None = None, seed: int = 0) -> lgb.Booster:
    """LambdaRank on graded labels; ``init_model`` (a booster or its saved file) is continued
    with more rounds: fine-tuning."""
    params = {**DEFAULT_PARAMS, "label_gain": list(range(levels + 1)), "seed": seed}
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    data = lgb.Dataset(X, label=labels, group=group_sizes, feature_name=FEATURES, free_raw_data=False)
    return lgb.train(params, data, num_boost_round=rounds, init_model=init_model, keep_training_booster=True)


@dataclass
class Ranker:
    booster: lgb.Booster
    meta: dict

    def scores(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X) if X.shape[0] else np.zeros(0)

    def choose(self, cands: list[Candidate], X: np.ndarray) -> int:
        """Board row of the highest-scoring candidate."""
        return cands[int(np.argmax(self.scores(X)))].row

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> Ranker:
        meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        return cls(lgb.Booster(model_file=str(path)), meta)
