"""Draft strategies compared in the backtest. Each chooses a board row for your seat.

* ``adp``: the lowest market ADP that fills an open starting slot (any, once full).
* ``points``: highest projected season value (Lock-In points), same slot rule.
* ``vor``: highest value over replacement, same slot rule. This is also the base policy
  the rollouts use for your later picks.
* ``greedy_survival``: the V3 recommender (exact marginal roster value plus the
  survival lookahead).
* ``rollout``: the V4 Monte Carlo rollout recommender.
* ``oracle``: highest realised season value, same slot rule. Hindsight; the ceiling.
* ``ltr`` / ``ltr_sim``: the learning-to-rank experiment's rankers (V5, ``backtest/ltr.py``):
  fine-tuned on realised outcomes, or distilled from the simulator only. Each season uses
  the rankers trained without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from draftai.backtest.history import SeasonData
from draftai.config import SimConfig
from draftai.draft.needs import NeedModel
from draftai.draft.opponents import OpponentModel
from draftai.draft.recommender import RosterEvaluator, recommend
from draftai.draft.rollout import RolloutConfig, RolloutEngine
from draftai.draft.state import DraftState

FILL_FIRST = 1e6


@dataclass
class Decision:
    pick: int
    state: DraftState
    a: int
    b: int | None
    confidence: float | None
    edge: float | None


class Strategy:
    name = "base"
    expensive = False

    def prepare(self, sd: SeasonData, needs: NeedModel, evaluator: RosterEvaluator, model: OpponentModel,
                sim: SimConfig, rollout: RolloutConfig) -> None:
        self.sd, self.needs, self.evaluator = sd, needs, evaluator
        self.id_row = {s: i for i, s in enumerate(sd.board.players["sleeper_id"].to_list())}

    def choose(self, state: DraftState, avail: np.ndarray, counts: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Called before each draft."""


@dataclass
class KeyGreedy(Strategy):
    name: str = "vor"
    key: str = "vor"
    expensive: bool = False
    values: np.ndarray = field(default=None, repr=False)

    def prepare(self, sd, needs, evaluator, model, sim, rollout) -> None:
        super().prepare(sd, needs, evaluator, model, sim, rollout)
        p = sd.board.players
        if self.key == "adp":
            v = -p["adp"].to_numpy()
        elif self.key == "realised":
            v = sd.realised.season_value
        else:
            v = p[self.key].to_numpy()
        self.values = np.where(np.isfinite(v), v, -1e9).astype(float)

    def choose(self, state, avail, counts) -> int:
        fills = self.needs.fills(counts)[self.needs.type_of]
        score = np.where(avail, self.values + FILL_FIRST * fills, -np.inf)
        return int(score.argmax())


class GreedySurvival(Strategy):
    name = "greedy_survival"
    expensive = True

    def prepare(self, sd, needs, evaluator, model, sim, rollout) -> None:
        super().prepare(sd, needs, evaluator, model, sim, rollout)
        self.sim = sim

    def choose(self, state, avail, counts) -> int:
        rec = recommend(self.sd.board, state, self.sim, self.evaluator)
        return self.id_row[rec.best.sleeper_id]


class Rollout(Strategy):
    name = "rollout"
    expensive = True

    def prepare(self, sd, needs, evaluator, model, sim, rollout) -> None:
        super().prepare(sd, needs, evaluator, model, sim, rollout)
        self.engine = RolloutEngine(sd.board, model, rollout, sim, evaluator=evaluator)
        self.decisions: list[Decision] = []

    def reset(self) -> None:
        self.decisions = []

    def choose(self, state, avail, counts) -> int:
        res = self.engine.recommend(state)
        a = res.best.row
        b = res.candidates[1].row if len(res.candidates) > 1 else None
        self.decisions.append(Decision(state.current_pick, state.copy(), a, b, res.confidence, res.edge))
        return a


class Ranked(Strategy):
    """A learning-to-rank ranker over the greedy recommender's top candidates."""

    def __init__(self, name: str, stage: str, paths):
        self.name, self.stage, self.paths = name, stage, paths

    def prepare(self, sd, needs, evaluator, model, sim, rollout) -> None:
        from draftai.backtest.ltr import ranker_path
        from draftai.draft.ranker import Ranker

        super().prepare(sd, needs, evaluator, model, sim, rollout)
        self.sim = sim
        path = ranker_path(self.paths, sd.season, self.stage)
        if not path.exists():
            raise FileNotFoundError(f"no {self.stage} ranker for {sd.season}; run `draftai ltr` first")
        self.ranker = Ranker.load(path)
        key = self.ranker.meta.get("season_key")
        if key and sd.meta.get("cache_key") and key != sd.meta["cache_key"]:
            import warnings

            # Its features are relative to the board, so it still applies (e.g. a real draft with
            # other scoring); after a retrain, rerun `draftai ltr` to refit on the new boards.
            warnings.warn(f"the {self.stage} ranker for {sd.season} was trained on another board ({key})")

    def choose(self, state, avail, counts) -> int:
        from draftai.draft.ranker import query

        _, cands, X = query(self.sd.board, state, self.sim, self.evaluator, self.needs, counts)
        return self.ranker.choose(cands, X)


LTR_STRATEGIES = {"ltr": "real", "ltr_sim": "sim", "ltr_real_only": "real_only"}


def build(names: list[str], paths=None) -> list[Strategy]:
    table = {
        "adp": lambda: KeyGreedy("adp", "adp"),
        "points": lambda: KeyGreedy("points", "season_value"),
        "vor": lambda: KeyGreedy("vor", "vor"),
        "greedy_survival": GreedySurvival,
        "rollout": Rollout,
        "oracle": lambda: KeyGreedy("oracle", "realised"),
        **{n: (lambda n=n, st=st: Ranked(n, st, paths)) for n, st in LTR_STRATEGIES.items()},
    }
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}; choose from {sorted(table)}")
    return [table[n]() for n in names]
