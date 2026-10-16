"""Greedy value-plus-survival recommender (the MVP decision rule).

For each candidate p, with your roster R and next pick t':

    delta_p   = V(R + p) - V(R)                                  marginal roster value
    E_next(p) = sum_i delta_{q_i | p} * s_i * prod_{j<i} (1 - s_j)  + P(none survive) * F
    score_p   = delta_p + E_next(p) - lambda * SD(season value of p)

where q_1, q_2, ... are the other candidates sorted by their marginal value given p is
already on the roster, s_i is each one's survival to t', and F is a fallback value.
With two candidates this reduces exactly to the "now or later" rule:
take A over B iff (1 - s_A)(delta_A - F) > (1 - s_B)(delta_B - F).

V(R) is the season sum over weeks of the best lineup's expected Lock-In value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from draftai.config import SimConfig
from draftai.draft.state import DraftState
from draftai.draft.survival import survival
from draftai.value.board import Board
from draftai.value.lineup import eligibility_matrix, optimise


MIN_REACH = 0.05  # when planning ahead, ignore players this unlikely to reach your pick


class RosterEvaluator:
    """Season value of a set of board rows, with memoisation."""

    def __init__(self, board: Board):
        active = board.weights > 0
        self.W = board.weekly[:, active]
        self.w = board.weights[active]
        self.elig = eligibility_matrix(board.positions(), board.slots)
        self._cache: dict[frozenset, float] = {}

    def value(self, rows) -> float:
        key = frozenset(rows)
        if key in self._cache:
            return self._cache[key]
        idx = np.fromiter(key, dtype=int)
        total = 0.0
        if idx.size:
            W, E = self.W[idx], self.elig[idx]
            for k in range(W.shape[1]):
                total += self.w[k] * optimise(W[:, k], E)[0]
        self._cache[key] = total
        return total


def expected_best(values: list[float], surv: list[float], floor: float) -> float:
    """Expected value of the best survivor, taking candidates in descending value."""
    acc, none = 0.0, 1.0
    for v, s in sorted(zip(values, surv), key=lambda t: -t[0]):
        if v <= floor:
            break
        acc += none * s * v
        none *= 1.0 - s
    return acc + none * floor


@dataclass
class Candidate:
    row: int
    sleeper_id: str
    name: str
    positions: list[str]
    delta: float
    survival: float
    e_next: float
    score: float
    adp: float
    risk: float
    reach: float = 1.0  # chance he is still there at the pick being planned
    detail: dict = field(default_factory=dict)


@dataclass
class Recommendation:
    version: int
    pick: int  # the pick being planned: the current pick, or your next one
    next_pick: int | None  # your pick after that
    roster_value: float
    candidates: list[Candidate]
    fallback: float
    weeks_weight: float

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def per_week(self, x: float) -> float:
        return x / max(self.weeks_weight, 1e-9)

    def table(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {"rank": i + 1, "player": c.name, "pos": "/".join(c.positions), "adp": c.adp,
                 "score": c.score, "delta": c.delta, "reach": round(c.reach, 3), "survival": round(c.survival, 3),
                 "e_next": c.e_next,
                 "delta_per_week": self.per_week(c.delta), "sleeper_id": c.sleeper_id}
                for i, c in enumerate(self.candidates)
            ]
        )


def _pool(board: Board, avail: np.ndarray, n: int) -> list[int]:
    p = board.players
    rows = np.flatnonzero(avail)
    out: list[int] = []
    for col, desc in (("season_value", True), ("vor", True), ("adp", False)):
        vals = p[col].to_numpy()[rows]
        order = rows[np.argsort(-vals if desc else vals, kind="stable")]
        out.extend(int(r) for r in order[: n if col != "adp" else n // 2])
    return list(dict.fromkeys(out))


def recommend(board: Board, state: DraftState, cfg: SimConfig, evaluator: RosterEvaluator | None = None) -> Recommendation:
    ev = evaluator or RosterEvaluator(board)
    p = board.players
    ids = p["sleeper_id"].to_list()
    id_row = {s: i for i, s in enumerate(ids)}
    taken = state.drafted
    avail = np.array([s not in taken for s in ids])
    mine = [id_row[s] for s in state.roster() if s in id_row]
    upcoming = state.my_upcoming()
    if not upcoming:
        return Recommendation(state.version, state.current_pick, None, ev.value(mine), [], 0.0, float(board.weights.sum()))
    # Plan your next pick; when it is not your turn yet, this is the anticipatory view.
    t = upcoming[0]
    t_next = state.next_pick_after(t)

    base = ev.value(mine)
    adp = p["adp"].to_numpy()
    reach_all = survival(adp, state.current_pick, t, cfg.adp_sigma_a, cfg.adp_sigma_b)
    if t > state.current_pick:
        # Planning ahead: only players with a real chance of lasting to your pick.
        likely = avail & (reach_all >= MIN_REACH)
        if likely.sum() >= cfg.shortlist:
            avail = likely
    pool = _pool(board, avail, cfg.candidate_pool)
    delta = {q: ev.value([*mine, q]) - base for q in pool}
    s_all = survival(adp[pool], t, t_next, cfg.adp_sigma_a, cfg.adp_sigma_b)
    s = dict(zip(pool, s_all.tolist()))
    reach = {q: float(reach_all[q]) for q in pool}
    floor = max(0.0, float(np.quantile(list(delta.values()), 0.1))) if delta else 0.0
    risk_sd = p["season_sd"].to_numpy()

    shortlist = sorted(pool, key=lambda q: -delta[q])[: cfg.shortlist]
    shortlist += [q for q in sorted(pool, key=lambda q: adp[q])[:3] if q not in shortlist]
    cands = []
    for q in pool:
        if t_next is None:
            e_next = 0.0
        elif q in shortlist:
            base_q = base + delta[q]
            others = [r for r in pool if r != q]
            cond = [ev.value([*mine, q, r]) - base_q for r in others]
            e_next = expected_best(cond, [s[r] for r in others], floor)
        else:
            others = [r for r in pool if r != q]
            e_next = expected_best([delta[r] for r in others], [s[r] for r in others], floor)
        score = delta[q] + e_next - cfg.risk_lambda * risk_sd[q]
        cands.append(
            Candidate(
                row=q, sleeper_id=ids[q], name=p["name"][q], positions=list(p["positions"][q]),
                delta=delta[q], survival=s[q] if t_next else 0.0, e_next=e_next, score=score,
                adp=float(adp[q]), risk=float(risk_sd[q]), reach=reach[q],
            )
        )
    cands.sort(key=lambda c: -c.score)
    return Recommendation(
        version=state.version, pick=t, next_pick=t_next, roster_value=base, candidates=cands,
        fallback=floor, weeks_weight=float(board.weights.sum()),
    )
