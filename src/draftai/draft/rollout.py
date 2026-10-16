"""Monte Carlo rollout recommender (V4).

At your pick, for each shortlisted candidate: simulate the rest of the draft in N worlds
(opponents from the fitted conditional logit, your later picks from a greedy base
policy), evaluate your final roster in each world, and recommend the candidate with the
highest mean. This is the classical rollout algorithm: up to simulation error it does at
least as well as the base policy it rolls out.

* Common random numbers: every candidate sees the same season outcomes and the same
  opponent random numbers, so differences between candidates are measured, not sampled.
* Racing: after a first batch of worlds, candidates whose paired difference to the leader
  is clearly negative are dropped and the remaining worlds go to the close ones.
* Simulated to your last pick (opponent picks after it cannot change your roster), with
  no truncation and no terminal-value estimate.

    confidence = P(V(R_T | A) > V(R_T | B))   over the shared worlds

where A is the recommendation and B the runner-up: "in 81% of simulated drafts and
seasons, taking A finished with a better roster than taking B".

The shortlist comes from the greedy value-plus-survival recommender, which also supplies
the explanation fields. When you are not on the clock, the picks before yours are
simulated first and each candidate is scored only in the worlds where he is still there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from draftai.config import SimConfig
from draftai.draft.needs import NeedModel
from draftai.draft.opponents import Market, OpponentModel, choose_batch
from draftai.draft.recommender import Recommendation, RosterEvaluator, recommend
from draftai.draft.state import DraftState
from draftai.value.board import Board
from draftai.value.lineup import NON_STARTING, eligibility_matrix, greedy_lineups, slot_specificity
from draftai.value.worlds import Worlds

FILL_FIRST = 1e6  # base policy: fill an open starting slot before adding depth


@dataclass
class RolloutConfig:
    worlds: int = 600
    shortlist: int = 8
    race_after: int = 200
    race_z: float = 2.5
    waiver_slots: int = 1
    seed: int = 11

    @classmethod
    def from_sim(cls, sim: SimConfig) -> "RolloutConfig":
        return cls(worlds=sim.rollout_worlds, shortlist=sim.rollout_shortlist, race_after=sim.rollout_race_after,
                   waiver_slots=sim.waiver_slots, seed=sim.seed)


@dataclass
class RolloutCandidate:
    row: int
    sleeper_id: str
    name: str
    positions: list[str]
    mean: float  # expected final-roster season value (FP, playoff weeks weighted)
    se: float
    worlds: int  # worlds in which he was evaluated
    reach: float  # share of worlds in which he is still there at your pick
    p_best: float  # share of first-batch worlds in which he beat every other candidate still there
    greedy_rank: int
    dropped: bool = False


@dataclass
class RolloutResult:
    version: int
    pick: int
    next_pick: int | None
    candidates: list[RolloutCandidate]
    confidence: float | None
    edge: float | None  # mean paired difference A - B over shared worlds (season FP)
    edge_se: float | None
    weeks_weight: float
    survival: dict[str, float]  # sleeper_id -> simulated P(still there at your next pick)
    greedy: Recommendation
    elapsed: float
    worlds: int

    @property
    def best(self) -> RolloutCandidate | None:
        return self.candidates[0] if self.candidates else None

    def per_week(self, x: float) -> float:
        return x / max(self.weeks_weight, 1e-9)

    def table(self) -> pl.DataFrame:
        top = self.candidates[0].mean if self.candidates else 0.0
        return pl.DataFrame(
            [
                {"rank": i + 1, "player": c.name, "pos": "/".join(c.positions),
                 "value_per_week": round(self.per_week(c.mean), 2),
                 "gap_per_week": round(self.per_week(top - c.mean), 2),
                 "se_per_week": round(self.per_week(c.se), 2), "p_best": round(c.p_best, 3),
                 "reach": round(c.reach, 3), "survival_sim": self.survival.get(c.sleeper_id),
                 "worlds": c.worlds, "greedy_rank": c.greedy_rank, "dropped": c.dropped, "sleeper_id": c.sleeper_id}
                for i, c in enumerate(self.candidates)
            ]
        )


def waiver_level(board: Board, drafted: int, width: int = 10) -> np.ndarray:
    """Expected weekly value of the best undrafted players: what a waiver claim is worth."""
    active = board.weights > 0
    order = np.argsort(-board.players["season_value"].to_numpy())
    pool = order[drafted: drafted + width]
    if pool.size == 0:
        return np.zeros(int(active.sum()))
    return board.weekly[pool][:, active].mean(axis=0)


def drafted_per_league(board: Board) -> int:
    return board.teams * sum(1 for s in board.roster_positions if s not in NON_STARTING or s == "BN")


class RolloutEngine:
    """Holds everything that does not change between picks: worlds, market, base policy."""

    def __init__(self, board: Board, opponents: OpponentModel, cfg: RolloutConfig, sim: SimConfig,
                 worlds: Worlds | None = None, evaluator: RosterEvaluator | None = None):
        self.board, self.model, self.cfg, self.sim = board, opponents, cfg, sim
        self.needs = NeedModel(board.positions(), board.slots)
        self.market = Market(board.players["adp"].to_numpy(), self.needs)
        self.worlds = worlds if worlds is not None else Worlds.draw(board, cfg.worlds, cfg.seed)
        self.evaluator = evaluator or RosterEvaluator(board)
        self.ids = board.players["sleeper_id"].to_list()
        self.id_row = {s: i for i, s in enumerate(self.ids)}
        vor = board.players["vor"].to_numpy().astype(float)
        self.policy_col = np.where(np.isfinite(vor), vor, -1e9)[self.market.order]
        self.elig = eligibility_matrix(board.positions(), board.slots)
        self.spec = slot_specificity(board.slots)
        self.waiver = waiver_level(board, drafted_per_league(board)).astype(np.float32)

    # --- simulation pieces ----------------------------------------------------------
    def _start(self, state: DraftState, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = self.market
        avail = np.ones(m.size, dtype=bool)
        counts = np.zeros((state.teams + 1, self.needs.n_types), dtype=np.int64)
        for slot, ids in state.rosters().items():
            for sid in ids:
                row = self.id_row.get(sid)
                if row is not None:
                    avail[m.col_of[row]] = False
                    counts[slot, self.needs.type_of[row]] += 1
        for sid in state.drafted:
            row = self.id_row.get(sid)
            if row is not None:
                avail[m.col_of[row]] = False
        size = len(state.my_picks())
        mine = np.full(size, -1, dtype=np.int64)
        rows = [self.id_row[s] for s in state.roster() if s in self.id_row]
        mine[: len(rows)] = rows
        return (np.tile(avail, (n, 1)), np.tile(counts, (n, 1, 1)), np.tile(mine, (n, 1)),
                np.full(n, len(rows), dtype=np.int64))

    def _run(self, picks: list[tuple[int, int]], my_slot: int, avail, counts, mine, nmine, uniforms: dict[int, np.ndarray],
             snapshot_at: int | None = None) -> np.ndarray | None:
        """Advance every world through ``picks`` (overall pick, owning slot), in place."""
        m, n = self.market, avail.shape[0]
        rows = np.arange(n)
        snap = None
        for t, owner in picks:
            if t == snapshot_at:
                snap = avail.copy()
            fills = self.needs.fills_batch(counts[:, owner])
            if owner == my_slot:
                score = np.where(avail, self.policy_col + FILL_FIRST * fills[:, m.type_col], -np.inf)
                col = score.argmax(axis=1)
                col = np.where(np.isfinite(score[rows, col]), col, -1)
            else:
                col = choose_batch(self.model, m, avail, fills, t, uniforms[t])
            ok = col >= 0
            r, c = rows[ok], col[ok]
            avail[r, c] = False
            counts[r, owner, m.type_col[c]] += 1
            if owner == my_slot:
                mine[r, nmine[r]] = m.order[c]
                nmine[r] += 1
        return snap

    def season_values(self, world_idx: np.ndarray, mine: np.ndarray) -> np.ndarray:
        """Final-roster season value in each world: weekly best lineups, playoff-weighted."""
        vals = self.worlds.values(world_idx, mine)
        elig = self.elig[np.maximum(mine, 0)] & (mine >= 0)[..., None]
        ws = self.cfg.waiver_slots
        if ws:
            n = mine.shape[0]
            vals = np.concatenate([vals, np.broadcast_to(self.waiver, (n, ws, self.waiver.size))], axis=1)
            elig = np.concatenate([elig, np.ones((n, ws, elig.shape[2]), dtype=bool)], axis=1)
        return greedy_lineups(vals, elig, self.spec) @ self.worlds.weights

    def _uniforms(self, state: DraftState, batch: int, worlds: np.ndarray) -> dict[int, np.ndarray]:
        rng = np.random.default_rng([self.cfg.seed, state.version, batch])
        u = rng.random((state.total_picks + 1, worlds.size))
        return {t: u[t] for t in range(state.current_pick, state.total_picks + 1)}

    # --- the decision -----------------------------------------------------------------
    def recommend(self, state: DraftState, greedy: Recommendation | None = None) -> RolloutResult:
        start = time.time()
        greedy = greedy or recommend(self.board, state, self.sim, self.evaluator)
        weeks_weight = float(self.board.weights.sum())
        upcoming = state.my_upcoming()
        if not upcoming or not greedy.candidates:
            return RolloutResult(state.version, state.current_pick, None, [], None, None, None, weeks_weight, {},
                                 greedy, time.time() - start, 0)
        t = upcoming[0]
        t_next = state.next_pick_after(t)
        last = upcoming[-1]
        owners = state.owners
        before = [(p, owners[p - 1]) for p in range(state.current_pick, t)]
        after = [(p, owners[p - 1]) for p in range(t + 1, last + 1)]
        shortlist = [c.row for c in greedy.candidates[: self.cfg.shortlist]]
        k = len(shortlist)
        n_total = min(self.cfg.worlds, self.worlds.n)
        first = min(self.cfg.race_after, n_total) if k > 1 else n_total
        phases = [np.arange(first)] + ([np.arange(first, n_total)] if n_total > first else [])

        V = np.full((k, n_total), np.nan)
        alive = np.ones(k, dtype=bool)
        snaps: dict[int, np.ndarray] = {}  # candidate -> availability at your next pick (first batch)
        at_pick: np.ndarray | None = None  # availability at your pick (first batch)
        for b, widx in enumerate(phases):
            if b > 0:
                self._race(V[:, : phases[0].size], alive)
            if not alive.any():
                break
            uni = self._uniforms(state, b, widx)
            A, C, M, NM = self._start(state, widx.size)
            self._run(before, state.my_slot, A, C, M, NM, uni)
            if b == 0:
                at_pick = A.copy()
            for j, row in enumerate(shortlist):
                if not alive[j]:
                    continue
                col = int(self.market.col_of[row])
                valid = A[:, col].copy()
                a, c, mm, nm = A.copy(), C.copy(), M.copy(), NM.copy()
                rr = np.arange(widx.size)
                a[:, col] = False
                c[rr, state.my_slot, self.needs.type_of[row]] += 1
                mm[rr, nm] = row
                nm += 1
                snap = self._run(after, state.my_slot, a, c, mm, nm, uni, snapshot_at=t_next)
                if b == 0 and snap is not None:
                    snaps[j] = snap
                v = self.season_values(widx, mm)
                V[j, widx] = np.where(valid, v, np.nan)

        cands = self._summarise(V, alive, shortlist, phases[0].size)
        conf = edge = edge_se = None
        if len(cands) > 1:
            ia, ib = shortlist.index(cands[0].row), shortlist.index(cands[1].row)
            both = ~np.isnan(V[ia]) & ~np.isnan(V[ib])
            if both.any():
                d = V[ia, both] - V[ib, both]
                conf = float(((d > 0) + 0.5 * (d == 0)).mean())
                edge = float(d.mean())
                edge_se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        survival: dict[str, float] = {}
        if cands and at_pick is not None:
            # P(still there at your next pick | there at this one), read from the branch in which
            # you take the recommendation; for the recommendation himself, from the runner-up's.
            branch = [shortlist.index(c.row) for c in cands[:2] if shortlist.index(c.row) in snaps]
            for q in greedy.candidates:
                use = [j for j in branch if shortlist[j] != q.row]
                if not use:
                    continue
                snap, col = snaps[use[0]], self.market.col_of[q.row]
                den = at_pick[:, col].sum()
                if den:
                    survival[q.sleeper_id] = float((snap[:, col] & at_pick[:, col]).sum() / den)
        return RolloutResult(state.version, t, t_next, cands, conf, edge, edge_se, weeks_weight, survival, greedy,
                             time.time() - start, n_total)

    def _race(self, V1: np.ndarray, alive: np.ndarray) -> None:
        """Drop candidates whose paired difference to the leader is clearly negative."""
        n_valid = (~np.isnan(V1)).sum(axis=1)
        means = np.where(alive & (n_valid > 0), np.nansum(V1, axis=1) / np.maximum(n_valid, 1), -np.inf)
        if not np.isfinite(means).any():
            return
        lead = int(np.argmax(means))
        for j in np.flatnonzero(alive):
            if j == lead:
                continue
            both = ~np.isnan(V1[j]) & ~np.isnan(V1[lead])
            if both.sum() < 20:
                continue
            d = V1[j, both] - V1[lead, both]
            if d.mean() + self.cfg.race_z * d.std(ddof=1) / np.sqrt(d.size) < 0:
                alive[j] = False

    def _summarise(self, V: np.ndarray, alive: np.ndarray, shortlist: list[int], first: int) -> list[RolloutCandidate]:
        p = self.board.players
        V1 = V[:, :first]
        # Best among the candidates still there in each world (planning ahead, some are not).
        any_valid = ~np.isnan(V1).all(axis=0)
        best_counts = np.zeros(len(shortlist))
        if any_valid.any():
            winners = np.argmax(np.where(np.isnan(V1[:, any_valid]), -np.inf, V1[:, any_valid]), axis=0)
            best_counts = np.bincount(winners, minlength=len(shortlist)) / any_valid.sum()
        out = []
        for j, row in enumerate(shortlist):
            v = V[j][~np.isnan(V[j])]
            evaluated = V.shape[1] if alive[j] else first
            out.append(RolloutCandidate(
                row=row, sleeper_id=self.ids[row], name=p["name"][row], positions=list(p["positions"][row]),
                mean=float(v.mean()) if v.size else float("-inf"),
                se=float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
                worlds=int(v.size), reach=float(v.size / max(evaluated, 1)), p_best=float(best_counts[j]),
                greedy_rank=j + 1, dropped=not alive[j],
            ))
        out.sort(key=lambda c: (c.dropped, -c.mean))
        return out
