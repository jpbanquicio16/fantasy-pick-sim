"""Offline mock drafts that draft themselves.

Every other team picks with the opponent model (the conditional logit the rollouts use),
so a mock is a sample from the engine's own picture of the room: tight to ADP early,
looser late, with some pull toward open lineup slots. Each pick's random number is seeded
by the mock's seed and the pick number, so undoing a pick and redoing it replays the room.

Your own picks can be entered by hand, taken from the recommender, or auto-drafted to the
end. ``projection`` scores the finished league: projected points per week from the
lineup optimiser, and P(playoffs) / P(title) from simulated seasons.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl

from draftai.draft.league import LeagueContext
from draftai.draft.needs import NeedModel
from draftai.draft.opponents import Market, OpponentModel, choose, probabilities
from draftai.draft.recommender import RosterEvaluator
from draftai.draft.state import DraftState
from draftai.value.board import Board
from draftai.value.lineup import eligibility_matrix, greedy_lineups, slot_specificity
from draftai.value.season import season_outcomes
from draftai.value.worlds import Worlds


@dataclass
class MockPick:
    pick: int
    slot: int
    sleeper_id: str
    name: str
    positions: list[str]
    adp: float
    by: str  # "you", "engine" (auto-pick for you) or "auto" (opponent model)


@dataclass
class MockDraft:
    ctx: LeagueContext
    board: Board
    model: OpponentModel
    seed: int = field(default_factory=lambda: random.randrange(2**31))
    state: DraftState = field(init=False)
    log: list[MockPick] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.state = self.ctx.new_state()
        self.needs = NeedModel(self.board.positions(), self.board.slots)
        self.market = Market(self.board.players["adp"].to_numpy(), self.needs)
        p = self.board.players
        self.ids = p["sleeper_id"].to_list()
        self.id_row = {s: i for i, s in enumerate(self.ids)}
        self._names = p["name"].to_list()
        self._pos = p["positions"].to_list()
        self._adp = p["adp"].to_list()

    # --- state ----------------------------------------------------------------------
    def team_name(self, slot: int) -> str:
        # Simulated teams are not your league-mates: the model has no per-manager behaviour.
        return "You" if slot == self.state.my_slot else f"Team {slot}"

    def _avail(self) -> np.ndarray:
        avail = np.ones(self.market.size, dtype=bool)
        for sid in self.state.drafted:
            row = self.id_row.get(sid)
            if row is not None:
                avail[self.market.col_of[row]] = False
        return avail

    def _counts(self, slot: int) -> np.ndarray:
        rows = [self.id_row[s] for s in self.state.roster(slot) if s in self.id_row]
        return self.needs.counts(rows) if rows else np.zeros(self.needs.n_types, dtype=np.int64)

    # --- picks ----------------------------------------------------------------------
    def pick(self, sleeper_id: str, by: str = "you") -> MockPick:
        slot = self.state.on_the_clock
        self.state.add_pick(sleeper_id)
        row = self.id_row.get(sleeper_id)
        mp = MockPick(self.state.version, slot, sleeper_id,
                      self._names[row] if row is not None else sleeper_id,
                      list(self._pos[row]) if row is not None else [],
                      float(self._adp[row]) if row is not None else float("nan"), by)
        self.log.append(mp)
        return mp

    def opponent_choice(self) -> str:
        """The opponent model's pick for the team on the clock (deterministic per pick)."""
        t = self.state.current_pick
        u = float(np.random.default_rng([self.seed, t]).random())
        col = choose(self.model, self.market, self._avail(), self._counts(self.state.on_the_clock), t, u)
        if col < 0:
            raise RuntimeError("no players left on the board")
        return self.ids[int(self.market.order[col])]

    def odds(self, top: int = 8) -> pl.DataFrame:
        """What the model thinks the team on the clock will take."""
        cols, p = probabilities(self.model, self.market, self._avail(), self._counts(self.state.on_the_clock),
                                self.state.current_pick)
        rows = self.market.order[cols]
        order = np.argsort(-p)[:top]
        return pl.DataFrame({"player": [self._names[r] for r in rows[order]], "prob": p[order].round(3)})

    def advance(self) -> list[MockPick]:
        """Let the other teams pick until you are on the clock (or the draft ends)."""
        made = []
        while not self.state.complete and not self.state.my_turn:
            made.append(self.pick(self.opponent_choice(), by="auto"))
        return made

    def run_to_end(self, chooser: Callable[[DraftState], str]) -> list[MockPick]:
        """Auto-draft the rest: ``chooser`` picks for you (e.g. the recommender's best)."""
        made = []
        while not self.state.complete:
            if self.state.my_turn:
                made.append(self.pick(chooser(self.state), by="engine"))
            else:
                made.append(self.pick(self.opponent_choice(), by="auto"))
        return made

    def undo_mine(self) -> None:
        """Take back your last pick and every auto pick after it."""
        mine = [i for i, m in enumerate(self.log) if m.slot == self.state.my_slot]
        if not mine:
            return
        while len(self.log) > mine[-1]:
            self.state.undo()
            self.log.pop()

    # --- results --------------------------------------------------------------------
    def log_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            [{"pick": m.pick, "team": self.team_name(m.slot), "player": m.name, "pos": "/".join(m.positions),
              "adp": round(m.adp, 1), "vs_adp": round(m.pick - m.adp, 1), "by": m.by} for m in self.log],
            schema={"pick": pl.Int64, "team": pl.Utf8, "player": pl.Utf8, "pos": pl.Utf8, "adp": pl.Float64,
                    "vs_adp": pl.Float64, "by": pl.Utf8},
        )

    def projection(self, evaluator: RosterEvaluator | None = None, worlds: Worlds | None = None,
                   n_worlds: int = 300, waiver_slots: int = 1) -> pl.DataFrame:
        return project_league(self.board, self.state, evaluator, worlds, n_worlds, waiver_slots,
                              self.ctx.playoff_teams, self.ctx.playoff_round_type, self.seed, self.team_name,
                              self.ctx.playoff_week_start)


def project_league(board: Board, state: DraftState, evaluator: RosterEvaluator | None = None,
                   worlds: Worlds | None = None, n_worlds: int = 300, waiver_slots: int = 1,
                   playoff_teams: int = 6, round_type: int = 0, seed: int = 0,
                   team_name: Callable[[int], str] = lambda s: f"Team {s}",
                   playoff_week_start: int | None = None) -> pl.DataFrame:
    """Every team's projected points per week and simulated playoff and title odds.

    Each simulated season draws every player's season (worlds), sets the best lineup each
    week, adds week-to-week noise from each starter's locked-score spread, and plays a
    random head-to-head schedule and the bracket.
    """
    from draftai.draft.rollout import drafted_per_league, waiver_level

    ev = evaluator or RosterEvaluator(board)
    worlds = worlds if worlds is not None else Worlds.draw(board, n_worlds, seed)
    n = min(n_worlds, worlds.n)
    id_row = {s: i for i, s in enumerate(board.players["sleeper_id"].to_list())}
    rosters = state.rosters()
    slots = sorted(rosters)
    size = max((len(r) for r in rosters.values()), default=0)
    rows = np.full((len(slots), size), -1, dtype=np.int64)
    for i, s in enumerate(slots):
        rr = [id_row[p] for p in rosters[s] if p in id_row]
        rows[i, : len(rr)] = rr
    elig_all = eligibility_matrix(board.positions(), board.slots)
    spec = slot_specificity(board.slots)
    waiver = waiver_level(board, drafted_per_league(board)).astype(np.float32)
    sd = board.players["lockin_sd_3"].to_numpy() if "lockin_sd_3" in board.players.columns \
        else 0.6 * board.players["fp_sd_game"].to_numpy()
    k_slots = len(board.slots)
    # Week-to-week noise of a team's score: its likely starters' locked-score spreads.
    team_sd = np.array([np.sqrt(np.sort(sd[r[r >= 0]] ** 2)[::-1][:k_slots].sum()) if (r >= 0).any() else 0.0
                        for r in rows])
    T, K = len(slots), worlds.weeks
    lineup = np.zeros((n, T, K))
    widx = np.arange(n)
    for i in range(T):
        mine = np.tile(rows[i], (n, 1))
        vals = worlds.values(widx, mine)
        elig = elig_all[np.maximum(mine, 0)] & (mine >= 0)[..., None]
        if waiver_slots:
            vals = np.concatenate([vals, np.broadcast_to(waiver, (n, waiver_slots, K))], axis=1)
            elig = np.concatenate([elig, np.ones((n, waiver_slots, elig.shape[2]), dtype=bool)], axis=1)
        lineup[:, i, :] = greedy_lineups(vals, elig, spec)
    rng = np.random.default_rng([seed, 7])
    scores = lineup + rng.standard_normal(lineup.shape) * team_sd[None, :, None]
    active = np.array([wk for wk, wt in zip(board.weeks, board.weights) if wt > 0])
    is_po = active >= playoff_week_start if playoff_week_start else worlds.weights > 1.0
    reg, po = scores[:, :, ~is_po], scores[:, :, is_po]
    out = season_outcomes(reg, po, playoff_teams, n, rng, round_type)
    weeks_weight = float(board.weights.sum())
    proj = [ev.value([int(x) for x in r if x >= 0]) / weeks_weight for r in rows]
    df = pl.DataFrame({
        "slot": slots, "team": [team_name(s) for s in slots],
        "proj_per_week": np.round(proj, 1),
        "p_playoffs": np.round(out.p_playoffs, 3), "p_title": np.round(out.p_title, 3),
        "mean_seed": np.round(out.mean_seed, 1),
    })
    return df.sort("proj_per_week", descending=True).with_row_index("proj_rank", 1)
