"""What the API serves: the board, the decision engine and draft sessions.

No web framework here and no decision logic: sessions hold a ``DraftState`` (a mock whose
other teams draft themselves, a manual draft, or a live Sleeper draft), and every answer
comes from the same recommender, rollouts and explanation cards as the CLI and the
Streamlit app. Everything returned is plain JSON-safe data.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from draftai.config import SimConfig
from draftai.draft.explain import card, minutes_text, role_text
from draftai.draft.league import LeagueContext
from draftai.draft.mock import MockDraft
from draftai.draft.opponents import OpponentModel
from draftai.draft.recommender import Recommendation, RosterEvaluator, recommend
from draftai.draft.rollout import RolloutConfig, RolloutEngine, RolloutResult
from draftai.draft.state import DraftState
from draftai.draft.survival import survival
from draftai.models.explain import top_drivers
from draftai.value.board import Board
from draftai.value.lineup import optimise

BOARD_COLUMNS = ["sleeper_id", "name", "team_now", "positions", "adp", "fp_pg", "fp_p10", "fp_p90", "lockin_3",
                 "lockin_premium_3", "mpg_hat", "share_hat", "games_proj", "vor_per_week", "season_value",
                 "season_p10", "season_p90", "rookie", "injury_status", "age"]


def clean(v):
    """JSON-safe: NaN and infinities become null, NumPy scalars plain Python."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v


class Engine:
    """The board and everything that does not change between picks."""

    def __init__(self, ctx: LeagueContext, board: Board, sim: SimConfig, opponents: OpponentModel):
        self.ctx, self.sim, self.opponents = ctx, sim, opponents
        self.lock = threading.RLock()
        self.set_board(board)

    def set_board(self, board: Board) -> None:
        with self.lock:
            self.board = board
            self.evaluator = RosterEvaluator(board)
            self.ids = board.players["sleeper_id"].to_list()
            self.row_of = {s: i for i, s in enumerate(self.ids)}
            self._rollout: RolloutEngine | None = None
            self._rows = board.players.to_dicts()

    def rollout(self) -> RolloutEngine:
        with self.lock:
            if self._rollout is None:
                self._rollout = RolloutEngine(self.board, self.opponents, RolloutConfig.from_sim(self.sim), self.sim,
                                              evaluator=self.evaluator)
            return self._rollout

    def row(self, sleeper_id: str) -> dict | None:
        i = self.row_of.get(sleeper_id)
        return self._rows[i] if i is not None else None

    # --- the board ---------------------------------------------------------------------
    def board_table(self) -> list[dict]:
        cols = [c for c in BOARD_COLUMNS if c in self.board.players.columns]
        return [clean(r) for r in self.board.players.select(cols).with_row_index("row").to_dicts()]

    def player(self, sleeper_id: str) -> dict | None:
        r = self.row(sleeper_id)
        if r is None:
            return None
        i = self.row_of[sleeper_id]
        active = self.board.weights > 0
        keep = {k: r.get(k) for k in BOARD_COLUMNS + ["lockin_2", "lockin_4", "vor", "season_sd", "fp_sd_game",
                                                     "p_game_40", "p_game_50", "rookie_source", "comps",
                                                     "draft_pick", "injury_body_part", "years_exp"] if k in r}
        return clean({
            **keep,
            "minutes": minutes_text(r),
            "role": role_text(r),
            "drivers": [{"group": g, "value": v} for g, v in top_drivers(r, "fp", 4)],
            "weekly": {"weeks": [w for w, a in zip(self.board.weeks, active) if a],
                       "value": self.board.weekly[i, active].tolist()},
        })

    def league(self) -> dict:
        c = self.ctx
        return clean({"season": c.season, "name": c.league_name, "teams": c.teams, "rounds": c.rounds,
                      "draft_type": c.draft_type, "reversal_round": c.reversal_round, "my_slot": c.my_slot,
                      "roster_positions": c.roster_positions, "slots": self.board.slots,
                      "playoff_teams": c.playoff_teams, "bonus_stacking": c.scoring.bonus_stacking,
                      "draft_id": c.draft_id, "players": self.board.players.height,
                      "weeks_weight": float(self.board.weights.sum())})


@dataclass
class Session:
    id: str
    kind: str  # "mock", "manual" or "live"
    ctx: LeagueContext
    state: DraftState
    mock: MockDraft | None = None
    live: object | None = None  # live.poller.LiveDraft
    auto: bool = True
    by: dict[int, str] = field(default_factory=dict)  # pick number -> "you", "engine", "auto", "manual", "live"
    listeners: list[Callable[[], None]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    _rec: tuple[tuple, Recommendation] | None = None  # keyed by the picks so far (an undo changes them)
    _ro: tuple[tuple, RolloutResult] | None = None

    def notify(self) -> None:
        for f in list(self.listeners):
            f()


class DraftService:
    def __init__(self, engine: Engine, live_factory: Callable[[str], tuple[LeagueContext, object]] | None = None):
        self.engine = engine
        self.sessions: dict[str, Session] = {}
        self.live_factory = live_factory

    # --- sessions ----------------------------------------------------------------------
    def get(self, sid: str) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        return s

    def create_mock(self, slot: int | None = None, seed: int | None = None, auto: bool = True) -> Session:
        ctx = replace(self.engine.ctx, my_slot=slot or self.engine.ctx.my_slot or 1)
        mock = MockDraft(ctx, self.engine.board, self.engine.opponents, **({"seed": seed} if seed is not None else {}))
        s = Session(uuid.uuid4().hex[:12], "mock", ctx, mock.state, mock=mock, auto=auto)
        self.sessions[s.id] = s
        if auto:
            self._advance(s)
        return s

    def create_manual(self, slot: int | None = None) -> Session:
        ctx = replace(self.engine.ctx, my_slot=slot or self.engine.ctx.my_slot or 1)
        s = Session(uuid.uuid4().hex[:12], "manual", ctx, ctx.new_state(), auto=False)
        self.sessions[s.id] = s
        return s

    def create_live(self, draft_id: str) -> Session:
        if self.live_factory is None:
            raise RuntimeError("live drafts need a Sleeper connection")
        ctx, live = self.live_factory(draft_id)
        live.poll()
        s = Session(uuid.uuid4().hex[:12], "live", ctx, live.state, live=live, auto=False)
        s.by = {i + 1: "live" for i in range(len(live.state.picks))}
        self.sessions[s.id] = s
        return s

    # --- actions -------------------------------------------------------------------------
    def _advance(self, s: Session) -> None:
        for mp in s.mock.advance():
            s.by[mp.pick] = "auto"

    def pick(self, s: Session, sleeper_id: str) -> None:
        with s.lock:
            if sleeper_id not in self.engine.row_of:
                raise ValueError(f"{sleeper_id} is not on the board")
            if sleeper_id in s.state.drafted:
                raise ValueError("already drafted")
            if s.kind == "live":
                raise ValueError("a live draft follows Sleeper; picks are made there")
            if s.kind == "mock":
                if not s.state.my_turn:
                    raise ValueError("not your pick: the other teams draft themselves")
                mp = s.mock.pick(sleeper_id)
                s.by[mp.pick] = "you"
                if s.auto:
                    self._advance(s)
            else:
                s.state.add_pick(sleeper_id)
                s.by[s.state.version] = "you" if s.state.pick_slots[-1] == s.state.my_slot else "manual"
        s.notify()

    def advance(self, s: Session) -> None:
        with s.lock:
            if s.kind == "mock":
                self._advance(s)
        s.notify()

    def undo(self, s: Session) -> None:
        with s.lock:
            if s.kind == "mock":
                s.mock.undo_mine()
            elif s.kind == "manual":
                s.state.undo()
            s.by = {k: v for k, v in s.by.items() if k <= s.state.version}
        s.notify()

    def autodraft(self, s: Session, use_rollout: bool = False) -> None:
        if s.kind != "mock":
            raise ValueError("auto-draft is for mocks")
        e = self.engine

        def chooser(state: DraftState) -> str:
            rec = recommend(e.board, state, e.sim, e.evaluator)
            if use_rollout:
                with e.lock:
                    return e.rollout().recommend(state, rec).best.sleeper_id
            return rec.best.sleeper_id

        with s.lock:
            for mp in s.mock.run_to_end(chooser):
                s.by[mp.pick] = mp.by
        s.notify()

    def sync(self, s: Session) -> bool:
        """Live: poll Sleeper; True if a pick arrived."""
        if s.kind != "live":
            return False
        with s.lock:
            changed = s.live.poll()
            if changed:
                s.state = s.live.state
                s.by = {i + 1: s.by.get(i + 1, "live") for i in range(len(s.state.picks))}
        if changed:
            s.notify()
        return changed

    # --- views ---------------------------------------------------------------------------
    def team_name(self, s: Session, slot: int) -> str:
        if slot == s.state.my_slot:
            return "You"
        return s.ctx.slot_names.get(slot) or f"Team {slot}"

    def lineup(self, rows: list[int]) -> list[dict]:
        """Your roster in starting slots (season-best assignment) and on the bench."""
        e = self.engine
        slots = e.board.slots
        if not rows:
            return [{"slot": sl, "sleeper_id": None, "name": None} for sl in slots]
        W = e.board.weekly[rows][:, e.board.weights > 0] @ e.board.weights[e.board.weights > 0]
        _, pairs = optimise(W, e.evaluator.elig[rows])
        used = {}
        for i, j in pairs:
            used[j] = rows[i]
        out = []
        for j, sl in enumerate(slots):
            r = used.get(j)
            out.append({"slot": sl, "sleeper_id": e.ids[r] if r is not None else None,
                        "name": e._rows[r]["name"] if r is not None else None})
        starters = set(used.values())
        out += [{"slot": "BN", "sleeper_id": e.ids[r], "name": e._rows[r]["name"]} for r in rows if r not in starters]
        return out

    def state_view(self, s: Session) -> dict:
        e = self.engine
        st = s.state
        picks = []
        for i, (pid, slot) in enumerate(zip(st.picks, st.pick_slots)):
            r = e.row(pid)
            picks.append({"pick": i + 1, "round": i // st.teams + 1, "slot": slot, "team": self.team_name(s, slot),
                          "sleeper_id": pid, "name": r["name"] if r else pid,
                          "positions": list(r["positions"]) if r else [], "adp": clean(r["adp"]) if r else None,
                          "by": s.by.get(i + 1, "")})
        upcoming = st.my_upcoming()
        nxt = upcoming[0] if upcoming else None
        after = st.next_pick_after(nxt) if nxt else None
        drafted = st.drafted
        adp = e.board.players["adp"].to_numpy()
        surv = survival(adp, st.current_pick, nxt if nxt and nxt > st.current_pick else after, e.sim.adp_sigma_a,
                        e.sim.adp_sigma_b) if not st.complete else np.zeros(len(e.ids))
        mine = [e.row_of[p] for p in st.roster() if p in e.row_of]
        return clean({
            "id": s.id, "kind": s.kind, "version": st.version, "teams": st.teams, "rounds": st.rounds,
            "my_slot": st.my_slot, "current_pick": st.current_pick, "total_picks": st.total_picks,
            "complete": st.complete, "on_the_clock": st.on_the_clock,
            "on_the_clock_team": self.team_name(s, st.on_the_clock) if st.on_the_clock else None,
            "my_turn": st.my_turn, "my_next_pick": nxt, "picks_until_mine": st.picks_until_mine(),
            "survival_to": nxt if nxt and nxt > st.current_pick else after,
            "my_picks": st.my_picks(), "picks": picks,
            "drafted": sorted(drafted),
            "survival": {sid: float(surv[i]) for i, sid in enumerate(e.ids) if sid not in drafted},
            "roster": self.lineup(mine),
            "seed": s.mock.seed if s.mock else None, "auto": s.auto,
            "draft_status": getattr(s.live, "status", None) if s.live else None,
        })

    def recommendation(self, s: Session, use_rollout: bool = False, top: int = 12) -> dict:
        e = self.engine
        with s.lock:
            st = s.state.copy()
        if st.complete or not st.my_upcoming():
            return {"version": st.version, "pick": None, "candidates": [], "card": None, "rollout": None}
        key = tuple(st.picks)
        if s._rec and s._rec[0] == key:
            rec = s._rec[1]
        else:
            rec = recommend(e.board, st, e.sim, e.evaluator)
            s._rec = (key, rec)
        ro = None
        if use_rollout:
            if s._ro and s._ro[0] == key:
                ro = s._ro[1]
            else:
                with e.lock:
                    ro = e.rollout().recommend(st, rec)
                s._ro = (key, ro)
        best = rec.candidates[0].score if rec.candidates else 0.0
        cands = [{"rank": i + 1, "sleeper_id": c.sleeper_id, "name": c.name, "positions": c.positions,
                  "team": e._rows[c.row].get("team_now"), "adp": c.adp, "delta_pw": rec.per_week(c.delta),
                  "e_next_pw": rec.per_week(c.e_next), "score_gap_pw": rec.per_week(best - c.score),
                  "survival": c.survival, "reach": c.reach, "fp_pg": e._rows[c.row].get("fp_pg"),
                  "lockin_3": e._rows[c.row].get("lockin_3"), "vor_per_week": e._rows[c.row].get("vor_per_week"),
                  "rookie": e._rows[c.row].get("rookie")}
                 for i, c in enumerate(rec.candidates[:top])]
        info = card(rec, e.board.players, e.ctx.scoring, rollout=ro)
        rollout = None
        if ro is not None:
            top_mean = ro.candidates[0].mean if ro.candidates else 0.0
            rollout = {"worlds": ro.worlds, "elapsed": ro.elapsed, "confidence": ro.confidence,
                       "edge_pw": ro.per_week(ro.edge) if ro.edge is not None else None,
                       "edge_se_pw": ro.per_week(ro.edge_se) if ro.edge_se is not None else None,
                       "candidates": [{"rank": i + 1, "sleeper_id": c.sleeper_id, "name": c.name,
                                       "positions": c.positions, "value_pw": ro.per_week(c.mean),
                                       "gap_pw": ro.per_week(top_mean - c.mean), "se_pw": ro.per_week(c.se),
                                       "p_best": c.p_best, "reach": c.reach, "dropped": c.dropped,
                                       "survival_sim": ro.survival.get(c.sleeper_id), "greedy_rank": c.greedy_rank}
                                      for i, c in enumerate(ro.candidates)]}
        return clean({"version": st.version, "pick": rec.pick, "next_pick": rec.next_pick,
                      "planning_ahead": rec.pick != st.current_pick, "candidates": cands,
                      "card": {**info, "drivers": [{"group": g, "value": v} for g, v in info.get("drivers", [])]}
                      if info else None, "rollout": rollout})

    def what_if(self, s: Session, sleeper_id: str) -> dict:
        """One player's score against the top pick, for any available player."""
        rec = self.recommendation(s)
        by = {c["sleeper_id"]: c for c in rec["candidates"]}
        if sleeper_id in by:
            return by[sleeper_id]
        e = self.engine
        st = s.state
        if sleeper_id not in e.row_of or sleeper_id in st.drafted:
            raise ValueError("not an available player")
        full = recommend(e.board, st, e.sim, e.evaluator)
        for c in full.candidates:
            if c.sleeper_id == sleeper_id:
                return clean({"sleeper_id": c.sleeper_id, "name": c.name, "delta_pw": full.per_week(c.delta),
                              "score_gap_pw": full.per_week(full.candidates[0].score - c.score),
                              "survival": c.survival, "in_pool": True})
        return {"sleeper_id": sleeper_id, "name": e.row(sleeper_id)["name"], "in_pool": False}

    def odds(self, s: Session, top: int = 8) -> list[dict]:
        if s.kind != "mock" or s.state.complete or s.state.my_turn:
            return []
        return clean(s.mock.odds(top).to_dicts())

    def projection(self, s: Session) -> list[dict]:
        """Every team's projected points per week and simulated playoff and title odds."""
        from draftai.draft.mock import project_league

        e = self.engine
        n = max(50, e.sim.rollout_worlds // 2)
        df = project_league(e.board, s.state, e.evaluator, n_worlds=n, waiver_slots=e.sim.waiver_slots,
                            playoff_teams=s.ctx.playoff_teams, round_type=s.ctx.playoff_round_type,
                            seed=s.mock.seed if s.mock else 0, team_name=lambda sl: self.team_name(s, sl),
                            playoff_week_start=s.ctx.playoff_week_start)
        return clean(df.to_dicts())


def minutes_overrides(path) -> list[dict]:
    if not path.exists():
        return []
    df = pl.read_csv(path, schema_overrides={"sleeper_id": pl.Utf8, "mpg": pl.Float64, "note": pl.Utf8})
    return clean(df.to_dicts())
