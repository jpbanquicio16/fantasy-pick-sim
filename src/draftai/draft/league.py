"""League context: settings read from the Sleeper API at runtime, never hard-coded.

The YAML config is only a fallback for offline use (mock drafts, tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from draftai.config import LeagueConfig
from draftai.draft.state import DraftState
from draftai.ingestion.sleeper import SleeperClient, SleeperDraft, SleeperLeague
from draftai.scoring import ScoringSettings


@dataclass
class LeagueContext:
    season: int
    teams: int
    rounds: int
    roster_positions: list[str]
    scoring: ScoringSettings
    draft_type: str = "snake"
    reversal_round: int = 0
    playoff_week_start: int = 22
    playoff_teams: int = 6
    playoff_round_type: int = 0
    playoff_weight: float = 1.5
    include_cup_final: bool = False
    my_slot: int | None = None
    league_id: str | None = None
    league_name: str = "offline"
    draft_id: str | None = None
    slot_to_roster: dict[int, int] = field(default_factory=dict)
    traded: list[tuple[int, int, int]] = field(default_factory=list)  # (round, original slot, new slot)
    users: dict[str, str] = field(default_factory=dict)  # user_id -> display name
    slot_names: dict[int, str] = field(default_factory=dict)

    def new_state(self) -> DraftState:
        if self.my_slot is None:
            raise ValueError("draft slot unknown: set draft_slot in configs/league.yaml or a username")
        return DraftState.new(self.teams, self.rounds, self.my_slot, self.draft_type, self.reversal_round, self.traded)

    @classmethod
    def offline(cls, cfg: LeagueConfig) -> "LeagueContext":
        return cls(
            season=cfg.season, teams=cfg.teams, rounds=cfg.rounds, roster_positions=list(cfg.roster_positions),
            scoring=ScoringSettings.from_sleeper(cfg.scoring_settings, cfg.bonus_stacking),
            draft_type=cfg.draft_type, reversal_round=cfg.reversal_round, playoff_week_start=cfg.playoff_week_start,
            playoff_teams=cfg.playoff_teams, playoff_weight=cfg.playoff_weight, include_cup_final=cfg.include_cup_final,
            my_slot=cfg.draft_slot,
        )

    @classmethod
    def from_sleeper(cls, cfg: LeagueConfig, client: SleeperClient | None = None, league_id: str | None = None,
                     draft_id: str | None = None, username: str | None = None) -> "LeagueContext":
        client = client or SleeperClient()
        league_id = league_id or cfg.league_id
        draft: SleeperDraft | None = None
        league: SleeperLeague | None = None
        if league_id:
            league = client.league(league_id, ttl=600)
            draft_id = draft_id or cfg.draft_id or league.draft_id
        draft_id = draft_id or cfg.draft_id
        if draft_id:
            draft = client.draft(draft_id)
        if league is None and draft is None:
            return cls.offline(cfg)

        ls = league.settings if league else {}
        ds = draft.settings if draft else {}
        teams = int(ds.get("teams") or (league.total_rosters if league else cfg.teams))
        roster_positions = league.roster_positions if league else _positions_from_draft(ds, cfg)
        scoring = ScoringSettings.from_sleeper(league.scoring_settings if league else cfg.scoring_settings,
                                               cfg.bonus_stacking)
        ctx = cls(
            season=int(league.season) if league else cfg.season,
            teams=teams,
            rounds=int(ds.get("rounds") or len(roster_positions)),
            roster_positions=list(roster_positions),
            scoring=scoring,
            draft_type=(draft.type if draft else cfg.draft_type) or "snake",
            reversal_round=int(ds.get("reversal_round") or 0),
            playoff_week_start=int(ls.get("playoff_week_start") or cfg.playoff_week_start),
            playoff_teams=int(ls.get("playoff_teams") or cfg.playoff_teams),
            playoff_round_type=int(ls.get("playoff_round_type") or 0),
            playoff_weight=cfg.playoff_weight,
            include_cup_final=cfg.include_cup_final,
            league_id=league_id,
            league_name=league.name if league else (draft.metadata or {}).get("name", "mock draft"),
            draft_id=draft_id,
        )
        if draft is not None:
            ctx.slot_to_roster = {int(k): int(v) for k, v in (draft.slot_to_roster_id or {}).items()}
            roster_to_slot = {v: k for k, v in ctx.slot_to_roster.items()}
            if roster_to_slot:
                for tp in client.traded_picks(draft.draft_id):
                    orig, new = roster_to_slot.get(tp.roster_id), roster_to_slot.get(tp.owner_id)
                    if orig and new:
                        ctx.traded.append((tp.round, orig, new))
            order = draft.draft_order or {}
            if league_id:
                users = client.league_users(league_id)
                ctx.users = {u["user_id"]: u.get("display_name") or u["user_id"] for u in users}
            ctx.slot_names = {slot: ctx.users.get(uid, f"slot {slot}") for uid, slot in order.items()}
            username = username or cfg.username
            if username:
                u = client.user(username)
                if u and u.get("user_id") in order:
                    ctx.my_slot = int(order[u["user_id"]])
        if ctx.my_slot is None:
            ctx.my_slot = cfg.draft_slot
        return ctx

    def describe(self) -> list[str]:
        out = [
            f"League: {self.league_name} ({self.league_id or 'offline'}), season {self.season}",
            f"Teams {self.teams}, rounds {self.rounds}, draft {self.draft_type}"
            + (f" with reversal from round {self.reversal_round}" if self.reversal_round else ""),
            f"Lineup: {', '.join(self.roster_positions)}",
            f"Playoffs: {self.playoff_teams} teams from week {self.playoff_week_start} (round type {self.playoff_round_type}),"
            f" weight {self.playoff_weight}",
            f"Your slot: {self.my_slot}",
            f"Bonus stacking (40+/50+): {self.scoring.bonus_stacking}",
            "Scoring: " + ", ".join(f"{k}={v:g}" for k, v, _ in self.scoring.describe()),
        ]
        if self.scoring.unknown_keys:
            out.append(f"Ignored scoring keys: {self.scoring.unknown_keys}")
        if self.traded:
            out.append(f"Traded picks: {self.traded}")
        return out


def _positions_from_draft(ds: dict, cfg: LeagueConfig) -> list[str]:
    """Mock drafts carry slot counts (slots_pg, slots_util, slots_bn, ...) instead of a league."""
    order = ["pg", "sg", "g", "sf", "pf", "f", "c", "util", "bn"]
    out = []
    for key in order:
        out += [key.upper()] * int(ds.get(f"slots_{key}", 0) or 0)
    return out or list(cfg.roster_positions)
