"""Sleeper API client (https://docs.sleeper.com): read-only, no auth.

League settings, positional eligibility, injury status and live draft picks come from
here. Responses are parsed into Pydantic models so a renamed field fails loudly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from draftai.config import Paths
from draftai.ingestion.cache import HttpSource, RawCache

DAY = 24 * 3600.0


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class SleeperLeague(_Loose):
    league_id: str
    name: str = ""
    season: str
    sport: str = "nba"
    status: str | None = None
    total_rosters: int
    roster_positions: list[str]
    scoring_settings: dict[str, float] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    draft_id: str | None = None


class SleeperDraft(_Loose):
    draft_id: str
    league_id: str | None = None
    season: str | None = None
    status: str | None = None
    type: str = "snake"
    settings: dict[str, Any] = Field(default_factory=dict)
    draft_order: dict[str, int] | None = None
    slot_to_roster_id: dict[str, int] | None = None
    start_time: int | None = None
    last_picked: int | None = None
    metadata: dict[str, Any] | None = None


class SleeperPick(_Loose):
    pick_no: int
    round: int
    draft_slot: int
    player_id: str
    picked_by: str | None = None
    roster_id: int | None = None
    is_keeper: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradedPick(_Loose):
    season: str | None = None
    round: int
    roster_id: int  # original owner
    owner_id: int  # current owner
    previous_owner_id: int | None = None


class SleeperClient:
    def __init__(self, cache: RawCache | None = None):
        paths = Paths().ensure()
        self.http = HttpSource(
            name="sleeper",
            base_url="https://api.sleeper.app/v1",
            cache=cache or RawCache(paths.raw),
            # Sleeper asks for under 1,000 calls a minute.
            min_interval=0.08,
        )

    # --- players and state -------------------------------------------------------
    def players(self, ttl: float = DAY) -> dict[str, dict]:
        """The ~5 MB players file; fetch at most daily."""
        return self.http.get("/players/nba", ttl=ttl)

    def state(self) -> dict:
        return self.http.get("/state/nba", ttl=3600)

    # --- users and leagues -------------------------------------------------------
    def user(self, username_or_id: str) -> dict | None:
        return self.http.get(f"/user/{username_or_id}", ttl=DAY)

    def user_leagues(self, user_id: str, season: int) -> list[dict]:
        return self.http.get(f"/user/{user_id}/leagues/nba/{season}", ttl=3600) or []

    def user_drafts(self, user_id: str, season: int) -> list[dict]:
        return self.http.get(f"/user/{user_id}/drafts/nba/{season}", ttl=3600) or []

    def league(self, league_id: str, ttl: float = 3600) -> SleeperLeague:
        return SleeperLeague(**self.http.get(f"/league/{league_id}", ttl=ttl))

    def league_users(self, league_id: str) -> list[dict]:
        return self.http.get(f"/league/{league_id}/users", ttl=3600) or []

    def league_rosters(self, league_id: str) -> list[dict]:
        return self.http.get(f"/league/{league_id}/rosters", ttl=600) or []

    def league_drafts(self, league_id: str) -> list[SleeperDraft]:
        return [SleeperDraft(**d) for d in self.http.get(f"/league/{league_id}/drafts", ttl=600) or []]

    # --- drafts ------------------------------------------------------------------
    def draft(self, draft_id: str, ttl: float = 0) -> SleeperDraft:
        return SleeperDraft(**self.http.get(f"/draft/{draft_id}", ttl=ttl))

    def draft_picks(self, draft_id: str, ttl: float = 0) -> list[SleeperPick]:
        """Polled during a live draft, so uncached by default."""
        return [SleeperPick(**p) for p in self.http.get(f"/draft/{draft_id}/picks", ttl=ttl) or []]

    def traded_picks(self, draft_id: str, ttl: float = 60) -> list[TradedPick]:
        return [TradedPick(**p) for p in self.http.get(f"/draft/{draft_id}/traded_picks", ttl=ttl) or []]

    # --- undocumented stats feed (optional) --------------------------------------
    def season_stats(self, season: int, ttl: float | None = DAY) -> dict[str, dict]:
        """Season totals per Sleeper player id, including technical and flagrant fouls.

        Undocumented and may change; used only for foul rates and golden scoring checks.
        """
        return self.http.get(f"/stats/nba/regular/{season}", ttl=ttl) or {}

    def week_stats(self, season: int, week: int, ttl: float | None = None) -> dict[str, dict]:
        return self.http.get(f"/stats/nba/regular/{season}/{week}", ttl=ttl) or {}
