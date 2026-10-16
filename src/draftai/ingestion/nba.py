"""stats.nba.com via nba_api: game logs, bio stats, draft history and the schedule.

The endpoints are undocumented and drop requests from datacenter IPs, so run this from
a home connection. Raw responses are cached; past seasons are never refetched.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
from pydantic import BaseModel

from draftai.config import Paths, season_label, season_of
from draftai.ingestion.cache import RateLimiter, RawCache

DAY = 24 * 3600.0
_LIMITER = RateLimiter(0.7)


class ResultSet(BaseModel):
    name: str
    headers: list[str]
    rowSet: list[list[Any]]


class SchemaDriftError(RuntimeError):
    pass


def _result_frame(body: dict, name: str, required: set[str]) -> pl.DataFrame:
    sets = [ResultSet(**rs) for rs in body["resultSets"]]
    rs = next((s for s in sets if s.name == name), None)
    if rs is None:
        raise SchemaDriftError(f"result set {name!r} missing; got {[s.name for s in sets]}")
    missing = required - set(rs.headers)
    if missing:
        raise SchemaDriftError(f"{name}: expected columns missing: {sorted(missing)}")
    return pl.DataFrame(rs.rowSet, schema=rs.headers, orient="row", infer_schema_length=None)


class NbaClient:
    def __init__(self, cache: RawCache | None = None, today: date | None = None):
        self.cache = cache or RawCache(Paths().ensure().raw)
        self.today = today or date.today()

    def _ttl(self, season: int) -> float | None:
        return None if season < season_of(self.today) else DAY

    def _fetch(self, endpoint: str, params: dict, ttl: float | None) -> dict:
        def call() -> dict:
            from nba_api.stats import endpoints  # imported lazily: slow import

            _LIMITER.wait()
            cls = getattr(endpoints, endpoint)
            return cls(**params, timeout=60).get_dict()

        return self.cache.fetch("nba", endpoint, params, call, ttl)

    # --- game logs ---------------------------------------------------------------
    PLAYER_LOG_COLS = {
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
        "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PF", "PTS",
    }

    def player_game_logs(self, season: int) -> pl.DataFrame:
        body = self._fetch(
            "LeagueGameLog",
            {
                "season": season_label(season),
                "player_or_team_abbreviation": "P",
                "season_type_all_star": "Regular Season",
            },
            self._ttl(season),
        )
        return _result_frame(body, "LeagueGameLog", self.PLAYER_LOG_COLS)

    def team_game_logs(self, season: int) -> pl.DataFrame:
        body = self._fetch(
            "LeagueGameLog",
            {
                "season": season_label(season),
                "player_or_team_abbreviation": "T",
                "season_type_all_star": "Regular Season",
            },
            self._ttl(season),
        )
        return _result_frame(body, "LeagueGameLog", self.PLAYER_LOG_COLS - {"PLAYER_ID", "PLAYER_NAME"})

    # --- players -----------------------------------------------------------------
    def player_bio(self, season: int) -> pl.DataFrame:
        body = self._fetch(
            "LeagueDashPlayerBioStats",
            {"season": season_label(season), "season_type_all_star": "Regular Season"},
            self._ttl(season),
        )
        return _result_frame(
            body,
            "LeagueDashPlayerBioStats",
            {"PLAYER_ID", "PLAYER_NAME", "AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "DRAFT_YEAR",
             "DRAFT_NUMBER"},
        )

    def draft_history(self, year: int) -> pl.DataFrame:
        body = self._fetch(
            "DraftHistory", {"season_year_nullable": str(year)}, self._ttl(year - 1)
        )
        return _result_frame(body, "DraftHistory", {"PERSON_ID", "PLAYER_NAME", "OVERALL_PICK", "SEASON"})

    # --- schedule ----------------------------------------------------------------
    def schedule(self, season: int) -> pl.DataFrame:
        """Full league schedule (all game types) as one row per game."""
        body = self._fetch(
            "ScheduleLeagueV2", {"season": season_label(season), "league_id": "00"}, DAY
        )
        rows = []
        for gd in body["leagueSchedule"]["gameDates"]:
            for g in gd["games"]:
                rows.append(
                    {
                        "game_id": g["gameId"],
                        "game_date_est": g["gameDateEst"][:10],
                        "game_datetime_est": g.get("gameDateTimeEst"),
                        "home": g["homeTeam"]["teamTricode"],
                        "away": g["awayTeam"]["teamTricode"],
                        "game_label": g.get("gameLabel") or "",
                        "game_subtype": g.get("gameSubtype") or "",
                    }
                )
        if not rows:
            raise SchemaDriftError("schedule contains no games")
        return pl.DataFrame(rows)
