"""NBA person id <-> Sleeper player id.

Sleeper exposes no NBA id, so players are matched on a normalised name, using team
and age to break ties, with a manual overrides file for the rest.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import polars as pl

from draftai.config import Paths

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
# Sleeper sometimes uses a short or long first name where the NBA uses the other.
_NICKNAMES = {
    "nicolas": "nic", "herbert": "herb", "cameron": "cam", "alexandre": "alex", "nahshon": "bones",
    "moritz": "moe", "gregory": "greg", "patrick": "pat", "sviatoslav": "svi", "vincent": "vince",
    "joshua": "josh", "jeffrey": "jeff", "william": "will",
}

# Sleeper team abbreviations that differ from NBA tricodes.
SLEEPER_TEAM_FIX = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "PHO": "PHX", "UTAH": "UTA",
                    "WSH": "WAS", "BKN": "BKN"}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_variants(name: str) -> set[str]:
    n = norm_name(name)
    parts = n.split(" ")
    out = {n.replace(" ", "")}
    if len(parts) >= 2:
        first, rest = parts[0], "".join(parts[1:])
        if first in _NICKNAMES:
            out.add(_NICKNAMES[first] + rest)
        inv = {v: k for k, v in _NICKNAMES.items()}
        if first in inv:
            out.add(inv[first] + rest)
    return out


def _number(v) -> float | None:
    """Sleeper sends height and weight as strings ("80", "205"); blanks become null."""
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def sleeper_players_table(players: dict[str, dict], snapshot: date | None = None) -> pl.DataFrame:
    rows = []
    for pid, p in players.items():
        if p.get("sport", "nba") != "nba" or not p.get("full_name"):
            continue
        team = p.get("team")
        rows.append(
            {
                "sleeper_id": str(pid),
                "full_name": p["full_name"],
                "team": SLEEPER_TEAM_FIX.get(team, team) if team else None,
                "positions": [x for x in (p.get("fantasy_positions") or []) if x],
                "position": p.get("position"),
                "search_rank": p.get("search_rank"),
                "injury_status": p.get("injury_status"),
                "injury_body_part": p.get("injury_body_part"),
                "birth_date": p.get("birth_date"),
                "years_exp": p.get("years_exp"),
                "active": bool(p.get("active")),
                "status": p.get("status"),
                # Rookie-model inputs for players with no NBA bio yet.
                "height_in": _number(p.get("height")),
                "weight_lb": _number(p.get("weight")),
                "college": p.get("college"),
            }
        )
    df = pl.DataFrame(
        rows,
        schema_overrides={"positions": pl.List(pl.Utf8), "search_rank": pl.Int64, "years_exp": pl.Int64,
                          "height_in": pl.Float64, "weight_lb": pl.Float64, "college": pl.Utf8},
    )
    return df.with_columns(
        pl.col("birth_date").str.to_date(strict=False),
        pl.lit(snapshot or date.today()).alias("snapshot_date"),
    )


def _load_overrides(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    df = pl.read_csv(path, schema_overrides={"nba_id": pl.Int64, "sleeper_id": pl.Utf8})
    return dict(zip(df["nba_id"].to_list(), df["sleeper_id"].to_list()))


def build_crosswalk(
    nba_players: pl.DataFrame,
    sleeper: pl.DataFrame,
    overrides_path: Path | None = None,
) -> pl.DataFrame:
    """Match NBA players to Sleeper players.

    ``nba_players`` needs columns player_id, player_name, team (latest), age (latest season,
    may be null) and last_season. Returns nba_id, sleeper_id, method.
    """
    overrides = _load_overrides(overrides_path or Paths().configs / "id_overrides.csv")
    by_name: dict[str, list[dict]] = {}
    for r in sleeper.iter_rows(named=True):
        for v in name_variants(r["full_name"]):
            by_name.setdefault(v, []).append(r)

    out, used = [], set(overrides.values())
    for nba_id, sid in overrides.items():
        out.append({"nba_id": nba_id, "sleeper_id": sid, "method": "override"})
    done = set(overrides)

    rows = sorted(nba_players.iter_rows(named=True), key=lambda r: -(r.get("last_season") or 0))
    for r in rows:
        if r["player_id"] in done:
            continue
        cands: dict[str, dict] = {}
        for v in name_variants(r["player_name"]):
            for c in by_name.get(v, []):
                if c["sleeper_id"] not in used:
                    cands[c["sleeper_id"]] = c
        if not cands:
            continue
        pick, method = None, "name"
        if len(cands) > 1:
            same_team = [c for c in cands.values() if c["team"] and c["team"] == r.get("team")]
            if len(same_team) == 1:
                pick, method = same_team[0], "name+team"
            elif r.get("age") is not None:
                aged = [
                    c for c in cands.values()
                    if c["birth_date"] is not None
                    and abs((date(int(r["last_season"]) + 1, 1, 1) - c["birth_date"]).days / 365.25 - r["age"]) < 1.6
                ]
                if len(aged) == 1:
                    pick, method = aged[0], "name+age"
            if pick is None:
                active = [c for c in cands.values() if c["active"]]
                if len(active) == 1:
                    pick, method = active[0], "name+active"
        else:
            pick = next(iter(cands.values()))
        if pick is not None:
            out.append({"nba_id": r["player_id"], "sleeper_id": pick["sleeper_id"], "method": method})
            used.add(pick["sleeper_id"])
            done.add(r["player_id"])
    return pl.DataFrame(out, schema={"nba_id": pl.Int64, "sleeper_id": pl.Utf8, "method": pl.Utf8})


def nba_player_directory(player_games: pl.DataFrame, bio: pl.DataFrame, draft: pl.DataFrame) -> pl.DataFrame:
    """Latest name, team, season and age per NBA player, including drafted rookies."""
    latest = (
        player_games.sort("game_date")
        .group_by("player_id")
        .agg(
            pl.col("player_name").last(),
            pl.col("team").last(),
            pl.col("season").max().alias("last_season"),
        )
    )
    ages = bio.sort("season").group_by("player_id").agg(pl.col("age").last())
    latest = latest.join(ages, on="player_id", how="left")
    rookies = (
        draft.join(latest.select("player_id"), on="player_id", how="anti")
        .select(
            "player_id", "player_name", "team",
            (pl.col("draft_year") - 1).cast(pl.Int32).alias("last_season"),
            pl.lit(None, dtype=pl.Float64).alias("age"),
        )
    )
    return pl.concat([latest.with_columns(pl.col("last_season").cast(pl.Int32)), rookies], how="diagonal_relaxed")
