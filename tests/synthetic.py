"""A small synthetic league with the same table shapes as the real normalised data.

Players have latent per-minute rates, minutes and durability; they age, change teams,
retire, and are replaced by drafted rookies. Used by the integration and feature tests
so CI never touches the network.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from draftai.config import TrainConfig
from draftai.scoring import ScoringSettings

STATS = ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")
S = ScoringSettings()
CFG = TrainConfig(
    first_target_season=2018, validation_seasons=[2020, 2021], test_season=2022, target_season=2023,
    lgbm={"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "max_depth": 3, "min_child_samples": 10,
          "verbose": -1},
)
ROSTER = ["PG", "SG", "G", "SF", "PF", "F", "C", "C", "UTIL", "UTIL", "BN", "BN", "BN", "IL"]


def make_league(seasons=range(2016, 2023), teams: int = 10, per_team: int = 12, games: int = 40,
                seed: int = 0) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    team_names = [f"T{i:02d}" for i in range(teams)]
    next_id = 1000
    players: dict[int, dict] = {}

    def new_player(team: str, season: int, rookie: bool) -> int:
        nonlocal next_id
        pid = next_id
        next_id += 1
        big = rng.random() < 0.3
        players[pid] = {
            "team": team,
            "age": float(rng.integers(19, 22) if rookie else rng.integers(21, 33)),
            "height": float(rng.normal(83 if big else 77, 1.5)),
            "mpg": float(np.clip(rng.normal(24, 7), 8, 38)),
            "rates": {
                "pts": rng.uniform(0.3, 0.8), "reb": rng.uniform(0.25, 0.45) if big else rng.uniform(0.1, 0.25),
                "ast": rng.uniform(0.03, 0.1) if big else rng.uniform(0.08, 0.3), "stl": rng.uniform(0.01, 0.05),
                "blk": rng.uniform(0.03, 0.08) if big else rng.uniform(0.0, 0.02), "tov": rng.uniform(0.04, 0.1),
                "fg3m": rng.uniform(0.0, 0.04) if big else rng.uniform(0.03, 0.12),
            },
            "durability": float(rng.uniform(0.75, 0.98)),
            "draft_year": season if rookie else None,
            "pick": int(rng.integers(1, 61)) if rookie else None,
            "name": f"Player {pid}",
        }
        return pid

    for t in team_names:
        for _ in range(per_team):
            new_player(t, min(seasons), rookie=False)

    pg_rows, tg_rows, bio_rows, draft_rows = [], [], [], []
    for season in seasons:
        opener = date(season, 10, 20)
        # Offseason: retirements, moves, rookies.
        if season > min(seasons):
            for pid in list(players):
                p = players[pid]
                if p["team"] is None:
                    continue
                if p["age"] > 34 or rng.random() < 0.05:
                    p["team"] = None
                    continue
                p["age"] += 1
                p["mpg"] = float(np.clip(p["mpg"] + rng.normal(1.5 if p["age"] < 26 else -1.0, 3), 6, 38))
                if rng.random() < 0.15:
                    p["team"] = team_names[rng.integers(teams)]
            for t in team_names:
                while sum(1 for p in players.values() if p["team"] == t) < per_team:
                    pid = new_player(t, season, rookie=True)
                    draft_rows.append({"draft_year": season, "player_id": pid, "player_name": players[pid]["name"],
                                       "overall_pick": players[pid]["pick"], "team": t})
        for pid, p in players.items():
            if p["team"] is not None:
                bio_rows.append({"season": season, "player_id": pid, "player_name": p["name"], "age": p["age"],
                                 "height_in": p["height"], "weight_lb": 220.0, "draft_year": p["draft_year"],
                                 "draft_number": p["pick"]})
        # Schedule: each day, teams paired at random; every team plays `games` games.
        injured: dict[int, int] = {}
        day = 0
        for g in range(games):
            day += int(rng.choice([1, 1, 2, 2, 3]))
            d = opener + timedelta(days=day)
            order = rng.permutation(team_names)
            for k in range(0, teams, 2):
                home, away = order[k], order[k + 1]
                gid = f"002{season % 100:02d}{g:03d}{k:02d}"[:10]
                for team, opp, is_home in ((home, away, True), (away, home, False)):
                    roster = [pid for pid, p in players.items() if p["team"] == team]
                    tot = {s: 0 for s in STATS} | {"fga": 0, "fta": 0, "oreb": 0}
                    for pid in roster:
                        p = players[pid]
                        if injured.get(pid, 0) > 0:
                            injured[pid] -= 1
                            continue
                        if rng.random() > p["durability"]:
                            if rng.random() < 0.25:
                                injured[pid] = int(rng.integers(3, 12))
                            continue
                        m = float(np.clip(rng.normal(p["mpg"], 4), 1, 46))
                        box = {s: int(rng.poisson(p["rates"][s] * m)) for s in STATS}
                        box["pts"] = max(box["pts"], 3 * box["fg3m"])
                        box["fga"] = int(box["pts"] * 0.8) + 1
                        box["fta"] = int(box["pts"] * 0.25)
                        box["oreb"] = box["reb"] // 4
                        for s in tot:
                            tot[s] += box[s]
                        pg_rows.append({"season": season, "game_id": gid, "game_date": d, "player_id": pid,
                                        "player_name": p["name"], "team_id": int(team[1:]) + 1, "team": team, "opp": opp,
                                        "home": is_home, "min": round(m), **box, "dreb": box["reb"] - box["oreb"],
                                        "fgm": box["fga"] // 2, "ftm": int(box["fta"] * 0.75), "fg3a": box["fg3m"] * 3, "pf": 2})
                    tg_rows.append({"season": season, "game_id": gid, "game_date": d, "team_id": int(team[1:]) + 1,
                                    "team": team, "opp": opp, "home": is_home, "team_min": 240.0,
                                    **{f"team_{s}": tot[s] for s in tot}})
    pg = pl.DataFrame(pg_rows).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("min").cast(pl.Float64),
        *[pl.col(c).cast(pl.Int32) for c in (*STATS, "fga", "fta", "oreb", "dreb", "fgm", "ftm", "fg3a", "pf")],
    )
    tg = pl.DataFrame(tg_rows).with_columns(pl.col("season").cast(pl.Int32),
                                            *[pl.col(c).cast(pl.Int32) for c in tg_rows[0] if c.startswith("team_") and c not in ("team_id", "team_min")])
    bio = pl.DataFrame(bio_rows, schema_overrides={"draft_year": pl.Float64, "draft_number": pl.Float64}).with_columns(
        pl.col("season").cast(pl.Int32))
    draft = pl.DataFrame(draft_rows).with_columns(pl.col("draft_year").cast(pl.Int32), pl.col("overall_pick").cast(pl.Int32))
    positions = {pid: (["C"] if p["height"] > 81 else ["PF", "SF"] if p["height"] > 78 else ["PG", "SG"])
                 for pid, p in players.items()}
    current = {pid: p["team"] for pid, p in players.items() if p["team"] is not None}
    return {"player_games": pg, "team_games": tg, "bio": bio, "draft": draft, "positions": positions, "current": current}


def target_schedule(teams: int = 10, season: int = 2023, games: int = 40, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"T{i:02d}" for i in range(teams)]
    rows, day = [], 0
    opener = date(season, 10, 20)
    for g in range(games):
        day += int(rng.choice([1, 1, 2, 2, 3]))
        d = opener + timedelta(days=day)
        order = rng.permutation(names)
        for k in range(0, teams, 2):
            gid = f"002{season % 100:02d}{g:03d}{k:02d}"[:10]
            rows.append({"game_id": gid, "game_date": d, "team": order[k], "opp": order[k + 1], "home": True})
            rows.append({"game_id": gid, "game_date": d, "team": order[k + 1], "opp": order[k], "home": False})
    return pl.DataFrame(rows).with_columns(pl.lit(season, dtype=pl.Int32).alias("season"))
