"""Counterfactual replay of real Sleeper drafts (design section 25).

Keep every human pick and give one seat to a strategy. When the strategy takes a player a
human picked later, that human's pick is replaced by the opponent model's choice from what
is left (the only modelled picks). Then play the real season for every team under the
same rules. The comparison is with the person who actually held the seat: did the
strategy draft a better team than they did, against the same room?

The same drafts fit the opponent model (``fit_opponents``): every human pick except your
own, as a conditional logit on the season's market rank and positional need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from draftai.backtest.history import LeagueRules, SeasonData, season_data
from draftai.backtest.league import play
from draftai.backtest.run import Room, rosters_of
from draftai.backtest.strategies import Strategy, build
from draftai.config import BacktestConfig, Paths, SimConfig, TrainConfig
from draftai.draft.opponents import DraftRecord, Market, OpponentModel, choose, fit
from draftai.draft.rollout import RolloutConfig
from draftai.draft.state import DraftState
from draftai.scoring import ScoringSettings


@dataclass
class RealDraft:
    draft_id: str
    season: int
    label: str
    rules: LeagueRules
    picks: list[tuple[int, int, str]]  # (pick_no, owning slot, sleeper_id)
    user_slot: int | None
    slot_names: dict[int, str] = field(default_factory=dict)


def load_real_drafts(client, username: str, seasons: list[int], log=print) -> list[RealDraft]:
    """Completed drafts of yours in the given seasons (leagues and mock drafts alike)."""
    user = client.user(username)
    if not user:
        raise ValueError(f"no Sleeper user {username!r}")
    uid = user["user_id"]
    out = []
    for season in seasons:
        for d in client.user_drafts(uid, season):
            if d.get("status") != "complete" or d.get("type") not in ("snake", "linear"):
                continue
            ds = d.get("settings") or {}
            teams, rounds = int(ds.get("teams") or 0), int(ds.get("rounds") or 0)
            if teams < 4 or rounds < 4:
                continue
            draft = client.draft(d["draft_id"], ttl=None)
            league = client.league(d["league_id"], ttl=None) if d.get("league_id") else None
            positions = list(league.roster_positions) if league else _positions(ds)
            ls = league.settings if league else {}
            rules = LeagueRules(
                teams=teams, rounds=rounds, roster_positions=positions,
                scoring=ScoringSettings.from_sleeper(league.scoring_settings if league else None),
                draft_type=d.get("type", "snake"), reversal_round=int(ds.get("reversal_round") or 0),
                playoff_week_start=int(ls.get("playoff_week_start") or 22), playoff_teams=int(ls.get("playoff_teams") or 6),
                playoff_round_type=int(ls.get("playoff_round_type") or 0),
            )
            roster_to_slot = {int(r): int(s) for s, r in (draft.slot_to_roster_id or {}).items()}
            picks = []
            for p in client.draft_picks(d["draft_id"], ttl=None):
                slot = roster_to_slot.get(p.roster_id, p.draft_slot) if p.roster_id is not None else p.draft_slot
                picks.append((p.pick_no, int(slot), str(p.player_id)))
            order = draft.draft_order or {}
            names = {}
            if league:
                users = {u["user_id"]: u.get("display_name") or u["user_id"] for u in client.league_users(league.league_id)}
                names = {int(s): users.get(u, f"slot {s}") for u, s in order.items()}
            label = (league.name.strip() if league else (draft.metadata or {}).get("name", "mock")) + f" {season}"
            out.append(RealDraft(d["draft_id"], season, label, rules, sorted(picks), order.get(uid), names))
            log(f"real draft: {label} ({teams} teams, {len(picks)} picks), your slot {order.get(uid)}")
    return out


def _positions(ds: dict) -> list[str]:
    out = []
    for key in ("pg", "sg", "g", "sf", "pf", "f", "c", "util", "bn"):
        out += [key.upper()] * int(ds.get(f"slots_{key}", 0) or 0)
    return out


def _owners(rd: RealDraft) -> list[int]:
    return [slot for _, slot, _ in sorted(rd.picks)]


def record(rd: RealDraft, sd: SeasonData) -> DraftRecord:
    row_of = {s: i for i, s in enumerate(sd.board.players["sleeper_id"].to_list())}
    return DraftRecord([(no, slot, row_of.get(sid)) for no, slot, sid in rd.picks],
                       {rd.user_slot} if rd.user_slot else set(), rd.label)


def fit_opponents(drafts: list[RealDraft], sim: SimConfig, train_cfg: TrainConfig, paths: Paths,
                  game_samples: int = 2000, log=print) -> OpponentModel:
    """Fit on every past-season draft against that season's (naive) market."""
    from draftai.draft.needs import NeedModel

    records, markets = [], []
    for rd in drafts:
        sd = season_data(rd.season, rd.rules, sim, train_cfg, paths, game_samples, log=log)
        needs = NeedModel(sd.board.positions(), sd.board.slots)
        markets.append(Market(sd.board.players["adp"].to_numpy(), needs))
        records.append(record(rd, sd))
    return fit(records, markets, sim.opponent_window)


def replay(rd: RealDraft, sim: SimConfig, train_cfg: TrainConfig, bt: BacktestConfig, paths: Paths,
           model: OpponentModel, strategies: list[str], log=print) -> pl.DataFrame:
    sd = season_data(rd.season, rd.rules, sim, train_cfg, paths, bt.game_samples, log=log)
    room = Room.build(sd, rd.rules, model)
    rcfg = RolloutConfig(worlds=bt.rollout_worlds, shortlist=bt.rollout_shortlist, race_after=bt.rollout_race_after,
                         waiver_slots=sim.waiver_slots, seed=bt.seed + rd.season)
    strats: list[Strategy] = build(strategies)
    for s in strats:
        s.prepare(sd, room.needs, room.evaluator, model, sim, rcfg)
    owners = _owners(rd)
    row_of = {s: i for i, s in enumerate(room.ids)}
    actual = {no: row_of.get(sid) for no, _, sid in rd.picks}
    teams = rd.rules.teams
    weeks_weight = float(sd.board.weights.sum())
    rng_seed = [bt.seed, rd.season, 77]

    def outcome(rosters, seat_rows: dict[str, list[int]]) -> dict:
        res = play(rosters, sd.realised, room.elig, rd.rules, np.random.default_rng(rng_seed), bt.replay_schedules,
                   bt.waiver_claims)
        reg = res.regular_per_week(sd.realised.playoff)
        return {"res": res, "reg": reg}

    human_rosters = [[actual[no] for no, slot, _ in rd.picks if slot == s and actual[no] is not None]
                     for s in range(1, teams + 1)]
    human = outcome(human_rosters, {})
    missing = sum(1 for v in actual.values() if v is None)
    rows = []
    for seat in range(1, teams + 1):
        i = seat - 1
        rows.append(_row(rd, seat, "human", human, i, room, human_rosters[i], weeks_weight, sd))
        for s in strats:
            s.reset()
            state = DraftState(teams, rd.rules.rounds, list(owners), seat)
            avail = np.ones(len(room.ids), dtype=bool)
            cols = np.ones(room.market.size, dtype=bool)
            counts = np.zeros((teams + 1, room.needs.n_types), dtype=np.int64)
            u = np.random.default_rng([bt.seed, rd.season, seat]).random(len(owners) + 2)
            while not state.complete:
                t, owner = state.current_pick, state.on_the_clock
                if owner == seat:
                    row = s.choose(state, avail, counts[owner])
                else:
                    row = actual.get(t)
                    if row is None or not avail[row]:
                        col = choose(model, room.market, cols, counts[owner], t, float(u[t]))
                        row = int(room.market.order[col])
                state.add_pick(room.ids[row])
                avail[row] = False
                cols[room.market.col_of[row]] = False
                counts[owner, room.needs.type_of[row]] += 1
            rosters = rosters_of(room, state)
            rows.append(_row(rd, seat, s.name, outcome(rosters, {}), i, room, rosters[i], weeks_weight, sd))
        log(f"replay {rd.label}: seat {seat} done")
    df = pl.DataFrame(rows)
    return df.with_columns(pl.lit(missing).alias("unmatched_picks"))


def _row(rd: RealDraft, seat: int, strategy: str, out: dict, i: int, room: Room, roster: list[int],
         weeks_weight: float, sd: SeasonData) -> dict:
    o = out["res"].outcome
    return {
        "draft": rd.label, "season": rd.season, "seat": seat, "manager": rd.slot_names.get(seat, f"slot {seat}"),
        "yours": seat == rd.user_slot, "strategy": strategy,
        "p_playoffs": float(o.p_playoffs[i]), "p_title": float(o.p_title[i]), "win_rate": float(o.win_rate[i]),
        "realised_per_week": float(out["reg"][i]), "proj_per_week": room.evaluator.value(roster) / weeks_weight,
        "first_picks": ", ".join(sd.board.players["name"][r] for r in roster[:3]),
    }


def summarise(df: pl.DataFrame) -> pl.DataFrame:
    human = df.filter(pl.col("strategy") == "human").select(
        "draft", "seat", pl.col("p_playoffs").alias("h_po"), pl.col("realised_per_week").alias("h_pts"),
        pl.col("p_title").alias("h_title"))
    j = df.filter(pl.col("strategy") != "human").join(human, on=["draft", "seat"])
    return j.group_by("strategy").agg(
        pl.len().alias("seats"),
        (pl.col("realised_per_week") - pl.col("h_pts")).mean().round(2).alias("pts_per_week_vs_human"),
        (pl.col("p_playoffs") - pl.col("h_po")).mean().round(3).alias("p_playoffs_vs_human"),
        (pl.col("p_title") - pl.col("h_title")).mean().round(3).alias("p_title_vs_human"),
        ((pl.col("realised_per_week") > pl.col("h_pts")).mean()).round(3).alias("beat_human_share"),
    ).sort("pts_per_week_vs_human", descending=True)


def replay_section(df: pl.DataFrame, meta: dict) -> list[str]:
    from draftai.backtest.run import _md

    yours = df.filter(pl.col("yours")).select("draft", "strategy", "p_playoffs", "p_title", "realised_per_week",
                                              "first_picks")
    lines = [
        "## Replay of real Sleeper drafts",
        "Every human pick kept; one seat at a time given to a strategy (a human pick the strategy already",
        "took is replaced by the opponent model's choice). Every team then plays the real season under the",
        "same lineup and waiver rules. Compared with the person who actually held each seat.",
        "Titles rest on three real playoff weeks, so in a replay they are close to all-or-nothing. The",
        "league keeps a player and has IL slots: humans' stash picks score zero in a one-season replay,",
        "so the gaps below are an upper bound.",
        "",
        f"Drafts: {', '.join(meta.get('drafts', []))}. Unmatched picks (not on the board): {meta.get('unmatched', 0)}.",
        "",
        _md(summarise(df)),
        "",
        "Your seat:",
        _md(yours),
    ]
    return lines
