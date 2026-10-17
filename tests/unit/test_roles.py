"""Team moves, reduced roles and the team minutes budget."""

import numpy as np
import polars as pl
import pytest
from synthetic import S

from draftai.draft.explain import minutes_text, role_text
from draftai.features.roles import CHANNELS, elasticity, opportunity
from draftai.models.budget import allocate, apply_budget, apply_overrides, reprice
from draftai.models.components import linear_fp

G = 82


def _prev(stints: list[tuple[int, str, float]]) -> pl.DataFrame:
    """Last season's games: each (player, team, minutes) stint spread over 82 games."""
    rows = []
    for pid, team, minutes in stints:
        for g in range(G):
            rows.append({"player_id": pid, "team": team, "game_id": f"{team}{g}", "min": minutes / G, "season": 2020})
    return pl.DataFrame(rows)


def _rows(players: list[dict]) -> pl.DataFrame:
    base = {"min_1": 2000.0, "mpg_w": 25.0, "gp_1": 80, "gp_2": 80, "share_1": 0.95, "share_2": 0.95,
            "share_3": 0.95, "longest_absence": 0}
    return pl.DataFrame([{**base, **p} for p in players])


# 101 (guard) left A for B; 102 (guard) and 103 (big) stay on A; 104 (guard) is on B.
# 105 (wing, B) missed most of last season and is back; 106 (wing) is his teammate.
PLAYERS = [
    {"player_id": 101, "team_now": "B", "hbucket": "guard", "min_1": 2460.0},
    {"player_id": 102, "team_now": "A", "hbucket": "guard"},
    {"player_id": 103, "team_now": "A", "hbucket": "big"},
    {"player_id": 104, "team_now": "B", "hbucket": "guard"},
    {"player_id": 105, "team_now": "B", "hbucket": "wing", "min_1": 300.0, "mpg_w": 30.0, "gp_1": 10,
     "share_1": 0.12, "share_2": 0.85, "share_3": 0.85, "longest_absence": 60},
    {"player_id": 106, "team_now": "B", "hbucket": "wing"},
]
STINTS = [(101, "A", 2460.0), (102, "A", 2000.0), (103, "A", 2000.0), (104, "B", 2000.0), (105, "B", 300.0),
          (106, "B", 2000.0)]


def _opp(newcomers=None) -> dict[int, dict]:
    out = opportunity(_rows(PLAYERS), _prev(STINTS), newcomers)
    return {r["player_id"]: r for r in out.iter_rows(named=True)}


def test_departure_opens_minutes_at_his_position():
    o = _opp()
    assert o[102]["opp_pos_mpg"] > 10  # his guard teammate left
    assert o[103]["opp_dep"] == pytest.approx(0.0, abs=1e-9)  # guards and bigs do not share minutes
    assert o[104]["opp_pos_mpg"] < -10  # a guard arrived on his team


def test_own_arrival_and_return_are_not_competition():
    o = _opp()
    assert o[101]["opp_arr"] < o[104]["opp_arr"]  # the mover is not crowded out by himself
    assert o[105]["opp_ret"] < o[106]["opp_ret"]  # nor the returning player by his own return
    assert o[106]["opp_ret"] > 5  # his wing teammate's minutes come back


def test_channels_are_relative_to_the_average_team():
    o = _opp()
    # One guard's minutes left A for B: against the average of the two teams, A's guards
    # see +15 min/game of departures and B's guards -15.
    assert o[102]["opp_dep"] == pytest.approx(15.0) and o[104]["opp_dep"] == pytest.approx(-15.0)
    assert set(CHANNELS) <= set(o[102])


def test_rookies_fill_minutes_on_their_team():
    rookies = pl.DataFrame({"player_id": [900], "team": ["A"], "mpg": [28.0], "share": [0.8]})
    with_r, without = _opp(rookies), _opp()
    assert with_r[102]["opp_pos_mpg"] < without[102]["opp_pos_mpg"]
    assert with_r[104]["opp_pos_mpg"] > without[104]["opp_pos_mpg"]  # relative to an average team
    out = opportunity(_rows(PLAYERS), _prev(STINTS), rookies)
    a = out.filter(pl.col("team_now") == "A").row(0, named=True)
    assert a["team_new_emin"] == pytest.approx(28.0 * 0.8)


def test_elasticity_protects_starters():
    e = elasticity(np.array([8.0, 20.0, 36.0]))
    assert e[1] > e[0] and e[1] > e[2]
    assert e[2] < 0.3 * 36.0


def _team(mpg, share=None, team="A"):
    mpg = np.array(mpg, float)
    share = np.full_like(mpg, 0.8) if share is None else np.array(share, float)
    return np.array([team] * len(mpg), dtype=object), mpg, share


def test_budget_strength_scales_the_correction():
    team, mpg, share = _team([36, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10])
    S0 = (mpg * share).sum()
    target = S0 - 20.0
    assert np.allclose(allocate(team, mpg, share, target, 0.0), mpg)
    for lam in (0.25, 0.5, 1.0):
        new = allocate(team, mpg, share, target, lam)
        assert (new * share).sum() - target == pytest.approx((1 - lam) * (S0 - target))
    new = allocate(team, mpg, share, target, 1.0)
    cut = mpg - new
    assert cut[0] < cut[6]  # the 36-minute starter gives up less than a 22-minute rotation player


def test_budget_shares_with_rookies_it_does_not_score():
    team, mpg, share = _team([30, 25, 20])
    S0 = (mpg * share).sum()
    alone = allocate(team, mpg, share, S0 - 10, 1.0)
    shared = allocate(team, mpg, share, S0 - 10, 1.0, extra_e={"A": 0.0}, extra_w={"A": 1e9})
    assert np.allclose(shared, mpg, atol=1e-6)  # rookies with huge weight absorb it all
    assert (alone * share).sum() == pytest.approx(S0 - 10)


def _proj() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 16
    rows = {
        "sleeper_id": [str(i) for i in range(n)], "name": [f"P{i}" for i in range(n)],
        "team_now": ["A"] * 8 + ["B"] * 8,
        "positions": [["PG"], ["SG"], ["PG", "SG"], ["SF"], ["PF"], ["C"], ["C"], ["SF", "PF"]] * 2,
        "mpg_hat": rng.uniform(12, 36, n), "share_hat": rng.uniform(0.6, 0.95, n),
        "bonus_hat": rng.uniform(0, 1, n), "fp_p10": np.full(n, 10.0), "fp_p50": np.full(n, 20.0),
        "fp_p90": np.full(n, 30.0), "fp_direct_hat": np.full(n, 20.0),
    }
    for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m"):
        rows[f"rate_{c}_hat"] = rng.uniform(0.02, 0.5, n)
    df = pl.DataFrame(rows)
    rates = {c: df[f"rate_{c}_hat"].to_numpy() for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")}
    return df.with_columns(pl.Series("fp_pg_hat", linear_fp(rates, df["mpg_hat"].to_numpy(), S) + df["bonus_hat"]))


def test_reprice_follows_minutes():
    p = _proj()
    new = p["mpg_hat"].to_numpy() * 1.1
    q = reprice(p, new, S)
    rates = {c: q[f"rate_{c}_hat"].to_numpy() for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")}
    assert np.allclose(q["fp_pg_hat"], linear_fp(rates, new, S) + p["bonus_hat"].to_numpy() * 1.1)
    assert (q["fp_p90"] > p["fp_p90"]).all() and np.allclose(q["mpg_hat"], new)


def test_apply_budget_records_the_model_minutes():
    p = _proj()
    out = apply_budget(p, target=100.0, strength=0.5, settings=S)
    assert np.allclose(out["mpg_model"], p["mpg_hat"])
    assert np.allclose(out["budget_delta"], out["mpg_hat"] - out["mpg_model"])
    for t in ("A", "B"):
        g = out.filter(pl.col("team_now") == t)
        before = (g["mpg_model"] * g["share_hat"]).sum()
        after = (g["mpg_hat"] * g["share_hat"]).sum()
        assert after - 100.0 == pytest.approx(0.5 * (before - 100.0), rel=1e-6)


def test_override_is_absorbed_by_teammates_at_his_position():
    p = apply_budget(_proj(), target=200.0, strength=0.0, settings=S)
    o = pl.DataFrame({"sleeper_id": ["0", "999"], "mpg": [38.0, 20.0], "note": ["named starter", None]})
    out, log = apply_overrides(p, o, S)
    a0, a1 = out.filter(pl.col("team_now") == "A"), p.filter(pl.col("team_now") == "A")
    assert out["mpg_hat"][0] == 38.0 and out["minutes_override"][0] and out["minutes_note"][0] == "named starter"
    # The team's expected minutes are unchanged; the other team is untouched.
    assert (a0["mpg_hat"] * a0["share_hat"]).sum() == pytest.approx((a1["mpg_hat"] * a1["share_hat"]).sum())
    assert np.allclose(out.filter(pl.col("team_now") == "B")["mpg_hat"], p.filter(pl.col("team_now") == "B")["mpg_hat"])
    # Guards (rows 1 and 2 share PG/SG with row 0) give up more, relative to elasticity, than bigs.
    loss = (p["mpg_hat"] - out["mpg_hat"]).to_numpy()
    e = elasticity(p["mpg_hat"].to_numpy())
    assert loss[2] / e[2] > 2 * loss[5] / e[5]
    assert log.filter(pl.col("sleeper_id") == "999")["status"][0] == "not on the board"
    assert out["override_delta"][0] == pytest.approx(38.0 - p["mpg_hat"][0])


def test_minutes_overrides_file_is_logged(tmp_path):
    from draftai.models.project import apply_minutes_overrides

    p = apply_budget(_proj(), target=200.0, strength=0.0, settings=S)
    f = tmp_path / "minutes_overrides.csv"
    assert not apply_minutes_overrides(p, f, S, log=lambda *_: None)["minutes_override"].any()
    f.write_text("sleeper_id,name,mpg,note\n3,P3,30,starting\n")
    out = apply_minutes_overrides(p, f, S, log_dir=tmp_path, log=lambda *_: None)
    assert out["mpg_hat"][3] == 30.0
    apply_minutes_overrides(p, f, S, log_dir=tmp_path, log=lambda *_: None)
    logged = pl.read_csv(tmp_path / "minutes_overrides_log.csv")
    assert logged.height == 2 and (logged["status"] == "applied").all()


def test_card_shows_minutes_and_role():
    row = {"mpg_hat": 31.0, "mpg_model": 26.4, "minutes_override": True, "minutes_note": "named starter"}
    assert minutes_text(row) == "31.0 min/game (your override: named starter; model 26.4)"
    row = {"mpg_hat": 24.0, "mpg_model": 24.6, "budget_delta": -0.6, "override_delta": 0.0}
    assert "team minutes budget -0.6" in minutes_text(row)
    row = {"new_team": 1, "last_team": "MIL", "team_now": "MIA", "opp_pos_mpg": -18.0, "opp_dep": 1.0,
           "opp_arr": 3.0, "opp_new": 0.0, "opp_ret": 16.0}
    text = role_text(row)
    assert text.startswith("new team (MIL to MIA)") and "filled 18 min/game" in text and "teammates returning -16" in text
    assert role_text({"new_team": 0, "opp_pos_mpg": 2.0}) is None
