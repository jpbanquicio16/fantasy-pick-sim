"""The FastAPI app (V5) on the synthetic board: mocks that draft themselves, manual drafts,
recommendations with and without rollouts, the WebSocket stream."""

import pytest
from fastapi.testclient import TestClient
from synthetic import ROSTER

from draftai.api.app import create_app
from draftai.api.service import DraftService, Engine
from draftai.config import LeagueConfig, SimConfig
from draftai.draft.league import LeagueContext
from draftai.draft.opponents import OpponentModel

SIM = SimConfig(candidate_pool=20, shortlist=6, rollout_worlds=60, rollout_shortlist=3, rollout_race_after=40)
ROOM = OpponentModel(c0=1.0, c1=-0.7, gamma=0.8)


@pytest.fixture(scope="module")
def client(board):
    ctx = LeagueContext.offline(LeagueConfig(teams=8, rounds=13, draft_slot=2, roster_positions=ROSTER))
    app = create_app(DraftService(Engine(ctx, board, SIM, ROOM)), web_dist=None)
    return TestClient(app)


def test_league_and_board(client, board):
    lg = client.get("/api/league").json()
    assert lg["teams"] == 8 and lg["my_slot"] == 2 and lg["players"] == board.players.height
    rows = client.get("/api/board").json()
    assert len(rows) == board.players.height and {"sleeper_id", "fp_pg", "vor_per_week", "adp"} <= set(rows[0])
    p = client.get(f"/api/players/{rows[0]['sleeper_id']}").json()
    assert len(p["weekly"]["value"]) == len(p["weekly"]["weeks"]) > 0
    assert client.get("/api/players/nobody").status_code == 404


def test_mock_drafts_itself_around_your_picks(client):
    st = client.post("/api/drafts", json={"mode": "mock", "slot": 3, "seed": 5}).json()
    did = st["id"]
    assert st["my_turn"] and st["current_pick"] == 3 and [p["by"] for p in st["picks"]] == ["auto", "auto"]
    rec = client.get(f"/api/drafts/{did}/recommendation").json()
    assert rec["pick"] == 3 and rec["card"]["player"] == rec["candidates"][0]["name"]
    taken = st["picks"][0]["sleeper_id"]
    assert client.post(f"/api/drafts/{did}/picks", json={"sleeper_id": taken}).status_code == 409
    st = client.post(f"/api/drafts/{did}/picks", json={"sleeper_id": rec["candidates"][0]["sleeper_id"]}).json()
    assert st["my_turn"] and st["current_pick"] > 3
    assert [r["name"] for r in st["roster"] if r["name"]] == [rec["candidates"][0]["name"]]
    assert set(st["survival"]).isdisjoint(st["drafted"])
    st = client.post(f"/api/drafts/{did}/undo").json()
    assert st["current_pick"] == 3 and st["my_turn"]
    # Undo, then a different pick: the recommendation is recomputed, not the cached one.
    before = client.get(f"/api/drafts/{did}/recommendation").json()
    other = rec["candidates"][1]["sleeper_id"]
    st = client.post(f"/api/drafts/{did}/picks", json={"sleeper_id": other}).json()
    after = client.get(f"/api/drafts/{did}/recommendation").json()
    assert other not in {c["sleeper_id"] for c in after["candidates"]}
    assert set(st["drafted"]).isdisjoint(c["sleeper_id"] for c in after["candidates"])
    assert before["pick"] == 3 and after["pick"] == st["current_pick"]
    st = client.post(f"/api/drafts/{did}/undo").json()
    wi = client.get(f"/api/drafts/{did}/what-if/{rec['candidates'][2]['sleeper_id']}").json()
    assert wi["score_gap_pw"] >= 0
    st = client.post(f"/api/drafts/{did}/autodraft", json={"rollout": False}).json()
    assert st["complete"] and len(st["picks"]) == st["total_picks"]
    proj = client.get(f"/api/drafts/{did}/projection").json()
    assert len(proj) == 8 and all(0 <= t["p_playoffs"] <= 1 for t in proj)
    # The same seed replays the same room.
    again = client.post("/api/drafts", json={"mode": "mock", "slot": 3, "seed": 5}).json()
    assert [p["sleeper_id"] for p in again["picks"]] == [p["sleeper_id"] for p in client.get(
        f"/api/drafts/{did}").json()["picks"][:2]]


def test_rollout_recommendation_has_confidence(client):
    did = client.post("/api/drafts", json={"mode": "mock", "slot": 1, "seed": 1}).json()["id"]
    rec = client.get(f"/api/drafts/{did}/recommendation", params={"rollout": True}).json()
    ro = rec["rollout"]
    assert ro["worlds"] == 60 and 0 <= ro["confidence"] <= 1
    assert rec["card"]["confidence"] == ro["confidence"] and ro["candidates"][0]["gap_pw"] == 0


def test_manual_draft_records_every_team(client):
    st = client.post("/api/drafts", json={"mode": "manual", "slot": 2}).json()
    did = st["id"]
    board = client.get("/api/board").json()
    for r in board[:3]:
        st = client.post(f"/api/drafts/{did}/picks", json={"sleeper_id": r["sleeper_id"]}).json()
    assert st["current_pick"] == 4 and [p["slot"] for p in st["picks"]] == [1, 2, 3]
    assert st["picks"][1]["by"] == "you" and st["picks"][0]["team"] == "Team 1"
    st = client.post(f"/api/drafts/{did}/undo").json()
    assert st["current_pick"] == 3
    assert client.get("/api/drafts/nope").status_code == 404
    assert client.post("/api/drafts", json={"mode": "live"}).status_code == 422


def test_websocket_pushes_state_on_every_pick(client):
    st = client.post("/api/drafts", json={"mode": "mock", "slot": 2, "seed": 9}).json()
    did = st["id"]
    with client.websocket_connect(f"/api/drafts/{did}/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "state" and first["state"]["version"] == st["version"]
        rec = client.get(f"/api/drafts/{did}/recommendation").json()
        client.post(f"/api/drafts/{did}/picks", json={"sleeper_id": rec["candidates"][0]["sleeper_id"]})
        pushed = ws.receive_json()
        assert pushed["state"]["version"] > st["version"]


def test_minutes_overrides_save_and_rebuild(board, tmp_path):
    ctx = LeagueContext.offline(LeagueConfig(teams=8, rounds=13, draft_slot=2, roster_positions=ROSTER))
    rebuilt = []
    path = tmp_path / "minutes_overrides.csv"
    app = create_app(DraftService(Engine(ctx, board, SIM, ROOM)), rebuild=lambda p: rebuilt.append(p) or board,
                     overrides_path=path, web_dist=None)
    c = TestClient(app)
    assert c.get("/api/overrides/minutes").json() == []
    sid = board.players["sleeper_id"][0]
    r = c.put("/api/overrides/minutes", json=[{"sleeper_id": sid, "name": "x", "mpg": 30.5, "note": "starting"}])
    assert r.json()["saved"] == 1 and rebuilt == [path]
    assert c.get("/api/overrides/minutes").json()[0]["mpg"] == 30.5
    assert c.put("/api/overrides/minutes", json=[{"sleeper_id": sid, "mpg": 60}]).status_code == 422
