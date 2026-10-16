"""The live poller against a recorded Sleeper draft (fixture in the API's format)."""

import json

import pytest

from draftai.config import LeagueConfig
from draftai.draft.league import LeagueContext
from draftai.ingestion.sleeper import SleeperDraft, SleeperLeague, SleeperPick, TradedPick
from draftai.live.poller import LiveDraft, state_from_picks


class FakeClient:
    """Serves a recorded draft one more pick per poll."""

    def __init__(self, rec: dict):
        self.rec = rec
        self.n = 0

    def league(self, league_id, ttl=None):
        return SleeperLeague(**self.rec["league"])

    def draft(self, draft_id, ttl=0):
        return SleeperDraft(**self.rec["draft"])

    def draft_picks(self, draft_id, ttl=0):
        self.n += 1
        return [SleeperPick(**p) for p in self.rec["picks"][: self.n]]

    def traded_picks(self, draft_id, ttl=60):
        return [TradedPick(**t) for t in self.rec["traded_picks"]]

    def league_users(self, league_id):
        return self.rec["users"]

    def user(self, name):
        return next((u for u in self.rec["users"] if u["display_name"] == name), None)


@pytest.fixture
def recorded(fixtures_dir):
    return json.loads((fixtures_dir / "sleeper_draft_recorded.json").read_text())


def test_context_from_recorded_league(recorded):
    ctx = LeagueContext.from_sleeper(LeagueConfig(), FakeClient(recorded), league_id="L1", username="me")
    assert ctx.teams == 4 and ctx.rounds == 3 and ctx.reversal_round == 3
    assert ctx.my_slot == 2
    assert ctx.roster_positions[:3] == ["PG", "SG", "G"]
    assert ctx.traded == [(3, 4, 2)]  # roster 4 (slot 4) traded its round-3 pick to roster 2 (slot 2)
    assert ctx.scoring.weights["pts"] == 0.5


def test_poller_follows_picks_and_detects_changes(recorded):
    client = FakeClient(recorded)
    ctx = LeagueContext.from_sleeper(LeagueConfig(), client, league_id="L1", username="me")
    live = LiveDraft(ctx, client)
    seen = []
    live.follow(lambda s: seen.append(s.version), interval=0, max_polls=20)
    assert seen == list(range(1, len(recorded["picks"]) + 1))
    assert live.state.complete
    assert live.state.roster(2) == ["p2", "p7", "p9", "p11"]  # includes the traded pick 9


def test_out_of_order_picks_rejected(recorded):
    ctx = LeagueContext.from_sleeper(LeagueConfig(), FakeClient(recorded), league_id="L1", username="me")
    picks = [SleeperPick(**p) for p in recorded["picks"]]
    with pytest.raises(ValueError):
        state_from_picks(ctx, [picks[0], picks[2]])
