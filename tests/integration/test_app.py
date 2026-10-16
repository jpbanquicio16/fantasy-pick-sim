"""The Streamlit app: the connect screen without a board or network, and an offline mock
whose other teams draft themselves (synthetic board, no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from draftai.draft.opponents import OpponentModel

APP = Path(__file__).parents[2] / "app" / "streamlit_app.py"


def test_connect_screen_renders():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any(b.label == "Start offline mock" for b in at.button)


def _button(at, label_start: str):
    return next(b for b in at.button if b.label.startswith(label_start))


@pytest.fixture
def offline_app(board, monkeypatch):
    from draftai import config, pipeline

    monkeypatch.setattr(pipeline, "load_board", lambda paths, season: board)
    monkeypatch.setattr(pipeline, "live_opponents", lambda *a, **k: OpponentModel(c0=1.0, c1=-0.7))
    # No league connection: the mock uses the offline settings typed in the form.
    monkeypatch.setattr(config, "load_league_config",
                        lambda path=None: config.LeagueConfig(teams=6, rounds=12, draft_slot=2))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_mock_auto_draft_end_to_end(offline_app):
    at = offline_app
    _button(at, "Start offline mock").click().run()
    assert not at.exception
    assert any("You are on the clock" in s.value for s in at.success)  # team 1 picked by itself
    assert any("Draft log" in h.value for h in at.subheader)
    _button(at, "Draft ").click().run()  # take the recommendation
    assert not at.exception
    assert any("You are on the clock" in s.value for s in at.success)  # the room drafted back to you
    _button(at, "Auto-draft the rest").click().run()
    assert not at.exception
    assert any("Mock complete" in s.value for s in at.success)
    assert any(m.label == "P(playoffs)" for m in at.metric)
