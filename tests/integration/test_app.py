"""The Streamlit app renders its connect screen without a board or network."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[2] / "app" / "streamlit_app.py"


def test_connect_screen_renders():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any(b.label == "Start offline mock" for b in at.button)
