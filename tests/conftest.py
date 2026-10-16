import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_league  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def league():
    return make_league()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
