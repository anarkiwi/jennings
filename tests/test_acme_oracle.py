"""jennings.acme differentially against the installed acme binary (skipped without it)."""

import random
import sys
from pathlib import Path

import pytest

from acme_cases import ERRORS, SOURCES

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import acme_oracle  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(acme_oracle.ACME is None, reason="acme is not installed")


@pytest.mark.parametrize("source", list(SOURCES.values()) + sorted(ERRORS))
def test_cases(source):
    assert not acme_oracle.differences(source)


@pytest.mark.parametrize("seed", range(8))
def test_fuzz(seed):
    rng = random.Random(seed)
    for _ in range(40):
        text = acme_oracle.program(rng)
        assert not acme_oracle.differences(text), text
