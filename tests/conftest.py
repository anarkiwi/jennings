"""Put the repo root on sys.path so tests import the in-tree ``jennings``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402  pylint: disable=wrong-import-position

from jennings.memory import NATIVE, NativeMemory, ObservableMemory  # noqa: E402

BACKENDS = ["python", "native"] if NATIVE else ["python"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def new_mem(backend):
    """MPU memory for ``backend`` from optional bytes: None/bytearray (Python) or NativeMemory."""

    def make(data=None):
        if backend == "native":
            return NativeMemory(None if data is None else bytes(data))
        return None if data is None else bytearray(data)

    return make


@pytest.fixture
def new_observable(backend):
    """An observable memory for ``backend``, optionally over ``subject``'s contents."""

    def make(subject=None):
        if backend == "native":
            return NativeMemory(None if subject is None else bytes(subject))
        return ObservableMemory() if subject is None else ObservableMemory(subject=subject)

    return make
