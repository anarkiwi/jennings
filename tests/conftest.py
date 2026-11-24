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


def _factory(backend, python):
    """``make(data=None)`` building ``NativeMemory`` for the native backend, else ``python(data)``."""

    def make(data=None):
        if backend == "native":
            return NativeMemory(None if data is None else bytes(data))
        return python(data)

    return make


@pytest.fixture
def new_mem(backend):
    """MPU memory for ``backend`` from optional bytes: None/bytearray (Python) or NativeMemory."""
    return _factory(backend, lambda data: None if data is None else bytearray(data))


@pytest.fixture
def new_observable(backend):
    """An observable memory for ``backend``, optionally over ``subject``'s contents."""
    return _factory(
        backend,
        lambda subject: (
            ObservableMemory() if subject is None else ObservableMemory(subject=subject)
        ),
    )
