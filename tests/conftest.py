"""Deterministic, headless defaults for package tests.

Tests use the installed package; install with ``pip install -e '.[dev,ppo]'``.
The integration fixtures are the small NYC inputs committed under data/.
"""

import os
from uuid import uuid4

import pytest

# Configure before test modules import NumPy, PyTorch, or Pygame. No test should
# open a window or consume a machine-wide pool of numerical worker threads.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Avoid sharing pytest's Windows temp root across execution identities."""
    if config.option.basetemp is not None or os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return
    parent = config.rootpath / "tests" / ".tmp"
    parent.mkdir(parents=True, exist_ok=True)
    # pytest clears an explicit basetemp before use. Give it a fresh path so it
    # cannot clear another run's artifacts, including concurrent test runs.
    destination = parent / f"run-{uuid4().hex}"
    if destination.exists():
        raise FileExistsError(destination)
    config.option.basetemp = str(destination)
