"""Keep the developer's real configuration out of the test suite.

`core.llm` loads `.env` so that scripts work without ceremony. That makes tests depend
on whether the machine happens to have one, and on what it says -- a suite that passes
on a fresh clone and fails once you configure an endpoint is worse than useless.

Every test therefore runs with the VERIMEM_* environment cleared. A test that wants a
setting sets it explicitly.
"""

import os

import pytest

VERIMEM_VARS = [
    "VERIMEM_API_BASE_URL",
    "VERIMEM_API_KEY",
    "VERIMEM_API_MODEL",
    "VERIMEM_API_SPEND_CAP_USD",
    "VERIMEM_DISABLE_THINKING",
    "VERIMEM_CACHE_DIR",
    "VERIMEM_DATA_DIR",
    "VERIMEM_RESULTS_DIR",
    "VERIMEM_SEARCH_PROVIDER",
    "VERIMEM_DISABLE_CACHE",
]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for var in VERIMEM_VARS:
        monkeypatch.delenv(var, raising=False)
    # core.llm._load_env would put them straight back from .env
    monkeypatch.setattr("core.llm._load_env", lambda: None, raising=False)
    yield


@pytest.fixture(autouse=True)
def _no_accidental_network(monkeypatch, request):
    """Nothing in the suite should open a socket. Fail loudly if something tries."""
    if request.node.get_closest_marker("slow"):
        return
    import socket

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "a test tried to open a network connection; mark it @pytest.mark.slow "
            "or inject a fake transport"
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def pytest_configure(config):
    os.environ.setdefault("VERIMEM_DISABLE_CACHE", "0")
