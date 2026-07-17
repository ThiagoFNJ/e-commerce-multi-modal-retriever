import pytest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Neutralise the backoff during tests.

    Without this the suite spends ~35s sleeping: the retry path does
    2**attempt + jitter, which sums to ~15s per exhaustion test.
    The backoff itself is covered by call count, not by the clock.
    """
    monkeypatch.setattr("esci_ma.data.images.time.sleep", lambda _: None)
