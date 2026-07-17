import pytest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Neutraliza o backoff nos testes.

    Sem isto a suite gasta ~35s dormindo: o caminho de retry faz
    2**attempt + jitter, que soma ~15s por teste de esgotamento.
    O backoff em si e testado pela contagem de chamadas, nao pelo relogio.
    """
    monkeypatch.setattr("esci_ma.data.images.time.sleep", lambda _: None)
