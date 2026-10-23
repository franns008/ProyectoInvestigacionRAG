"""
Tests de los reintentos de descarga (src/ingestion/_http.py). Sin red: `requests.get` y
`time.sleep` se reemplazan.
"""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ingestion"))

import _http  # noqa: E402

URL = "https://example.test/feed"


def response(status: int) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = URL
    return resp


@pytest.fixture
def fake_get(monkeypatch):
    """`requests.get` devuelve (o lanza) cada elemento de `outcomes`, en orden."""
    waits = []
    monkeypatch.setattr(_http.time, "sleep", waits.append)

    def install(*outcomes):
        pending = list(outcomes)

        def get(url, timeout, **kwargs):
            outcome = pending.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(_http.requests, "get", get)
        return waits

    return install


def test_timeout_y_despues_ok_devuelve_la_respuesta(fake_get):
    waits = fake_get(requests.ReadTimeout("Read timed out."), response(200))
    assert _http.get_with_retries(URL, "test").status_code == 200
    assert waits == [2]


def test_corte_de_conexion_en_todos_los_intentos_relanza(fake_get):
    waits = fake_get(*[requests.ConnectionError("Read timed out.")] * _http.MAX_RETRIES)
    with pytest.raises(requests.ConnectionError):
        _http.get_with_retries(URL, "test")
    assert waits == [2, 4]


def test_status_malo_y_despues_ok_sigue_reintentando(fake_get):
    waits = fake_get(response(503), response(200))
    assert _http.get_with_retries(URL, "test").status_code == 200
    assert waits == [2]


def test_status_malo_persistente_levanta_http_error(fake_get):
    fake_get(*[response(404)] * _http.MAX_RETRIES)
    with pytest.raises(requests.HTTPError):
        _http.get_with_retries(URL, "test")


def test_max_retries_y_timeout_configurables(monkeypatch):
    waits = []
    timeouts = []
    monkeypatch.setattr(_http.time, "sleep", waits.append)

    def get(url, timeout, **kwargs):
        timeouts.append(timeout)
        raise requests.ReadTimeout("Read timed out.")

    monkeypatch.setattr(_http.requests, "get", get)
    with pytest.raises(requests.ReadTimeout):
        _http.get_with_retries(URL, "test", max_retries=5, timeout=90)
    assert waits == [2, 4, 8, 16]
    assert timeouts == [90] * 5
