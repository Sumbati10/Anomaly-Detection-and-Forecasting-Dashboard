from __future__ import annotations

from fastapi.testclient import TestClient

from src.app.main import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_forecast_smoke() -> None:
    client = TestClient(app)
    r = client.get("/forecast", params={"horizon": 7})
    assert r.status_code == 200
    payload = r.json()
    assert payload["horizon"] == 7
    assert len(payload["rows"]) == 7
