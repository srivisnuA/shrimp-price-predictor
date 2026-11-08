import asyncio
import csv
import io

import pytest
from fastapi.testclient import TestClient

from backend.limits import UploadLimitMiddleware
from backend.main import create_app
from backend.settings import Settings

AUTH = {"Authorization": "Bearer unit-test-token"}
SERIES = "in:vannamei:40"


def test_health_readiness_and_security_headers(client):
    response = client.get("/api/health")
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert client.get("/api/ready").status_code == 503  # Only farmgate is installed in this fixture.
    assert client.get("/api/ready").json()["datasets"]["farmgate"]["ready"]


def test_serving_never_trains(client, monkeypatch):
    import shrimp_forecast.engine as engine

    monkeypatch.setattr(engine, "train", lambda *a, **k: pytest.fail("No request-time training"))
    monkeypatch.setattr(engine, "fit", lambda *a, **k: pytest.fail("No request-time fitting"))
    response = client.get("/api/v1/datasets/farmgate/forecast", params={"series_id": SERIES})
    assert response.status_code == 200
    result = response.json()
    assert len(result["predictions"]) == 4
    assert result["run_id"]
    assert result["currency"] == "USD"
    assert result["interval"]["guaranteed_coverage"] is False


def test_series_history_evaluation_and_quality_are_consistent(client):
    series = client.get("/api/v1/datasets/farmgate/series").json()
    assert len(series) == 12
    quality = client.get("/api/v1/datasets/farmgate/quality").json()
    assert quality["series"] == len(series)
    history = client.get("/api/v1/datasets/farmgate/history", params={"series_id": SERIES, "limit": 5}).json()
    assert len(history) == 5
    assert sorted(x["date"] for x in history) == [x["date"] for x in history]
    evaluation = client.get("/api/v1/datasets/farmgate/evaluation").json()
    assert evaluation["run_id"] == quality["run_id"]
    assert set(evaluation["partitions"]) == {"validation", "calibration", "test"}


@pytest.mark.parametrize(
    "path,params,code",
    [
        ("farmgate/forecast", {"series_id": SERIES, "horizon": 0}, 422),
        ("farmgate/forecast", {"series_id": SERIES, "horizon": 5}, 422),
        ("farmgate/forecast", {"series_id": "unknown"}, 404),
        ("farmgate/history", {"series_id": "unknown"}, 404),
        ("farmgate/history", {"series_id": SERIES, "limit": 99999}, 422),
        ("nonsense/series", {}, 422),
        ("benchmark/series", {}, 503),
    ],
)
def test_explicit_error_boundaries(client, path, params, code):
    response = client.get("/api/v1/datasets/" + path, params=params)
    assert response.status_code == code
    assert "detail" in response.json()


def test_forecast_csv_is_traceable(client):
    response = client.get(
        "/api/v1/datasets/farmgate/forecast.csv", params={"series_id": SERIES, "horizon": 2}
    )
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 2 and all(row["run_id"] for row in rows)


def test_upload_requires_auth_and_defaults_to_preview(client, source_csv):
    url = "/api/v1/datasets/farmgate/uploads?source=unit-test"
    files = {"file": ("data.csv", source_csv, "text/csv")}
    assert client.post(url, files=files).status_code == 401
    response = client.post(url, files=files, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["dry_run"] and not response.json()["staged"]
    assert not client.app.state.settings.upload_dir.exists()


def test_upload_stages_without_changing_live_model(client, source_csv):
    before = client.get("/api/v1/datasets/farmgate/quality").json()["run_id"]
    response = client.post(
        "/api/v1/datasets/farmgate/uploads?source=unit-test&dry_run=false",
        headers=AUTH,
        files={"file": ("../../escape.csv", source_csv, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["staged"] and not response.json()["active_model_changed"]
    assert client.get("/api/v1/datasets/farmgate/quality").json()["run_id"] == before
    saved = list(client.app.state.settings.upload_dir.rglob("*.csv"))
    assert len(saved) == 1 and saved[0].stem == response.json()["sha256"]


@pytest.mark.parametrize(
    "filename,raw,code",
    [
        ("data.exe", b"anything", 415),
        ("data.csv", b"year,price\n2024,4", 422),
        ("data.csv", b"x" * 2_100_000, 413),
    ],
)
def test_bad_uploads_preserve_data(client, filename, raw, code):
    response = client.post(
        "/api/v1/datasets/farmgate/uploads?source=test",
        headers=AUTH,
        files={"file": (filename, raw, "text/csv")},
    )
    assert response.status_code == code
    assert not client.app.state.settings.upload_dir.exists()


def test_intake_disabled_without_token(tmp_path):
    with TestClient(create_app(Settings(artifact_dir=tmp_path, admin_token=None))) as client:
        response = client.post(
            "/api/v1/datasets/farmgate/uploads?source=test", files={"file": ("x.csv", b"x")}
        )
        assert response.status_code == 403


def test_chunked_upload_is_bounded_before_parser():
    sent, called = [], []
    chunks = iter(
        [
            {"type": "http.request", "body": b"x" * 6, "more_body": True},
            {"type": "http.request", "body": b"x" * 6, "more_body": False},
        ]
    )

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    async def app(*args):
        called.append(True)

    asyncio.run(UploadLimitMiddleware(app, 10)({"type": "http", "method": "POST"}, receive, send))
    assert sent[0]["status"] == 413 and not called


def test_metrics_and_static_assets(client):
    assert "shrimp_http_requests_total" in client.get("/metrics").text
    assert "Shrimp Forecast" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    assert "https://cdn.jsdelivr.net" in client.get("/docs").headers["content-security-policy"]
    assert "https://cdn.jsdelivr.net" not in client.get("/").headers["content-security-policy"]
    assert client.get("/", headers={"host": "untrusted.example"}).status_code == 400
