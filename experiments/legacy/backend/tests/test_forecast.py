from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_forecast_12():
    r = client.get("/api/forecast", params={"horizon": 12})
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 12
    p0 = data["predictions"][0]
    assert {"date", "prediction", "lower", "upper"} <= set(p0)
    assert p0["lower"] <= p0["prediction"] <= p0["upper"]
    assert all(p["prediction"] > 0 for p in data["predictions"])


def test_forecast_horizon_bounds():
    assert client.get("/api/forecast", params={"horizon": 24}).status_code == 200
    assert client.get("/api/forecast", params={"horizon": 0}).status_code == 422
    assert client.get("/api/forecast", params={"horizon": 999}).status_code == 422


def test_forecast_unknown_series():
    r = client.get("/api/forecast", params={"region": "Nowhere"})
    assert r.status_code == 400


def test_metrics_endpoint():
    r = client.get("/api/model/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["mae"] is not None and m["rmse"] is not None
    assert m["mae"] <= m["baseline_mae"] * 1.5
