from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_prices_endpoint():
    r = client.get("/api/prices")
    assert r.status_code == 200
    rows = r.json()
    assert rows and {"year", "month", "region", "size_class", "price_usd_kg"} <= set(rows[0])


def test_price_filters():
    r = client.get("/api/prices", params={"region": "Asia"})
    assert all(row["region"] == "Asia" for row in r.json())
    r = client.get("/api/prices", params={"region": "Nowhere"})
    assert r.json() == []


def test_dimensions_endpoint():
    r = client.get("/api/dimensions")
    assert r.status_code == 200
    assert r.json()["regions"]
