import io

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID_YEARLY = (
    "year,avg_price_usd_kg,avg_temp_c,rainfall_mm,disease_outbreak_severity,disease_name,global_production_tonnes\n"
    "2025,19.0,29.0,1400,45,ehp,5600000\n"
    "2026,19.8,29.1,1390,40,ehp,5750000\n"
)
OLD_MONTHLY = (
    "year,month,region,size_class,price_usd_kg\n"
    "2024,1,Global,Medium,18.5\n"
)
MISSING_PRICE = "year,avg_temp_c\n2025,29.0\n"


@pytest.fixture()
def restore_yearly():
    import data_store
    snap = data_store.YEARLY_PATH.read_bytes() if data_store.YEARLY_PATH.exists() else None
    yield
    if snap is not None:
        data_store.YEARLY_PATH.write_bytes(snap)
    elif data_store.YEARLY_PATH.exists():
        data_store.YEARLY_PATH.unlink()


def test_upload_valid_yearly_merges(restore_yearly):
    before = len(client.get("/api/prices/yearly").json())
    r = client.post("/api/data/upload", files={"file": ("data.csv", io.BytesIO(VALID_YEARLY.encode()), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["rows_added"] >= 1 and body["retrained"] is True
    after = client.get("/api/prices/yearly").json()
    assert len(after) >= before + 1
    assert any(row["year"] == 2026 for row in after)


def test_old_monthly_csv_rejected_with_reason(restore_yearly):
    before = client.get("/api/prices/yearly").json()
    r = client.post("/api/data/upload", files={"file": ("old.csv", io.BytesIO(OLD_MONTHLY.encode()), "text/csv")})
    assert r.status_code == 400
    assert "yearly" in r.json()["detail"].lower()
    assert client.get("/api/prices/yearly").json() == before


def test_partial_upload_loses_nothing(restore_yearly):
    before = client.get("/api/prices/yearly").json()
    r = client.post("/api/data/upload", files={"file": ("bad.csv", io.BytesIO(MISSING_PRICE.encode()), "text/csv")})
    assert r.status_code == 400
    assert "avg_price_usd_kg" in r.json()["detail"]
    assert client.get("/api/prices/yearly").json() == before


def test_replace_mode(restore_yearly):
    r = client.post("/api/data/upload?mode=replace", files={"file": ("data.csv", io.BytesIO(VALID_YEARLY.encode()), "text/csv")})
    assert r.status_code == 200
    rows = client.get("/api/prices/yearly").json()
    assert [row["year"] for row in rows] == [2025, 2026]


def test_duplicate_years_update_in_merge(restore_yearly):
    dup = "year,avg_price_usd_kg\n2024,19.9\n"
    r = client.post("/api/data/upload", files={"file": ("dup.csv", io.BytesIO(dup.encode()), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["rows_added"] == 0 and body["rows_updated"] == 1
    rows = client.get("/api/prices/yearly").json()
    assert len([y for y in rows if y["year"] == 2024]) == 1
