import pandas as pd
import pytest

import data_store


def test_yearly_dataset_loads():
    df = data_store.load_yearly()
    assert 15 <= len(df) <= 40
    assert df["year"].is_monotonic_increasing
    assert not df["year"].duplicated().any()
    assert (df["avg_price_usd_kg"] > 0).all()
    assert df["disease_outbreak_severity"].dropna().between(0, 100).all()
    assert df["avg_temp_c"].dropna().between(20, 35).all()
    assert (df["year"].max() - df["year"].min()) >= 15


def test_yearly_records_filter():
    rows = data_store.yearly_records(year_from=2015, year_to=2020)
    assert rows and all(2015 <= r["year"] <= 2020 for r in rows)


def test_validate_yearly_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        data_store._validate_yearly(pd.DataFrame({"year": [2020]}))


def test_validate_yearly_dedupes_years():
    df = pd.DataFrame([
        {"year": 2020, "avg_price_usd_kg": 10},
        {"year": 2020, "avg_price_usd_kg": 11},
        {"year": 2021, "avg_price_usd_kg": 12},
    ])
    out = data_store._validate_yearly(df)
    assert len(out) == 2
    assert out[out["year"] == 2020]["avg_price_usd_kg"].iloc[0] == 11


def test_monthly_still_works():
    df = data_store.load()
    assert not df.empty and "region" in df.columns
