import numpy as np
import pandas as pd
import pytest

from shrimp_forecast.data import build_existing, fingerprint, quality, read_csv, validate_observations


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("price_usd_kg", np.inf, "finite"),
        ("price_usd_kg", -1, "positive"),
        ("price_usd_kg", "bad", "finite"),
        ("country", None, "blank"),
        ("species", " ", "characters"),
        ("date", "2023-12-26", "Mondays"),
        ("available_at", "1999-01-01", "precede"),
        ("date", "garbage", "ISO dates"),
    ],
)
def test_rejects_bad_rows(panel, column, value, message):
    frame = panel.copy()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_observations(frame, "farmgate")


def test_duplicates_are_not_silently_averaged(panel):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_observations(pd.concat([panel, panel.iloc[:1]]), "farmgate")


def test_metadata_consistency(panel):
    changed = panel.copy()
    changed.loc[0, "country"] = "Ecuador"
    with pytest.raises(ValueError, match="constant"):
        validate_observations(changed, "farmgate")


def test_contract_and_fingerprint_roundtrip(panel, source_csv):
    assert fingerprint(read_csv(source_csv, "farmgate")) == fingerprint(panel)
    assert fingerprint(validate_observations(panel.sample(frac=1), "farmgate")) == fingerprint(panel)
    with pytest.raises(ValueError, match="Missing"):
        validate_observations(panel.drop(columns="available_at"), "farmgate")


def test_quality_counts_gaps_without_filling(panel):
    frame = panel.drop(index=12)
    report = quality(frame, "farmgate", pd.Timestamp("2024-08-01"))
    assert report["rows"] == len(panel) - 1
    assert sum(row["missing_periods"] for row in report["series_details"]) == 1


def test_existing_build_is_offline_and_separates_samples(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Build must stay offline"))
    reports = build_existing(output=tmp_path, as_of="2026-11-09")
    assert reports["farmgate"]["rows"] == 1000
    assert reports["farmgate"]["series"] == 64
    assert "illustrative" in reports["benchmark"]["source"]
    assert "weather" not in read_csv((tmp_path / "farmgate.csv").read_bytes(), "farmgate").columns
    assert (tmp_path / "benchmark.quality.json").exists()


def test_asof_build_excludes_future_source_rows(tmp_path):
    report = build_existing(output=tmp_path, as_of="2026-06-01")
    assert report["farmgate"]["excluded_future_rows"] > 0
    assert report["farmgate"]["end"] <= "2026-06-01"
