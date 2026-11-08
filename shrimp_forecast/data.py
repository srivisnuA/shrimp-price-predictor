"""Strict observation contracts. Targets are never interpolated or substituted."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["date", "available_at", "series_id", "country", "species", "size", "price_usd_kg"]
COUNTRIES = {"in": "India", "vn": "Viet Nam", "id": "Indonesia", "ec": "Ecuador", "th": "Thailand"}
FREQUENCIES = {"farmgate": "W-MON", "benchmark": "MS"}


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, allow_nan=False, sort_keys=True).encode()


def validate_observations(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset not in FREQUENCIES:
        raise ValueError("Unknown dataset")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Dataset is empty")
    out = df[COLUMNS].copy()
    for column in ["date", "available_at"]:
        parsed = pd.to_datetime(out[column], errors="coerce", format="ISO8601", utc=True)
        if parsed.isna().any() or not parsed.eq(parsed.dt.normalize()).all():
            raise ValueError(f"{column} must contain valid midnight ISO dates")
        out[column] = parsed.dt.tz_localize(None)
    for column in ["series_id", "country", "species", "size"]:
        if out[column].isna().any():
            raise ValueError(f"{column} cannot be blank")
        out[column] = out[column].astype(str).str.strip()
        if out[column].str.len().eq(0).any() or out[column].str.len().gt(120).any():
            raise ValueError(f"{column} must have 1–120 characters")
    out["price_usd_kg"] = pd.to_numeric(out["price_usd_kg"], errors="coerce")
    if not (np.isfinite(out.price_usd_kg) & out.price_usd_kg.gt(0)).all():
        raise ValueError("price_usd_kg must be finite and positive")
    if out.duplicated(["series_id", "date"]).any():
        raise ValueError("Duplicate series/date keys; resolve revisions explicitly before ingestion")
    if out.available_at.lt(out.date).any():
        raise ValueError("available_at cannot precede the observation date")
    valid_dates = out.date.dt.dayofweek.eq(0) if dataset == "farmgate" else out.date.dt.day.eq(1)
    if not valid_dates.all():
        raise ValueError("Dates must be Mondays (farmgate) or month starts (benchmark)")
    if out.groupby("series_id")[["country", "species", "size"]].nunique().gt(1).any().any():
        raise ValueError("Series metadata must stay constant")
    return out.sort_values(["date", "series_id"]).reset_index(drop=True)


def canonical_bytes(df: pd.DataFrame) -> bytes:
    return df[COLUMNS].to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g").encode()


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_bytes(df)).hexdigest()


def read_csv(raw: bytes, dataset: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype={"size": str}, float_precision="round_trip")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Could not parse UTF-8 CSV") from exc
    return validate_observations(df, dataset)


def quality(df: pd.DataFrame, dataset: str, as_of: pd.Timestamp | None = None) -> dict:
    as_of = pd.Timestamp(as_of or date.today()).normalize()
    details = []
    for series_id, group in df.groupby("series_id", sort=True):
        expected = pd.date_range(group.date.min(), group.date.max(), freq=FREQUENCIES[dataset])
        details.append(
            {
                "series_id": series_id,
                "country": group.country.iloc[0],
                "species": group.species.iloc[0],
                "size": group["size"].iloc[0],
                "observations": len(group),
                "start": str(group.date.min().date()),
                "end": str(group.date.max().date()),
                "missing_periods": len(expected.difference(group.date)),
                "staleness_days": max(0, (as_of - group.date.max()).days),
            }
        )
    return {
        "dataset": dataset,
        "frequency": FREQUENCIES[dataset],
        "rows": len(df),
        "dates": int(df.date.nunique()),
        "series": len(details),
        "series_details": details,
        "start": str(df.date.min().date()),
        "end": str(df.date.max().date()),
        "sha256": fingerprint(df),
        "future_or_unreleased_rows": int((df.available_at.gt(as_of)).sum()),
        "verification": "unverified_existing_data",
        "availability_assumption": "Bundled observations are assumed available on their labeled date; publication vintages are absent.",
    }


def build_existing(root: Path = ROOT, output: Path | None = None, as_of: str | None = None) -> dict:
    """Build only committed data, without a network dependency or invented context."""
    output = output or root / "data" / "processed"
    cutoff = pd.Timestamp(as_of or date.today()).normalize()
    raw = pd.read_csv(root / "data/farmgate/shrimp_insights_farmgate.csv")
    dates = pd.to_datetime(
        raw.Year.astype(str) + "-" + raw["Week number"].astype(str) + "-1", format="%G-%V-%u"
    )
    # strptime accepts week 53 for some 52-week years: verify the ISO year/week round trip.
    iso = dates.dt.isocalendar()
    if not (iso.year.eq(raw.Year) & iso.week.eq(raw["Week number"])).all():
        raise ValueError("Invalid ISO week in farmgate source")
    farm = pd.DataFrame(
        {
            "date": dates,
            "available_at": dates,
            "country": raw.Country.map(COUNTRIES),
            "species": raw.Species.str.lower(),
            "size": raw.Size.astype(str),
            "price_usd_kg": raw["Price USD"],
        }
    )
    farm["series_id"] = raw.Country + ":" + farm.species + ":" + farm["size"]
    sample = pd.read_csv(root / "backend/data/shrimp_prices_sample.csv")
    sample_dates = pd.to_datetime(dict(year=sample.year, month=sample.month, day=1))
    benchmark = pd.DataFrame(
        {
            "date": sample_dates,
            "available_at": sample_dates,
            "series_id": sample.region + ":" + sample.size_class,
            "country": sample.region,
            "species": "unspecified",
            "size": sample.size_class,
            "price_usd_kg": sample.price_usd_kg,
        }
    )
    reports = {}
    for name, frame in {"farmgate": farm, "benchmark": benchmark}.items():
        clean = validate_observations(frame, name)
        excluded = int(clean.available_at.gt(cutoff).sum())
        clean = clean[clean.available_at.le(cutoff)].reset_index(drop=True)
        if clean.empty:
            raise ValueError(f"No {name} observations available at cutoff")
        report = quality(clean, name, cutoff)
        report["excluded_future_rows"] = excluded
        report["as_of"] = str(cutoff.date())
        report["source"] = (
            "Committed farm-gate download"
            if name == "farmgate"
            else "Bundled illustrative monthly sample; not verified export prices"
        )
        atomic_bytes(output / f"{name}.csv", canonical_bytes(clean))
        atomic_bytes(output / f"{name}.quality.json", json_bytes(report))
        reports[name] = report
    return reports
