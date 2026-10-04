"""Normalize the complete committed Shrimp Insights farm-gate dataset.

The committed source keeps every available country, species and size.
This script does not filter varieties or sizes.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import re

import pandas as pd
import requests

SOURCE = Path("data/farmgate/shrimp_insights_farmgate.csv")
OUT = Path("data/raw/farmgate/shrimp_insights_farmgate.csv")
PORTAL_URL = "https://www.shrimpinsights.com/price-portal"


COUNTRY_NAMES = {
    "in": "India",
    "vn": "Viet Nam",
    "id": "Indonesia",
    "ec": "Ecuador",
    "th": "Thailand",
}


def _load_source() -> pd.DataFrame:
    direct = os.getenv("SHRIMP_FARMGATE_URL", "").strip()
    if direct:
        response = requests.get(direct, timeout=120, headers={"User-Agent": "shrimp-price-predictor/3.0"})
        response.raise_for_status()
        if direct.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(response.content))
        return pd.read_csv(BytesIO(response.content))

    if SOURCE.exists():
        return pd.read_csv(SOURCE)

    raise FileNotFoundError(
        f"Missing {SOURCE}. Download the complete farm-gate dataset and commit it there."
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]

    required = ["year", "week_number", "country", "species", "size", "price_usd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Farm-gate schema missing columns: {missing}")

    out = pd.DataFrame()
    out["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    out["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
    out["country_code"] = df["country"].astype(str).str.strip().str.lower()
    out["country"] = out["country_code"].map(COUNTRY_NAMES).fillna(out["country_code"].str.upper())
    out["species"] = df["species"].astype(str).str.strip().str.lower()
    out["size"] = pd.to_numeric(df["size"], errors="coerce")
    out["price_usd_kg"] = pd.to_numeric(df["price_usd"], errors="coerce")
    out["price_local_kg"] = pd.to_numeric(
        df["price_local"], errors="coerce"
    ) if "price_local" in df.columns else float("nan")

    out["date"] = pd.to_datetime(
        out["year"].astype(str) + "-" + out["week_number"].astype(str) + "-1",
        format="%G-%V-%u",
        errors="coerce",
    )
    out = out.dropna(
        subset=["date", "country_code", "species", "size", "price_usd_kg"]
    )
    out = (
        out.groupby(
            ["date", "year", "week_number", "country_code", "country", "species", "size"],
            as_index=False,
        )
        .agg(
            price_usd_kg=("price_usd_kg", "mean"),
            price_local_kg=("price_local_kg", "mean"),
        )
        .sort_values(["country_code", "species", "size", "date"])
    )
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize(_load_source())
    normalized.to_csv(OUT, index=False)
    print(
        f"saved {len(normalized):,} rows, "
        f"{normalized['country_code'].nunique()} countries, "
        f"{normalized['species'].nunique()} species, "
        f"{normalized['size'].nunique()} sizes -> {OUT}"
    )


if __name__ == "__main__":
    main()
