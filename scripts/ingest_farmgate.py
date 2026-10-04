"""Ingest public shrimp farm-gate price data.

Preferred source: Shrimp Insights Farm Gate Price portal.
The portal exposes a public Download data action but does not publish a
stable documented API. The script supports:
  1) SHRIMP_FARMGATE_URL pointing directly to the downloaded CSV/XLSX, and
  2) a local source file at data/raw/farmgate/source.(csv|xlsx).

The normalized output is monthly and keeps country/species/size explicitly.
"""
from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

PORTAL_URL = "https://www.shrimpinsights.com/price-portal"
OUT = Path("data/raw/farmgate/shrimp_insights_farmgate.csv")
LOCAL_SOURCE_CSV = Path("data/raw/farmgate/source.csv")
LOCAL_SOURCE_XLSX = Path("data/raw/farmgate/source.xlsx")


def _download(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0 shrimp-price-predictor/2.0"},
    )
    response.raise_for_status()
    return response.content


def _load_bytes(content: bytes, source_name: str) -> pd.DataFrame:
    if source_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(content))
    return pd.read_csv(BytesIO(content))


def _discover_download_url(html: str) -> str | None:
    # Best-effort only. The portal's download action is not a documented API.
    patterns = [
        r'https?://[^"\']+\.(?:csv|xlsx?|json)(?:\?[^"\']*)?',
        r'["\']([^"\']+\.(?:csv|xlsx?|json)(?:\?[^"\']*)?)["\']',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.I):
            candidate = match if isinstance(match, str) else match[0]
            if candidate.startswith("/"):
                return "https://www.shrimpinsights.com" + candidate
            if candidate.startswith("http"):
                return candidate
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        for c in df.columns
    ]

    def first(*names):
        for name in names:
            if name in df.columns:
                return name
        return None

    date_col = first("date", "week", "week_start", "period", "report_date", "date_week")
    country_col = first("country", "country_name", "origin")
    species_col = first("species", "species_name")
    size_col = first("size", "size_count", "count", "shrimp_size")
    price_col = first("price_usd_kg", "usd_kg", "price_usd", "price", "farm_gate_price")
    currency_col = first("currency", "currency_code")

    required = {
        "date": date_col,
        "country": country_col,
        "species": species_col,
        "size": size_col,
        "price": price_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Farm-gate dataset could not be normalized; missing fields: "
            + ", ".join(missing)
            + f". Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "country": df[country_col].astype(str).str.strip(),
            "species": df[species_col].astype(str).str.strip(),
            "size": pd.to_numeric(
                df[size_col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
                errors="coerce",
            ),
            "price_usd_kg": pd.to_numeric(df[price_col], errors="coerce"),
        }
    )
    if currency_col:
        out["currency"] = df[currency_col].astype(str).str.strip()
    else:
        out["currency"] = "USD"

    non_usd = out["price_usd_kg"].notna() & ~out["currency"].str.upper().isin({"USD", "US$"})
    if non_usd.any():
        raise ValueError(
            "Farm-gate file contains non-USD prices. Re-download the portal data "
            "with currency set to USD before running the pipeline."
        )

    out = out.dropna(subset=["date", "country", "species", "size", "price_usd_kg"])
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return (
        out.groupby(["date", "country", "species", "size"], as_index=False)["price_usd_kg"]
        .mean()
        .sort_values(["country", "species", "size", "date"])
    )


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    source_url = os.getenv("SHRIMP_FARMGATE_URL", "").strip()
    raw_df = None

    if source_url:
        raw_df = _load_bytes(_download(source_url), source_url)
        print(f"loaded farm-gate download: {source_url}")
    elif LOCAL_SOURCE_CSV.exists():
        raw_df = pd.read_csv(LOCAL_SOURCE_CSV)
        print(f"loaded local farm-gate source: {LOCAL_SOURCE_CSV}")
    elif LOCAL_SOURCE_XLSX.exists():
        raw_df = pd.read_excel(LOCAL_SOURCE_XLSX)
        print(f"loaded local farm-gate source: {LOCAL_SOURCE_XLSX}")
    else:
        try:
            html = _download(PORTAL_URL).decode("utf-8", errors="ignore")
            discovered = _discover_download_url(html)
            if discovered:
                raw_df = _load_bytes(_download(discovered), discovered)
                print(f"discovered farm-gate download: {discovered}")
        except Exception as exc:
            print(f"farm-gate portal discovery unavailable: {exc}")

    if raw_df is None:
        print(
            "WARNING: no farm-gate download was available. "
            "The export-price pipeline will still run. "
            "Set SHRIMP_FARMGATE_URL or place data/raw/farmgate/source.csv "
            "from the portal's Download data action to enable the farm-gate target."
        )
        return

    normalized = _normalize(raw_df)
    normalized.to_csv(OUT, index=False)
    print(f"saved {len(normalized):,} farm-gate monthly rows -> {OUT}")


if __name__ == "__main__":
    main()
