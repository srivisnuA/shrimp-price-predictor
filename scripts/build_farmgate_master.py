"""Build the weekly farm-gate modeling table and attach monthly market/weather/disease context."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

COUNTRY_REGION = {
    "in": "india_andhra_pradesh",
    "vn": "vietnam_mekong",
    "id": "indonesia_sumatra",
    "ec": "ecuador_guayas",
    "th": "thailand_eastern",
}

EXOGENOUS = [
    "t2m",
    "prectotcorr",
    "disease_event_count",
    "disease_severity",
    "shrimp_production_tonnes",
    "usd_inr",
    "wti_usd_bbl",
    "us_cpi",
    "price_usd_kg",
    "trade_unit_value_usd_kg",
]


def load_farmgate():
    p = Path("data/raw/farmgate/shrimp_insights_farmgate.csv")
    if not p.exists():
        raise FileNotFoundError("Run scripts/ingest_farmgate.py first")
    df = pd.read_csv(p, parse_dates=["date"])
    df["country_code"] = df["country_code"].astype(str).str.lower()
    df["region"] = df["country_code"].map(COUNTRY_REGION)
    return df


def load_context():
    p = Path("data/processed/shrimp_master_monthly.csv")
    if not p.exists():
        return pd.DataFrame(columns=["region", "date", *EXOGENOUS])

    cols = ["region", "date", *EXOGENOUS]
    df = pd.read_csv(p, usecols=lambda c: c in cols, parse_dates=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    # There can be two India regions. Prefer Andhra Pradesh for the national
    # India farm-gate series because the farm-gate source is country-level.
    df = (
        df.sort_values(["region", "date"])
        .drop_duplicates(["region", "date"], keep="first")
    )
    rename = {"price_usd_kg": "global_price_usd_kg", "trade_unit_value_usd_kg": "export_price_usd_kg"}
    return df.rename(columns=rename)


def main():
    fg = load_farmgate()
    fg["month_date"] = fg["date"].dt.to_period("M").dt.to_timestamp()

    context = load_context()
    if not context.empty:
        fg = fg.merge(
            context,
            left_on=["region", "month_date"],
            right_on=["region", "date"],
            how="left",
            suffixes=("", "_context"),
        )
        fg = fg.drop(columns=["date_context"], errors="ignore")

    fg["week_of_year"] = fg["date"].dt.isocalendar().week.astype(int)
    fg["month"] = fg["date"].dt.month.astype(int)

    key = ["country_code", "species", "size"]
    g = fg.sort_values(key + ["date"]).groupby(key)["price_usd_kg"]
    fg["price_usd_kg_lag1"] = g.shift(1)
    fg["price_usd_kg_lag2"] = g.shift(2)
    fg["price_usd_kg_roll4"] = g.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    # Make the context names explicit for the farm-gate model.
    if "t2m" not in fg:
        fg["t2m"] = np.nan
    if "prectotcorr" not in fg:
        fg["prectotcorr"] = np.nan
    if "disease_event_count" not in fg:
        fg["disease_event_count"] = np.nan
    if "disease_severity" not in fg:
        fg["disease_severity"] = np.nan
    if "shrimp_production_tonnes" not in fg:
        fg["shrimp_production_tonnes"] = np.nan
    if "usd_inr" not in fg:
        fg["usd_inr"] = np.nan
    if "wti_usd_bbl" not in fg:
        fg["wti_usd_bbl"] = np.nan
    if "us_cpi" not in fg:
        fg["us_cpi"] = np.nan
    if "global_price_usd_kg" not in fg:
        fg["global_price_usd_kg"] = np.nan
    if "export_price_usd_kg" not in fg:
        fg["export_price_usd_kg"] = np.nan

    out = Path("data/processed/farmgate_weekly.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fg.to_csv(out, index=False)
    print(
        f"saved {len(fg):,} farm-gate rows -> {out}; "
        f"countries={fg['country_code'].nunique()} "
        f"species={fg['species'].nunique()} sizes={fg['size'].nunique()} "
        f"weeks={fg['date'].nunique()}"
    )


if __name__ == "__main__":
    main()
