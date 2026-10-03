"""CSV-backed data store for shrimp price history."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("shrimp.data")

REQUIRED_COLUMNS = ["year", "month", "region", "size_class", "price_usd_kg"]

DATA_PATH = Path(__file__).parent / "data" / "shrimp_prices.csv"
SAMPLE_PATH = Path(__file__).parent / "data" / "shrimp_prices_sample.csv"


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a price dataframe. Raises ValueError on schema problems."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["price_usd_kg"] = pd.to_numeric(df["price_usd_kg"], errors="coerce")
    for col in ("region", "size_class"):
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna(subset=["year", "month", "price_usd_kg"])
    df = df[(df["month"].between(1, 12)) & (df["price_usd_kg"] > 0)]
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df.reset_index(drop=True)


def load(include_sample_fallback: bool = True) -> pd.DataFrame:
    """Load the working CSV; fall back to the bundled sample if no working file exists."""
    path = DATA_PATH if DATA_PATH.exists() else SAMPLE_PATH
    df = pd.read_csv(path)
    n0 = len(df)
    df = _validate(df)
    dropped = n0 - len(df)
    if dropped:
        logger.warning("Skipped %d malformed rows in %s", dropped, path.name)
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


def replace_with_upload(df: pd.DataFrame) -> int:
    """Atomically replace the working dataset with validated uploaded data."""
    clean = _validate(df)
    if clean.empty:
        raise ValueError("No valid rows found in the uploaded CSV")
    tmp = DATA_PATH.with_suffix(".tmp")
    clean.to_csv(tmp, index=False)
    tmp.replace(DATA_PATH)
    return len(clean)


def records(df: pd.DataFrame | None = None) -> list[dict]:
    if df is None:
        df = load()
    return df.to_dict(orient="records")


def dimensions() -> dict:
    df = load()
    return {
        "regions": sorted(df["region"].unique().tolist()),
        "size_classes": sorted(df["size_class"].unique().tolist()),
    }


# ---------------- Yearly dataset (with weather + disease drivers) ----------------

YEARLY_COLUMNS = [
    "year", "avg_price_usd_kg", "avg_temp_c", "rainfall_mm",
    "disease_outbreak_severity", "disease_name", "global_production_tonnes",
]
YEARLY_REQUIRED = ["year", "avg_price_usd_kg"]
YEARLY_PATH = Path(__file__).parent / "data" / "shrimp_prices_yearly.csv"


def _validate_yearly(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    missing = [c for c in YEARLY_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["avg_price_usd_kg"] = pd.to_numeric(df["avg_price_usd_kg"], errors="coerce")
    df = df.dropna(subset=["year", "avg_price_usd_kg"])
    df = df[(df["year"] >= 1980) & (df["year"] <= 2100) & (df["avg_price_usd_kg"] > 0)]
    df["year"] = df["year"].astype(int)
    # optional driver columns — fill sane defaults, clamp
    if "avg_temp_c" not in df.columns:
        df["avg_temp_c"] = None
    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = None
    if "disease_outbreak_severity" not in df.columns:
        df["disease_outbreak_severity"] = None
    if "disease_name" not in df.columns:
        df["disease_name"] = "none"
    if "global_production_tonnes" not in df.columns:
        df["global_production_tonnes"] = None
    for col in ("avg_temp_c", "rainfall_mm", "disease_outbreak_severity", "global_production_tonnes"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["disease_name"] = df["disease_name"].fillna("none").astype(str).str.strip().str.lower()
    for col in ("avg_temp_c", "rainfall_mm", "disease_outbreak_severity", "global_production_tonnes"):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).where(df[col].notna(), None)
    df = df.drop_duplicates(subset="year", keep="last")
    return df[YEARLY_COLUMNS].sort_values("year").reset_index(drop=True)


def load_yearly() -> pd.DataFrame:
    if not YEARLY_PATH.exists():
        return _validate_yearly(pd.DataFrame(columns=YEARLY_COLUMNS))
    n0 = sum(1 for _ in open(YEARLY_PATH)) - 1
    df = _validate_yearly(pd.read_csv(YEARLY_PATH))
    dropped = n0 - len(df)
    if dropped:
        logger.warning("Skipped %d malformed rows in %s", dropped, YEARLY_PATH.name)
    return df


def save_yearly(df: pd.DataFrame) -> None:
    YEARLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if YEARLY_PATH.exists():
        df_prev = YEARLY_PATH
        df_prev.rename(YEARLY_PATH.with_suffix(".csv.bak"))
    df.to_csv(YEARLY_PATH, index=False)


def merge_yearly(new_df: pd.DataFrame, mode: str = "merge") -> tuple[int, int]:
    """Merge validated yearly rows into the dataset. Returns (added, updated).

    In merge mode, driver columns missing from the upload preserve the
    existing row's values (an upload with only price columns never wipes
    weather/disease data). In replace mode, missing driver columns are blank.
    """
    clean = _validate_yearly(new_df)
    if clean.empty:
        raise ValueError("No valid rows found in the uploaded CSV")
    if mode == "replace":
        save_yearly(clean)
        return len(clean), 0
    existing = load_yearly()
    uploaded_years = set(clean["year"])
    old = existing[existing["year"].isin(uploaded_years)].set_index("year")
    driver_cols = YEARLY_COLUMNS[2:]
    for col in driver_cols:
        if col not in new_df.columns or pd.isna(new_df.get(col, pd.Series(dtype=float))).all():
            # upload didn't carry this column — backfill merged rows from existing
            clean = clean.set_index("year")
            clean.loc[clean.index.isin(old.index), col] = old.loc[
                old.index.isin(clean.index), col].values
            clean = clean.reset_index()
    combined = pd.concat([existing, clean], ignore_index=True)
    combined = combined.drop_duplicates(subset="year", keep="last").sort_values("year").reset_index(drop=True)
    added = len(combined) - len(existing)
    updated = len(clean) - added
    save_yearly(combined)
    return added, updated


def yearly_records(df: pd.DataFrame | None = None, year_from: int | None = None, year_to: int | None = None) -> list[dict]:
    if df is None:
        df = load_yearly()
    if year_from is not None:
        df = df[df["year"] >= year_from]
    if year_to is not None:
        df = df[df["year"] <= year_to]
    return df.to_dict(orient="records")
