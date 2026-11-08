"""The same origin-aware feature builder is used in backtesting and serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shrimp_forecast.data import FREQUENCIES

CATEGORICAL = ["country", "species", "size"]
NUMERIC = ["log_level", "lag_return", "mean_return", "volatility", "season_sin", "season_cos", "gap"]
FEATURES = CATEGORICAL + NUMERIC


def offset(dataset: str):
    return pd.tseries.frequencies.to_offset(FREQUENCIES[dataset])


def next_dates(origin: pd.Timestamp, dataset: str, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(origin + offset(dataset), periods=horizon, freq=FREQUENCIES[dataset])


def row_features(
    group: pd.DataFrame, target_date: pd.Timestamp, dataset: str, as_of: pd.Timestamp | None = None
) -> tuple[dict, float]:
    cutoff = as_of if as_of is not None else target_date - offset(dataset)
    known = group[group.date.le(cutoff) & group.available_at.le(cutoff)].sort_values("date")
    if known.empty:
        raise ValueError("No observations available at forecast origin")
    latest = known.iloc[-1]
    anchor = float(latest.price_usd_kg)
    # Calendar lags preserve gaps. No interpolation or compression of missing periods.
    grid = pd.date_range(end=cutoff, periods=4, freq=FREQUENCIES[dataset])
    lags = known.set_index("date").price_usd_kg.reindex(grid)
    period = target_date.month / 12 if dataset == "benchmark" else target_date.dayofyear / 365.25
    values = lags.dropna().to_numpy(dtype=float)
    lag_return = np.log(lags.iloc[-1] / lags.iloc[-2]) if lags.iloc[-2:].notna().all() else np.nan
    return {
        "country": latest.country,
        "species": latest.species,
        "size": latest["size"],
        "log_level": np.log(anchor),
        "lag_return": lag_return,
        "mean_return": np.log(float(values.mean()) / anchor) if len(values) else np.nan,
        "volatility": float(np.std(np.log(values))) if len(values) else np.nan,
        "season_sin": np.sin(2 * np.pi * period),
        "season_cos": np.cos(2 * np.pi * period),
        "gap": len(pd.date_range(latest.date, cutoff, freq=FREQUENCIES[dataset])) - 1,
    }, anchor


def training_matrix(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows, targets = [], []
    for _, group in df.groupby("series_id", sort=True):
        group = group.sort_values("date")
        for observation in group.itertuples(index=False):
            cutoff = observation.date - offset(dataset)
            known = group[group.date.le(cutoff) & group.available_at.le(cutoff)]
            if len(known) < 4:
                continue
            features, anchor = row_features(known, observation.date, dataset)
            rows.append(features)
            targets.append(np.log(observation.price_usd_kg / anchor))
    if not rows:
        raise ValueError("Need at least five released observations per trainable series")
    return pd.DataFrame(rows)[FEATURES], np.array(targets)
