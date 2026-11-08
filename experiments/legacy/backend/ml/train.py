"""Train the model and persist metrics for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import data_store
from ml.model import evaluate

METRICS_PATH = Path(__file__).parent / "metrics.json"


def default_series() -> pd.Series:
    df = data_store.load()
    df = df[(df["region"] == "Global") & (df["size_class"] == "Medium")]
    idx = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return pd.Series(df["price_usd_kg"].values, index=idx, name="price")


def train_and_save_metrics() -> dict:
    metrics = evaluate(default_series())
    METRICS_PATH.write_text(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_save_metrics(), indent=2))
