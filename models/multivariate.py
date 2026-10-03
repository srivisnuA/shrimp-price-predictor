"""Leakage-aware multivariate shrimp-price forecasting.

The model uses lagged price, weather and disease variables. Future exogenous variables
are supplied as scenarios; when absent, seasonal historical medians are used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    "month", "price_usd_kg_lag1", "price_usd_kg_lag3", "price_usd_kg_roll3",
    "t2m", "t2m_lag1", "t2m_roll3", "prectotcorr", "prectotcorr_lag1",
    "prectotcorr_roll3", "disease_event_count", "disease_severity_lag1", "disease_severity_roll3"
]

MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.5, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=2, loss="huber", random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "date"]).copy()
    for c in FEATURES:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    # Exogenous missing values are imputed from region-month historical medians.
    for c in FEATURES:
        if c == "month" or c not in df: continue
        df[c] = df.groupby(["region", "month"])[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())
    return df.dropna(subset=["price_usd_kg"] + FEATURES).reset_index(drop=True)


def walk_forward(df: pd.DataFrame, min_train=60, horizon=1):
    df = prepare(df)
    dates = sorted(df["date"].unique())
    scores = {name: [] for name in MODELS}
    for i in range(min_train, len(dates) - horizon + 1):
        train_dates = dates[:i]
        test_date = dates[i]
        train = df[df["date"].isin(train_dates)]
        test = df[df["date"] == test_date]
        if test.empty: continue
        for name, factory in MODELS.items():
            model = factory()
            model.fit(train[FEATURES], train["price_usd_kg"])
            pred = np.maximum(model.predict(test[FEATURES]), 0.01)
            scores[name].append({"mae": mean_absolute_error(test["price_usd_kg"], pred), "rmse": mean_squared_error(test["price_usd_kg"], pred) ** 0.5})
    summary = {}
    for name, rows in scores.items():
        if rows:
            summary[name] = {"mae": float(np.mean([r["mae"] for r in rows])), "rmse": float(np.mean([r["rmse"] for r in rows])), "folds": len(rows)}
    if not summary: raise ValueError("Not enough monthly observations for walk-forward validation")
    return min(summary, key=lambda k: summary[k]["mae"]), summary


def fit(df: pd.DataFrame):
    df = prepare(df)
    best, scores = walk_forward(df)
    model = MODELS[best]()
    model.fit(df[FEATURES], df["price_usd_kg"])
    return model, best, scores, df
