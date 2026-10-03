"""Leakage-aware multivariate shrimp-price forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "target_price_usd_kg"
FEATURES = [
    "month", "target_price_usd_kg_lag1", "target_price_usd_kg_lag3", "target_price_usd_kg_roll3",
    "price_usd_kg_lag1", "t2m", "t2m_lag1", "t2m_roll3", "prectotcorr",
    "prectotcorr_lag1", "prectotcorr_roll3", "disease_event_count",
    "disease_severity_lag1", "disease_severity_roll3"
]

MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.5, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=2, loss="huber", random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1),
}

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["region", "date"]).copy()
    for c in FEATURES + [TARGET]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in FEATURES:
        if c == "month" or c not in df:
            continue
        df[c] = df.groupby(["region", "month"])[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())
    return df.dropna(subset=[TARGET] + FEATURES).reset_index(drop=True)

def walk_forward(df: pd.DataFrame, min_train: int = 60, horizon: int = 1):
    df = prepare(df)
    scores = {name: [] for name in MODELS}
    for _, region_df in df.groupby("region"):
        dates = sorted(region_df["date"].unique())
        if len(dates) <= min_train:
            continue
        for i in range(min_train, len(dates) - horizon + 1):
            train = region_df[region_df["date"].isin(dates[:i])]
            test = region_df[region_df["date"] == dates[i]]
            if test.empty:
                continue
            for name, factory in MODELS.items():
                model = factory()
                model.fit(train[FEATURES], train[TARGET])
                pred = np.maximum(model.predict(test[FEATURES]), 0.01)
                scores[name].append({
                    "mae": mean_absolute_error(test[TARGET], pred),
                    "rmse": mean_squared_error(test[TARGET], pred) ** 0.5
                })
    summary = {}
    for name, rows in scores.items():
        if rows:
            summary[name] = {
                "mae": float(np.mean([r["mae"] for r in rows])),
                "rmse": float(np.mean([r["rmse"] for r in rows])),
                "folds": len(rows)
            }
    if not summary:
        raise ValueError("Not enough monthly observations for walk-forward validation")
    return min(summary, key=lambda k: summary[k]["mae"]), summary

def fit_region(df: pd.DataFrame, region: str):
    clean = prepare(df)
    region_df = clean[clean["region"] == region].copy()
    if len(region_df) < 24:
        raise ValueError(f"Not enough observations for region={region}; need at least 24 months")
    best, scores = walk_forward(region_df, min_train=max(24, min(60, len(region_df) // 2)))
    model = MODELS[best]()
    model.fit(region_df[FEATURES], region_df[TARGET])
    return model, best, scores, region_df

def forecast_region(df: pd.DataFrame, region: str, horizon: int = 12, scenario: dict | None = None):
    model, best, scores, history = fit_region(df, region)
    scenario = scenario or {}
    last = history.sort_values("date").iloc[-1]
    rows = []
    state = history.sort_values("date").copy()
    for h in range(1, horizon + 1):
        date = last["date"] + pd.offsets.MonthBegin(h)
        month = date.month
        prev = state.iloc[-1]
        seasonal = history[history["month"] == month]
        row = {
            "region": region,
            "date": date,
            "month": month,
            "target_price_usd_kg_lag1": float(prev[TARGET]),
            "target_price_usd_kg_lag3": float(state[TARGET].tail(3).iloc[0]) if len(state) >= 3 else float(prev[TARGET]),
            "target_price_usd_kg_roll3": float(state[TARGET].tail(3).mean()),
            "price_usd_kg_lag1": float(prev.get("price_usd_kg", prev[TARGET])),
            "t2m": float(scenario.get("temperature_c", seasonal["t2m"].median())),
            "t2m_lag1": float(prev.get("t2m", seasonal["t2m"].median())),
            "t2m_roll3": float(state["t2m"].tail(3).mean()),
            "prectotcorr": float(scenario.get("rainfall_mm", seasonal["prectotcorr"].median())),
            "prectotcorr_lag1": float(prev.get("prectotcorr", seasonal["prectotcorr"].median())),
            "prectotcorr_roll3": float(state["prectotcorr"].tail(3).mean()),
            "disease_event_count": float(scenario.get("disease_events", seasonal["disease_event_count"].median())),
            "disease_severity_lag1": float(prev.get("disease_severity", 0)),
            "disease_severity_roll3": float(state["disease_severity"].tail(3).mean()) if "disease_severity" in state else 0.0,
        }
        X = pd.DataFrame([row])[FEATURES]
        prediction = max(float(model.predict(X)[0]), 0.01)
        row[TARGET] = prediction
        row["prediction"] = round(prediction, 3)
        row["year"] = int(date.year)
        rows.append(row)
        state = pd.concat([state, pd.DataFrame([row])], ignore_index=True)
    return {
        "region": region,
        "model": best,
        "horizon_months": horizon,
        "scenario": scenario,
        "metrics": scores[best],
        "predictions": [{k: r[k] for k in ["year", "date", "prediction"]} for r in rows]
    }
