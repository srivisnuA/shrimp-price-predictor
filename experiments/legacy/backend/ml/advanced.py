"""Advanced multi-factor shrimp price forecasting.

Uses weather + disease + production as exogenous features, automatic model
selection via rolling-origin backtest, and scenario-based future projection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

import data_store

FEATURES = [
    "trend", "temp_anom", "rain_anom", "disease", "disease_lag1",
    "production", "temp_x_disease",
]
MODEL_FACTORIES = {
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=400, random_state=42),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=300, random_state=42),
    "ridge": lambda: Ridge(alpha=1.0),
}
SCENARIO_RANGES = {
    "avg_temp_c": (15.0, 40.0),
    "rainfall_mm": (0.0, 5000.0),
    "disease_outbreak_severity": (0.0, 100.0),
    "global_production_tonnes": (0.0, 50_000_000.0),
}


def _clamp(value: float, key: str) -> float:
    lo, hi = SCENARIO_RANGES[key]
    return float(min(max(value, lo), hi))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer model features from a yearly dataframe (sorted by year)."""
    df = df.sort_values("year").reset_index(drop=True).copy()
    df["trend"] = df["year"] - df["year"].min()
    temp = pd.to_numeric(df["avg_temp_c"], errors="coerce")
    rain = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    prod = pd.to_numeric(df["global_production_tonnes"], errors="coerce")
    df["temp_anom"] = temp - temp.rolling(5, min_periods=1).mean()
    df["rain_anom"] = rain - rain.rolling(5, min_periods=1).mean()
    sev = pd.to_numeric(df["disease_outbreak_severity"], errors="coerce").fillna(0.0)
    df["disease"] = sev
    df["disease_lag1"] = sev.shift(1).fillna(sev)
    df["production"] = prod
    df["temp_x_disease"] = df["temp_anom"].fillna(0.0) * sev
    return df


def _project_future_drivers(df: pd.DataFrame, horizon: int, scenario: dict) -> pd.DataFrame:
    """Build future driver rows: scenario values where given, else trend projection."""
    rows = []
    last_year = int(df["year"].max())
    temp = pd.to_numeric(df["avg_temp_c"], errors="coerce")
    rain = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    prod = pd.to_numeric(df["global_production_tonnes"], errors="coerce")
    sev = pd.to_numeric(df["disease_outbreak_severity"], errors="coerce")
    temp_trend = float(np.polyfit(range(len(temp.dropna())), temp.dropna(), 1)[0]) if temp.notna().sum() >= 3 else 0.0
    for h in range(1, horizon + 1):
        yr = last_year + h
        rows.append({
            "year": yr,
            "avg_temp_c": _clamp(scenario.get("avg_temp_c", temp.tail(3).mean() + temp_trend * h), "avg_temp_c") if temp.notna().any() else _clamp(scenario.get("avg_temp_c", 28.0), "avg_temp_c"),
            "rainfall_mm": _clamp(scenario.get("rainfall_mm", rain.tail(3).mean()), "rainfall_mm") if rain.notna().any() else _clamp(scenario.get("rainfall_mm", 1450.0), "rainfall_mm"),
            "disease_outbreak_severity": _clamp(scenario.get("disease_outbreak_severity", sev.tail(3).mean()), "disease_outbreak_severity") if sev.notna().any() else _clamp(scenario.get("disease_outbreak_severity", 30.0), "disease_outbreak_severity"),
            "global_production_tonnes": _clamp(scenario.get("global_production_tonnes", prod.tail(3).mean() * (1 + 0.02) ** h) if prod.notna().sum() >= 3 else prod.dropna().iloc[-1] if prod.notna().any() else 5_000_000.0, "global_production_tonnes"),
        })
    return pd.DataFrame(rows)


def _fit_model(X: pd.DataFrame, y: pd.Series, name: str):
    model = MODEL_FACTORIES[name]()
    model.fit(X, y)
    return model


def backtest(df: pd.DataFrame) -> tuple[str, dict]:
    """Rolling-origin expanding-window backtest; pick the best model by MAE."""
    feats = build_features(df)
    usable = feats.dropna(subset=["avg_price_usd_kg"])
    n = len(usable)
    if n < 8:
        raise ValueError("Not enough yearly data to train the advanced model (need at least 8 years)")
    start = max(6, n - 8)
    results = {}
    for name in MODEL_FACTORIES:
        maes, rmses, base = [], [], []
        for i in range(start, n):
            train, actual = usable.iloc[:i], usable.iloc[i]
            if len(train) < 5:
                continue
            try:
                model = _fit_model(train[FEATURES], train["avg_price_usd_kg"], name)
                p = float(model.predict(pd.DataFrame([train[FEATURES].iloc[-1].to_dict()]))[0])
            except Exception:
                continue
            p = max(p, 0.0)
            maes.append(abs(p - actual["avg_price_usd_kg"]))
            rmses.append((p - actual["avg_price_usd_kg"]) ** 2)
            base.append(abs(train["avg_price_usd_kg"].iloc[-1] - actual["avg_price_usd_kg"]))
        if maes:
            results[name] = {
                "mae": float(np.mean(maes)),
                "rmse": float(np.sqrt(np.mean(rmses))),
                "baseline_mae": float(np.mean(base)),
            }
    if not results:
        raise ValueError("Backtest failed for all candidate models")
    best = min(results, key=lambda k: results[k]["mae"])
    return best, results[best]


def train_and_forecast(df: pd.DataFrame | None = None, horizon: int = 5, scenario: dict | None = None) -> dict:
    df = data_store.load_yearly() if df is None else df
    if len(df) < 8:
        raise ValueError("Not enough yearly data to train the advanced model (need at least 8 years)")
    scenario = scenario or {}
    best_name, bt = backtest(df)
    feats = build_features(df)
    usable = feats.dropna(subset=["avg_price_usd_kg"])
    model = _fit_model(usable[FEATURES], usable["avg_price_usd_kg"], best_name)

    future = _project_future_drivers(df, horizon, scenario)
    # disease: use the user's severity for ALL future years so scenarios are stable
    if "disease_outbreak_severity" in scenario:
        sev = _clamp(float(scenario["disease_outbreak_severity"]), "disease_outbreak_severity")
        future["disease_outbreak_severity"] = sev
    future_feats = build_features(pd.concat([df, future], ignore_index=True)).tail(horizon)

    # residual sigma for the confidence band
    fitted = model.predict(usable[FEATURES])
    sigma = max(float(np.std(usable["avg_price_usd_kg"].values - fitted)), 0.05)

    preds = []
    for i, row in future_feats.reset_index(drop=True).iterrows():
        p = max(float(model.predict(pd.DataFrame([row[FEATURES].to_dict()]))[0]), 0.01)
        spread = 1.96 * sigma * np.sqrt(i + 1)
        preds.append({
            "year": int(row["year"]),
            "prediction": round(p, 3),
            "lower": round(max(p - spread, 0.0), 3),
            "upper": round(p + spread, 3),
        })

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = {f: round(float(v), 4) for f, v in sorted(
            zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1])}
    elif hasattr(model, "coef_"):
        importances = {f: round(float(abs(v)), 4) for f, v in sorted(
            zip(FEATURES, model.coef_), key=lambda kv: -kv[1])}

    return {
        "model": best_name,
        "horizon_years": horizon,
        "scenario": {k: _clamp(v, k) for k, v in scenario.items() if k in SCENARIO_RANGES},
        "predictions": preds,
        "feature_importances": importances,
        "metrics": {
            "mae": round(bt["mae"], 3),
            "rmse": round(bt["rmse"], 3),
            "baseline_mae": round(bt["baseline_mae"], 3),
        },
        "history": [
            {"year": int(r["year"]), "price": float(r["avg_price_usd_kg"]),
             "disease": float(r["disease_outbreak_severity"]) if pd.notna(r["disease_outbreak_severity"]) else None}
            for _, r in df.iterrows()
        ],
    }
