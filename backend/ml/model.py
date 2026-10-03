"""Forecasting models for shrimp prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))


def build_features(series: pd.Series) -> pd.DataFrame:
    """Engineer features from a monthly price series indexed by datetime."""
    df = pd.DataFrame({"price": series})
    df["t"] = np.arange(len(df))
    df["month"] = df.index.month
    df["lag1"] = df["price"].shift(1)
    df["lag12"] = df["price"].shift(12)
    df["roll3"] = df["price"].shift(1).rolling(3).mean()
    return df


def fit_series(series: pd.Series):
    """Fit gradient boosting (with lag features) on a datetime-indexed price series.

    Falls back to linear regression on time index when there is too little data
    for lags. Returns (kind, model, feature_frame).
    """
    series = series.astype(float).sort_index()
    if len(series) < 4:
        raise ValueError("Not enough data points to train a model (need at least 4)")
    feats = build_features(series)
    if len(series) >= 18:
        train = feats.dropna(subset=["lag1", "lag12", "roll3"])
        if len(train) >= 6:
            X = train[["t", "month", "lag1", "lag12", "roll3"]]
            y = train["price"]
            model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
            model.fit(X, y)
            return "gb", model, feats
    # fallback: linear trend on t + month
    X = feats[["t", "month"]]
    y = feats["price"]
    model = LinearRegression()
    model.fit(X, y)
    return "lin", model, feats


def _next_features(row: dict, t: int, month: int) -> list[float]:
    return [t, month, row["price"], row.get("lag12", row["price"]), row["roll3"] if pd.notna(row.get("roll3")) else row["price"]]


def forecast(series: pd.Series, horizon: int) -> list[dict]:
    """Forecast `horizon` future months. Returns list of dicts with date/prediction/lower/upper."""
    series = series.astype(float).sort_index()
    kind, model, feats = fit_series(series)
    history = list(series.values)
    if kind == "gb":
        # residual sigma from in-sample fit on rows with complete lag features
        train = feats.dropna(subset=["lag1", "lag12", "roll3"])
        cols = ["t", "month", "lag1", "lag12", "roll3"]
        fitted = model.predict(train[cols])
        resid_sigma = float(np.std(train["price"].values - fitted))
    else:
        fitted = model.predict(feats[["t", "month"]])
        resid_sigma = float(np.std(series.values - fitted))
    resid_sigma = max(resid_sigma, 0.05)

    preds = []
    state_lag1 = history[-1]
    state_lag12 = history[-12] if len(history) >= 12 else history[0]
    state_roll3 = float(np.mean(history[-3:]))
    last_t = int(feats["t"].iloc[-1])
    out = []
    for h in range(1, horizon + 1):
        next_date = series.index[-1] + pd.offsets.MonthBegin(h)
        t = last_t + h
        month = next_date.month
        if kind == "gb":
            X = pd.DataFrame([[t, month, state_lag1, state_lag12, state_roll3]],
                             columns=["t", "month", "lag1", "lag12", "roll3"])
            p = float(model.predict(X)[0])
        else:
            X = pd.DataFrame([[t, month]], columns=["t", "month"])
            p = float(model.predict(X)[0])
        p = max(p, 0.01)  # prices cannot be negative
        spread = 1.96 * resid_sigma * np.sqrt(h)
        out.append({
            "date": next_date.strftime("%Y-%m"),
            "prediction": round(p, 3),
            "lower": round(max(p - spread, 0.0), 3),
            "upper": round(p + spread, 3),
        })
        # roll state forward
        history.append(p)
        state_lag1 = p
        state_lag12 = history[-12] if len(history) >= 12 else history[0]
        state_roll3 = float(np.mean(history[-3:]))
    return out


def evaluate(series: pd.Series) -> dict:
    """Rolling holdout backtest: train on all-but-last-k, predict one step ahead.

    Compares model MAE/RMSE against a naive last-value baseline.
    """
    series = series.astype(float).sort_index()
    n = len(series)
    if n < 12:
        return {"mae": None, "rmse": None, "baseline_mae": None, "n_folds": 0}

    def one_step(hist: pd.Series, month: int) -> float:
        kind, model, feats = fit_series(hist)
        if kind == "gb":
            row = feats.dropna(subset=["lag1", "lag12", "roll3"]).iloc[-1]
            X = pd.DataFrame(
                [[row["t"] + 1, month, row["price"], row["lag12"], row["roll3"]]],
                columns=["t", "month", "lag1", "lag12", "roll3"])
            return float(model.predict(X)[0])
        X = pd.DataFrame([[feats["t"].iloc[-1] + 1, month]], columns=["t", "month"])
        return float(model.predict(X)[0])

    mae_scores, rmse_scores, base_scores = [], [], []
    k = min(8, n - 12)
    for i in range(n - k, n):
        hist = series.iloc[:i]
        month = series.index[i].month
        try:
            p = one_step(hist, month)
        except Exception:
            continue
        actual = series.iloc[i]
        mae_scores.append(abs(p - actual))
        rmse_scores.append((p - actual) ** 2)
        base_scores.append(abs(series.iloc[i - 1] - actual))
    if not mae_scores:
        return {"mae": None, "rmse": None, "baseline_mae": None, "n_folds": 0}
    return {
        "mae": round(float(np.mean(mae_scores)), 3),
        "rmse": round(float(np.sqrt(np.mean(rmse_scores))), 3),
        "baseline_mae": round(float(np.mean(base_scores)), 3),
        "n_folds": len(mae_scores),
    }
