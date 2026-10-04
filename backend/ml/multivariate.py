"""Leakage-aware multivariate shrimp-price forecasting with export/farm-gate targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET_COLUMNS = {
    "export": "export_price_usd_kg",
    "farm_gate": "farm_gate_price_usd_kg",
}

COMMON_FEATURES = [
    "month",
    "price_usd_kg_lag1",
    "t2m",
    "t2m_lag1",
    "t2m_roll3",
    "prectotcorr",
    "prectotcorr_lag1",
    "prectotcorr_roll3",
    "shrimp_production_tonnes",
    "shrimp_production_tonnes_lag1",
    "usd_inr",
    "wti_usd_bbl",
    "us_cpi",
    "disease_event_count",
    "disease_severity_lag1",
    "disease_severity_roll3",
]

MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}


def feature_columns(price_type: str) -> tuple[str, list[str]]:
    if price_type not in TARGET_COLUMNS:
        raise ValueError("price_type must be 'export' or 'farm_gate'")
    target = TARGET_COLUMNS[price_type]
    features = [
        "month",
        f"{target}_lag1",
        f"{target}_lag3",
        f"{target}_roll3",
        *COMMON_FEATURES[1:],
    ]
    # Farm-gate prices are directly linked to what processors/exporters can
    # pay, so the most recent export market level is a legitimate lagged input.
    if price_type == "farm_gate":
        features.insert(4, "export_price_usd_kg_lag1")
    return target, features


def prepare(df: pd.DataFrame, price_type: str = "export") -> pd.DataFrame:
    target, features = feature_columns(price_type)
    df = df.sort_values(["region", "date"]).copy()
    for c in features + [target]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in features:
        if c == "month" or c not in df:
            continue
        df[c] = df.groupby(["region", "month"])[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())

    return df.dropna(subset=[target] + features).reset_index(drop=True)


def walk_forward(
    df: pd.DataFrame,
    price_type: str = "export",
    min_train: int | None = None,
    horizon: int = 1,
):
    clean = prepare(df, price_type)
    scores = {name: [] for name in MODELS}

    for _, region_df in clean.groupby("region"):
        dates = sorted(region_df["date"].unique())
        if min_train is None:
            if price_type == "farm_gate":
                min_train = max(6, min(12, len(dates) // 2))
            else:
                min_train = 24
        if len(dates) <= min_train:
            continue

        for i in range(min_train, len(dates) - horizon + 1):
            train = region_df[region_df["date"].isin(dates[:i])]
            test = region_df[region_df["date"] == dates[i]]
            if test.empty:
                continue

            target, features = feature_columns(price_type)
            for name, factory in MODELS.items():
                model = factory()
                model.fit(train[features], train[target])
                pred = np.maximum(model.predict(test[features]), 0.01)
                scores[name].append(
                    {
                        "mae": mean_absolute_error(test[target], pred),
                        "rmse": mean_squared_error(test[target], pred) ** 0.5,
                    }
                )

    summary = {}
    for name, rows in scores.items():
        if rows:
            summary[name] = {
                "mae": float(np.mean([r["mae"] for r in rows])),
                "rmse": float(np.mean([r["rmse"] for r in rows])),
                "folds": len(rows),
            }

    if not summary:
        raise ValueError(
            f"Not enough {price_type.replace('_', ' ')} observations for walk-forward validation"
        )

    return min(summary, key=lambda k: summary[k]["mae"]), summary


def fit_region(df: pd.DataFrame, region: str, price_type: str = "export"):
    clean = prepare(df, price_type)
    region_df = clean[clean["region"] == region].copy()
    min_obs = 8 if price_type == "farm_gate" else 24
    if len(region_df) < min_obs:
        raise ValueError(
            f"Not enough observations for region={region}, price_type={price_type}; "
            f"need at least {min_obs} months with real target prices"
        )

    if price_type == "farm_gate":
        min_train = max(6, min(12, len(region_df) // 2))
    else:
        min_train = max(24, min(60, len(region_df) // 2))

    best, scores = walk_forward(region_df, price_type=price_type, min_train=min_train)
    target, features = feature_columns(price_type)
    model = MODELS[best]()
    model.fit(region_df[features], region_df[target])
    fitted = model.predict(region_df[features])
    sigma = max(float(np.std(region_df[target].to_numpy() - fitted)), 0.05)
    return model, best, scores, region_df, sigma


def forecast_region(
    df: pd.DataFrame,
    region: str,
    horizon: int = 12,
    scenario: dict | None = None,
    price_type: str = "export",
):
    target, features = feature_columns(price_type)
    model, best, scores, history, sigma = fit_region(df, region, price_type=price_type)
    scenario = scenario or {}
    last = history.sort_values("date").iloc[-1]
    rows = []
    state = history.sort_values("date").copy()

    for h in range(1, horizon + 1):
        date = last["date"] + pd.offsets.MonthBegin(h)
        month = date.month
        prev = state.iloc[-1]
        seasonal = history[history["month"] == month]

        if seasonal.empty:
            seasonal = history

        future_export = float(
            scenario.get("export_price_usd_kg", seasonal["export_price_usd_kg"].median())
        )
        row = {
            "region": region,
            "date": date,
            "month": month,
            "export_price_usd_kg": future_export,
            "farm_gate_price_usd_kg": np.nan,
            f"{target}_lag1": float(prev[target]),
            f"{target}_lag3": float(state[target].tail(3).iloc[0])
            if len(state) >= 3
            else float(prev[target]),
            f"{target}_roll3": float(state[target].tail(3).mean()),
            "price_usd_kg_lag1": float(prev.get("price_usd_kg", prev.get("export_price_usd_kg", prev[target]))),
            "export_price_usd_kg_lag1": float(
                prev.get("export_price_usd_kg", seasonal["export_price_usd_kg"].median())
            ),
            "t2m": float(scenario.get("temperature_c", seasonal["t2m"].median())),
            "t2m_lag1": float(prev.get("t2m", seasonal["t2m"].median())),
            "t2m_roll3": float(state["t2m"].tail(3).mean()),
            "prectotcorr": float(scenario.get("rainfall_mm", seasonal["prectotcorr"].median())),
            "prectotcorr_lag1": float(prev.get("prectotcorr", seasonal["prectotcorr"].median())),
            "prectotcorr_roll3": float(state["prectotcorr"].tail(3).mean()),
            "shrimp_production_tonnes": float(
                scenario.get("production_tonnes", state["shrimp_production_tonnes"].iloc[-1])
            ),
            "shrimp_production_tonnes_lag1": float(state["shrimp_production_tonnes"].iloc[-1]),
            "usd_inr": float(scenario.get("usd_inr", state["usd_inr"].iloc[-1])),
            "wti_usd_bbl": float(scenario.get("wti_usd_bbl", state["wti_usd_bbl"].iloc[-1])),
            "us_cpi": float(scenario.get("us_cpi", state["us_cpi"].iloc[-1])),
            "disease_event_count": float(
                scenario.get("disease_events", seasonal["disease_event_count"].median())
            ),
            "disease_severity_lag1": float(prev.get("disease_severity", 0)),
            "disease_severity_roll3": float(state["disease_severity"].tail(3).mean())
            if "disease_severity" in state
            else 0.0,
        }

        X = pd.DataFrame([row])[features]
        prediction = max(float(model.predict(X)[0]), 0.01)
        row[target] = prediction

        spread = 1.96 * sigma * np.sqrt(h)
        row["prediction"] = round(prediction, 3)
        row["lower"] = round(max(prediction - spread, 0.0), 3)
        row["upper"] = round(prediction + spread, 3)
        row["year"] = int(date.year)
        rows.append(row)
        state = pd.concat([state, pd.DataFrame([row])], ignore_index=True)

    history_months = int(history["date"].nunique())
    warning = None
    if price_type == "farm_gate" and history_months < 18:
        warning = (
            "Farm-gate history is currently limited. The target is real observed "
            "farm-gate pricing, but the uncertainty is higher than the export model."
        )

    return {
        "region": region,
        "price_type": price_type,
        "target_column": target,
        "model": best,
        "history_months": history_months,
        "warning": warning,
        "horizon_months": horizon,
        "scenario": scenario,
        "metrics": scores[best],
        "predictions": [
            {k: r[k] for k in ["year", "date", "prediction", "lower", "upper"]}
            for r in rows
        ],
    }
