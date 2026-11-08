"""Pooled farm-gate forecasting across every country, species and size in the source."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "price_usd_kg"

CATEGORICAL = ["country_code", "species"]
NUMERIC = [
    "size",
    "week_of_year",
    "month",
    "price_usd_kg_lag1",
    "price_usd_kg_lag2",
    "price_usd_kg_roll4",
    "t2m",
    "prectotcorr",
    "disease_event_count",
    "disease_severity",
    "shrimp_production_tonnes",
    "usd_inr",
    "wti_usd_bbl",
    "us_cpi",
    "global_price_usd_kg",
    "export_price_usd_kg",
]

MODELS = {
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.75,
        random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.035,
        max_depth=2,
        loss="huber",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=450,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["country_code", "species", "size", "date"])

    for c in CATEGORICAL:
        df[c] = df[c].astype(str)

    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(float)
    df["month"] = df["date"].dt.month.astype(float)

    key = ["country_code", "species", "size"]
    g = df.groupby(key, sort=False)[TARGET]
    df["price_usd_kg_lag1"] = g.shift(1)
    df["price_usd_kg_lag2"] = g.shift(2)
    df["price_usd_kg_roll4"] = g.transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    for c in NUMERIC:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["date", "country_code", "species", "size", TARGET]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame):
    cats = [c for c in CATEGORICAL if c in df.columns]
    nums = [c for c in NUMERIC if c in df.columns and df[c].notna().any()]
    return cats + nums


def make_pipeline(factory, train: pd.DataFrame):
    cats = [c for c in CATEGORICAL if c in train.columns]
    nums = [c for c in NUMERIC if c in train.columns and train[c].notna().any()]
    transformers = []
    if cats:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cats,
            )
        )
    if nums:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                nums,
            )
        )
    prep = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", factory())]), cats + nums


def walk_forward(df: pd.DataFrame, min_train_weeks: int = 5):
    clean = prepare(df)
    dates = sorted(clean["date"].unique())
    if len(dates) <= min_train_weeks:
        raise ValueError("Not enough farm-gate weekly dates for walk-forward validation")

    score_rows = {name: [] for name in MODELS}
    baseline_rows = []

    for i in range(min_train_weeks, len(dates)):
        train = clean[clean["date"].isin(dates[:i])]
        test = clean[clean["date"].eq(dates[i])]

        if train.empty or test.empty:
            continue

        baseline_pred = test["price_usd_kg_lag1"].fillna(train[TARGET].median())
        baseline_rows.append(
            {
                "mae": mean_absolute_error(test[TARGET], baseline_pred),
                "rmse": mean_squared_error(test[TARGET], baseline_pred) ** 0.5,
            }
        )

        for name, factory in MODELS.items():
            pipe, cols = make_pipeline(factory, train)
            pipe.fit(train[cols], train[TARGET])
            pred = np.maximum(pipe.predict(test[cols]), 0.01)
            score_rows[name].append(
                {
                    "mae": mean_absolute_error(test[TARGET], pred),
                    "rmse": mean_squared_error(test[TARGET], pred) ** 0.5,
                }
            )

    scores = {}
    for name, rows in score_rows.items():
        if rows:
            scores[name] = {
                "mae": float(np.mean([r["mae"] for r in rows])),
                "rmse": float(np.mean([r["rmse"] for r in rows])),
                "folds": len(rows),
            }

    baseline = {
        "mae": float(np.mean([r["mae"] for r in baseline_rows])),
        "rmse": float(np.mean([r["rmse"] for r in baseline_rows])),
        "folds": len(baseline_rows),
    }

    if not scores:
        raise ValueError("No farm-gate walk-forward folds were produced")

    candidates = {**scores, "latest_price_baseline": baseline}
    best = min(candidates, key=lambda name: candidates[name]["mae"])
    return best, scores, baseline


def _seasonal_context(history: pd.DataFrame, country_code: str, month: int, scenario: dict):
    subset = history[
        (history["country_code"].eq(country_code)) & (history["month"].eq(month))
    ]
    if subset.empty:
        subset = history[history["country_code"].eq(country_code)]
    if subset.empty:
        subset = history

    context = {}
    mapping = {
        "t2m": "temperature_c",
        "prectotcorr": "rainfall_mm",
        "disease_event_count": "disease_events",
        "disease_severity": "disease_severity",
        "shrimp_production_tonnes": "production_tonnes",
        "usd_inr": "usd_inr",
        "wti_usd_bbl": "wti_usd_bbl",
        "us_cpi": "us_cpi",
        "global_price_usd_kg": "global_price_usd_kg",
        "export_price_usd_kg": "export_price_usd_kg",
    }
    for col, scenario_key in mapping.items():
        value = scenario.get(scenario_key)
        if value is None and col in subset.columns:
            numeric = pd.to_numeric(subset[col], errors="coerce").dropna()
            value = float(numeric.median()) if not numeric.empty else np.nan
        context[col] = value
    return context


def forecast(
    df: pd.DataFrame,
    country_code: str,
    species: str,
    size: float,
    horizon_weeks: int = 26,
    scenario: dict | None = None,
):
    scenario = scenario or {}
    clean = prepare(df)

    selected = clean[
        clean["country_code"].eq(country_code)
        & clean["species"].eq(species)
        & clean["size"].eq(float(size))
    ].sort_values("date")

    if selected.empty:
        raise ValueError("No farm-gate data for the selected country/species/size")
    if len(selected) < 4:
        raise ValueError("Selected series has fewer than 4 weekly observations")

    best, scores, baseline = walk_forward(clean)
    model = None
    feature_cols = feature_columns(clean)

    if best != "latest_price_baseline":
        model, feature_cols = make_pipeline(MODELS[best], clean)
        model.fit(clean[feature_cols], clean[TARGET])

    if best == "latest_price_baseline":
        residuals = selected[TARGET] - selected["price_usd_kg_lag1"].fillna(selected[TARGET].median())
    else:
        fitted = model.predict(clean[feature_cols])
        residuals = clean[TARGET] - fitted
    sigma = max(float(np.std(residuals)), 0.05)

    state = selected.copy()
    last = selected.iloc[-1]
    predictions = []

    for step in range(1, horizon_weeks + 1):
        date = last["date"] + pd.Timedelta(weeks=step)
        month = int(date.month)
        week = int(date.isocalendar().week)

        prev1 = state.iloc[-1]
        prev2 = state.iloc[-2] if len(state) >= 2 else prev1
        roll4 = float(state[TARGET].tail(4).mean())

        row = {
            "country_code": country_code,
            "species": species,
            "size": float(size),
            "date": date,
            "week_of_year": week,
            "month": month,
            "price_usd_kg_lag1": float(prev1[TARGET]),
            "price_usd_kg_lag2": float(prev2[TARGET]),
            "price_usd_kg_roll4": roll4,
            **_seasonal_context(clean, country_code, month, scenario),
        }

        if best == "latest_price_baseline":
            prediction = float(prev1[TARGET])
        else:
            x = pd.DataFrame([row])
            prediction = float(model.predict(x[feature_cols])[0])

        prediction = max(prediction, 0.01)
        row[TARGET] = prediction
        state = pd.concat([state, pd.DataFrame([row])], ignore_index=True)

        spread = 1.96 * sigma * np.sqrt(step)
        predictions.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "week": week,
                "prediction": round(prediction, 3),
                "lower": round(max(prediction - spread, 0.0), 3),
                "upper": round(prediction + spread, 3),
            }
        )

    years = sorted(set(clean["date"].dt.year.astype(int)))
    return {
        "price_type": "farm_gate",
        "country_code": country_code,
        "species": species,
        "size": float(size),
        "history_weeks": int(len(selected)),
        "total_available_rows": int(len(clean)),
        "data_start": selected["date"].min().strftime("%Y-%m-%d"),
        "data_end": selected["date"].max().strftime("%Y-%m-%d"),
        "available_years": years,
        "model": best,
        "models": scores,
        "metrics": scores.get(best, baseline),
        "naive_baseline": baseline,
        "warning": (
            f"The uploaded farm-gate file currently contains {len(years)} calendar year(s) "
            f"and {len(clean['date'].unique())} weekly dates. All countries, species and sizes "
            "are retained, but the historical window is still short for a long-horizon farm-price model."
        ),
        "predictions": predictions,
    }


def history(df: pd.DataFrame, country_code: str, species: str, size: float):
    clean = prepare(df)
    out = clean[
        clean["country_code"].eq(country_code)
        & clean["species"].eq(species)
        & clean["size"].eq(float(size))
    ].sort_values("date")

    if out.empty:
        raise ValueError("No farm-gate history for the selected series")

    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "price": round(float(row[TARGET]), 3),
            "price_local": None if pd.isna(row.get("price_local_kg")) else round(float(row["price_local_kg"]), 0),
            "temperature": None if pd.isna(row.get("t2m")) else round(float(row["t2m"]), 2),
            "rainfall": None if pd.isna(row.get("prectotcorr")) else round(float(row["prectotcorr"]), 2),
            "disease": None if pd.isna(row.get("disease_severity")) else round(float(row["disease_severity"]), 2),
        }
        for _, row in out.iterrows()
    ]


def options(df: pd.DataFrame):
    clean = prepare(df)
    combos = (
        clean.groupby(["country_code", "country", "species", "size"], as_index=False)
        .agg(observations=(TARGET, "size"))
        .sort_values(["country", "species", "size"])
    )
    return [
        {
            "country_code": r["country_code"],
            "country": r["country"],
            "species": r["species"],
            "size": int(r["size"]),
            "observations": int(r["observations"]),
        }
        for _, r in combos.iterrows()
    ]
