"""Panel challengers, purged rolling origins, independent calibration and final audit.

No future target or contemporaneous unreleased driver enters a feature. A model
is selected on validation only, intervals are calibrated on the next block,
and a final untouched block measures performance without changing selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shrimp_forecast.data import fingerprint, quality, validate_observations
from shrimp_forecast.features import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    next_dates,
    offset,
    row_features,
    training_matrix,
)

BASELINES = ("latest_price", "seasonal_naive", "damped_drift")


@dataclass(frozen=True)
class TrainConfig:
    max_horizon: int = 4
    min_train_dates: int = 8
    coverage: float = 0.90
    min_improvement: float = 0.02
    seed: int = 42
    lightgbm: bool = False
    max_validation_origins: int = 6

    def __post_init__(self):
        if not 1 <= self.max_horizon <= 12 or self.min_train_dates < 6 or self.max_validation_origins < 1:
            raise ValueError("Invalid training horizon or minimum history")
        if not 0.5 <= self.coverage < 1 or not 0 <= self.min_improvement < 1:
            raise ValueError("Invalid coverage or improvement threshold")


def pipeline(name: str, config: TrainConfig) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            (
                "numbers",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
        ]
    )
    if name == "ridge":
        estimator = Ridge(alpha=20)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            learning_rate=0.05,
            l2_regularization=5,
            early_stopping=False,
            random_state=config.seed,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=15,
            reg_lambda=5,
            random_state=config.seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("features", preprocess), ("model", estimator)])


def fit(df: pd.DataFrame, dataset: str, name: str, config: TrainConfig):
    if name in BASELINES:
        return None
    features, targets = training_matrix(df, dataset)
    model = pipeline(name, config)
    model.fit(features, targets)
    return model


def predict_panel(
    df: pd.DataFrame, dataset: str, name: str, model, origin: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    known = df[df.date.le(origin) & df.available_at.le(origin)].copy()
    if known.empty:
        raise ValueError("No released history")
    state = {key: group.sort_values("date").copy() for key, group in known.groupby("series_id", sort=True)}
    rows = []
    for step, target_date in enumerate(next_dates(origin, dataset, horizon), start=1):
        ids, features, anchors, predictions = [], [], [], []
        for series_id, group in state.items():
            feature, anchor = row_features(group, target_date, dataset, as_of=origin if step == 1 else None)
            ids.append(series_id)
            features.append(feature)
            anchors.append(anchor)
            pred = anchor
            if name == "seasonal_naive":
                season = 12 if dataset == "benchmark" else 52
                seasonal_date = target_date - season * offset(dataset)
                seasonal = group[group.date.eq(seasonal_date)]
                if not seasonal.empty:
                    pred = float(seasonal.price_usd_kg.iloc[-1])
            elif name == "damped_drift":
                # Compute slope from observations available at the original cutoff only.
                original = group[group.date.le(origin)].tail(6)
                if len(original) > 1:
                    elapsed = max(
                        1,
                        len(
                            pd.date_range(
                                original.date.iloc[0],
                                original.date.iloc[-1],
                                freq="W-MON" if dataset == "farmgate" else "MS",
                            )
                        )
                        - 1,
                    )
                    slope = np.log(original.price_usd_kg.iloc[-1] / original.price_usd_kg.iloc[0]) / elapsed
                    pred = float(original.price_usd_kg.iloc[-1]) * np.exp(
                        np.clip(slope, -0.15, 0.15) * sum(0.8**j for j in range(step))
                    )
            predictions.append(pred)
        if name not in BASELINES:
            returns = model.predict(pd.DataFrame(features)[FEATURES])
            predictions = np.array(anchors) * np.exp(np.clip(returns, -0.5, 0.5))
        for series_id, prediction in zip(ids, predictions, strict=True):
            prediction = float(max(prediction, 0.001))
            rows.append(
                {"series_id": series_id, "date": target_date, "horizon": step, "prediction": prediction}
            )
            group = state[series_id]
            synthetic = group.iloc[-1].to_dict()
            synthetic.update(date=target_date, available_at=target_date, price_usd_kg=prediction)
            state[series_id] = pd.concat([group, pd.DataFrame([synthetic])], ignore_index=True)
    return pd.DataFrame(rows)


def metrics(rows: pd.DataFrame) -> dict:
    error = rows.prediction - rows.price_usd_kg
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "wape": float(error.abs().sum() / rows.price_usd_kg.abs().sum()),
        "bias": float(error.mean()),
        "observations": len(rows),
        "origins": int(rows.origin.nunique()),
        "macro_series_mae": float(rows.assign(error=error.abs()).groupby("series_id").error.mean().mean()),
    }


def splits(df: pd.DataFrame, config: TrainConfig) -> dict[str, list[pd.Timestamp]]:
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    # Last H targets calibrate, the next H targets audit, with no overlapping targets across stages.
    n, h = len(dates), config.max_horizon
    if n < config.min_train_dates + 3 * h:
        raise ValueError(
            f"Need at least {config.min_train_dates + 3 * h} distinct dates for independent validation, calibration and test"
        )
    validation = dates[config.min_train_dates - 1 : n - 3 * h]
    indices = np.unique(
        np.linspace(0, len(validation) - 1, min(len(validation), config.max_validation_origins), dtype=int)
    )
    return {
        "validation": list(validation[indices]),
        "calibration": [dates[n - 2 * h - 1]],
        "test": [dates[n - h - 1]],
    }


def backtest(
    df: pd.DataFrame, dataset: str, name: str, config: TrainConfig, origins: list[pd.Timestamp]
) -> pd.DataFrame:
    results = []
    for origin in origins:
        train = df[df.date.le(origin) & df.available_at.le(origin)]
        model = fit(train, dataset, name, config)
        predicted = predict_panel(train, dataset, name, model, origin, config.max_horizon)
        # An observation is a valid evaluation label only if eventually released in the input snapshot.
        joined = predicted.merge(
            df[["series_id", "date", "price_usd_kg"]],
            on=["series_id", "date"],
            how="inner",
            validate="one_to_one",
        )
        joined["origin"] = origin
        results.append(joined)
    out = pd.concat(results, ignore_index=True)
    if out.empty:
        raise ValueError(f"No matched backtest observations for {name}")
    return out


def calibrate(rows: pd.DataFrame, config: TrainConfig) -> dict[str, dict]:
    result = {}
    residual = np.abs(np.log(rows.price_usd_kg / rows.prediction))
    for horizon, block in rows.assign(residual=residual).groupby("horizon"):
        values = np.sort(block.residual.to_numpy())
        rank = int(np.ceil((len(values) + 1) * config.coverage))
        # Finite-sample correction may require an unbounded interval. Do not silently cap the rank.
        radius = float(values[rank - 1]) if rank <= len(values) else None
        result[str(int(horizon))] = {"log_radius": radius, "samples": len(values)}
    return result


def add_intervals(rows: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    out = rows.copy()
    radii = out.horizon.map(lambda h: calibration[str(int(h))]["log_radius"])
    out["lower"] = out.prediction * np.exp(-radii.astype(float))
    out["upper"] = out.prediction * np.exp(radii.astype(float))
    return out


def train(df: pd.DataFrame, dataset: str, config: TrainConfig | None = None) -> dict:
    config = config or TrainConfig()
    df = validate_observations(df, dataset)
    partitions = splits(df, config)
    names = [*BASELINES, "ridge", "hist_gradient_boosting"]
    if config.lightgbm:
        names.append("lightgbm")
    validation = {}
    for name in names:
        validation[name] = metrics(backtest(df, dataset, name, config, partitions["validation"]))
    best = min(names, key=lambda name: validation[name]["mae"])
    # A challenger must clear a modest improvement threshold to displace persistence.
    if validation[best]["mae"] >= validation["latest_price"]["mae"] * (1 - config.min_improvement):
        best = "latest_price"
    calibration_rows = backtest(df, dataset, best, config, partitions["calibration"])
    calibration = calibrate(calibration_rows, config)
    if len(calibration) != config.max_horizon:
        raise ValueError("Calibration requires matched observations at every served horizon")
    test_rows = add_intervals(backtest(df, dataset, best, config, partitions["test"]), calibration)
    baseline_test = backtest(df, dataset, "latest_price", config, partitions["test"])
    test_metrics = metrics(test_rows)
    finite = test_rows.lower.notna() & test_rows.upper.notna()
    test_metrics["interval_evaluated_observations"] = int(finite.sum())
    test_metrics["interval_coverage"] = (
        float(
            ((test_rows.price_usd_kg >= test_rows.lower) & (test_rows.price_usd_kg <= test_rows.upper))[
                finite
            ].mean()
        )
        if finite.any()
        else None
    )
    test_metrics["mean_interval_width"] = (
        float((test_rows.upper - test_rows.lower)[finite].mean()) if finite.any() else None
    )
    # All labels must be available at the training snapshot, including delayed releases.
    origin = max(df.date.max(), df.available_at.max())
    origin = pd.Timestamp(origin)
    model = fit(df, dataset, best, config)
    report = {
        "model": best,
        "config": asdict(config),
        "validation": validation,
        "test": test_metrics,
        "test_baseline": metrics(baseline_test),
        "calibration": calibration,
        "data": quality(df, dataset),
        "partitions": {stage: [str(t.date()) for t in origins] for stage, origins in partitions.items()},
        "per_series_test": {key: metrics(group) for key, group in test_rows.groupby("series_id")},
        "per_horizon_test": {str(int(h)): metrics(group) for h, group in test_rows.groupby("horizon")},
        "trained_at": datetime.now(UTC).isoformat(),
        "data_sha256": fingerprint(df),
        "limitations": [
            "Existing data is unverified; reported accuracy is a historical diagnostic.",
            "Calibration uses a single temporal origin with pooled series; correlated series and time drift invalidate exchangeability guarantees.",
            "The final test block measures the selected model and never changes model selection.",
            "Weather, disease, production and macro drivers are excluded until publication vintages and source quality are established.",
        ],
    }
    return {
        "model": model,
        "name": best,
        "history": df,
        "dataset": dataset,
        "config": config,
        "calibration": calibration,
        "report": report,
        "origin": origin,
    }


def forecast(bundle: dict, series_id: str, horizon: int) -> dict:
    config = bundle["config"]
    if not 1 <= horizon <= config.max_horizon:
        raise ValueError(
            f"Horizon must be between 1 and {config.max_horizon}; longer horizons are not calibrated"
        )
    history = bundle["history"]
    selected = history[history.series_id.eq(series_id)]
    if selected.empty:
        raise KeyError("Unknown series")
    # Models use only within-series target state; batching training does not require batching inference.
    rows = predict_panel(
        selected, bundle["dataset"], bundle["name"], bundle["model"], bundle["origin"], horizon
    )
    rows = add_intervals(rows, bundle["calibration"])
    predictions = []
    for row in rows.itertuples(index=False):
        predictions.append(
            {
                "date": str(row.date.date()),
                "horizon": int(row.horizon),
                "prediction": round(row.prediction, 4),
                "lower": round(row.lower, 4) if pd.notna(row.lower) else None,
                "upper": round(row.upper, 4) if pd.notna(row.upper) else None,
            }
        )
    return {
        "dataset": bundle["dataset"],
        "series_id": series_id,
        "model": bundle["name"],
        "forecast_origin": str(bundle["origin"].date()),
        "data_end": str(selected.date.max().date()),
        "frequency": "weekly" if bundle["dataset"] == "farmgate" else "monthly",
        "horizon": horizon,
        "history_observations": len(selected),
        "currency": "USD",
        "unit": "kg",
        "interval": {
            "method": "held_out_absolute_log_error",
            "nominal_coverage": config.coverage,
            "guaranteed_coverage": False,
            "scope": "pooled across series, per horizon",
        },
        "metrics": bundle["report"]["test"],
        "predictions": predictions,
        "warning": "Unverified existing data. Intervals are exploratory, and pooled calibration does not guarantee coverage for this series. Forecast dates are relative to the stored origin.",
    }
