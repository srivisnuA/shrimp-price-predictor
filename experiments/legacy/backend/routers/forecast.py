"""Forecast + metrics API routes."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

import data_store
from ml import advanced
from ml.model import evaluate, forecast

router = APIRouter(prefix="/api")

SCENARIO_KEYS = {"avg_temp_c", "rainfall_mm", "disease_outbreak_severity", "global_production_tonnes"}


class AdvancedRequest(dict):
    pass


@router.get("/forecast/advanced")
def advanced_forecast_get(horizon: int = Query(5, ge=1, le=10)):
    try:
        return advanced.train_and_forecast(horizon=horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/forecast/advanced")
def advanced_forecast_post(body: dict, horizon: int = Query(5, ge=1, le=10)):
    scenario = {k: float(v) for k, v in (body or {}).items() if k in SCENARIO_KEYS}
    try:
        return advanced.train_and_forecast(horizon=horizon, scenario=scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model/metrics/advanced")
def advanced_metrics():
    try:
        out = advanced.train_and_forecast(horizon=1)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"model": out["model"], **out["metrics"]}


@router.get("/forecast")
def get_forecast(
    horizon: int = Query(12, ge=1, le=24),
    region: str = "Global",
    size_class: str = "Medium",
):
    df = data_store.load()
    df = df[(df["region"] == region) & (df["size_class"] == size_class)]
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No data for region={region!r}, size_class={size_class!r}")
    if len(df) < 4:
        raise HTTPException(status_code=400, detail="Not enough history for this series to forecast (need at least 4 points)")
    series = pd.Series(df["price_usd_kg"].values,
                       index=pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)))
    try:
        preds = forecast(series, horizon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "horizon": horizon,
        "region": region,
        "size_class": size_class,
        "history_tail": [
            {"date": d.strftime("%Y-%m"), "price": float(p)}
            for d, p in series.tail(24).items()
        ],
        "predictions": preds,
    }


@router.get("/model/metrics")
def get_metrics(region: str = "Global", size_class: str = "Medium"):
    df = data_store.load()
    df = df[(df["region"] == region) & (df["size_class"] == size_class)]
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No data for region={region!r}, size_class={size_class!r}")
    series = pd.Series(df["price_usd_kg"].values,
                       index=pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)))
    return evaluate(series)
