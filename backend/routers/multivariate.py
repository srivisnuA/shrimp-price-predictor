"""Regional multivariate forecast API."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from ml.multivariate import forecast_region, prepare

router = APIRouter(prefix="/api")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "shrimp_master_monthly.csv"

@router.get("/forecast/regional")
def regional_forecast(
    region: str = Query("india_andhra_pradesh"),
    horizon: int = Query(12, ge=1, le=36),
    temperature_c: float | None = None,
    rainfall_mm: float | None = None,
    disease_events: float | None = None,
    production_tonnes: float | None = None,
):
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Master dataset is not available. Run scripts/run_pipeline.py first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if region not in set(df["region"]):
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
    scenario = {k: v for k, v in {
        "temperature_c": temperature_c,
        "rainfall_mm": rainfall_mm,
        "disease_events": disease_events,
        "production_tonnes": production_tonnes,
    }.items() if v is not None}
    try:
        return forecast_region(df, region, horizon=horizon, scenario=scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.get("/model/multivariate/metrics")
def multivariate_metrics():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Master dataset is not available.")
    from ml.multivariate import walk_forward
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    best, scores = walk_forward(df)
    return {"model": best, "models": scores}
