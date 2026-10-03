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
    usd_inr: float | None = None,
    wti_usd_bbl: float | None = None,
    us_cpi: float | None = None,
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
        "usd_inr": usd_inr,
        "wti_usd_bbl": wti_usd_bbl,
        "us_cpi": us_cpi,
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

@router.get("/regional/history")
def regional_history(region: str = Query("india_andhra_pradesh"), limit: int = Query(120, ge=12, le=240)):
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Master dataset is not available.")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["region"] == region].sort_values("date").tail(limit)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
    return [
        {"date": r["date"].strftime("%Y-%m"), "price": None if pd.isna(r["target_price_usd_kg"]) else round(float(r["target_price_usd_kg"]), 3),
         "temperature": None if pd.isna(r["t2m"]) else round(float(r["t2m"]), 2),
         "rainfall": None if pd.isna(r["prectotcorr"]) else round(float(r["prectotcorr"]), 2),
         "disease": round(float(r["disease_severity"]), 2),
         "production": None if pd.isna(r["shrimp_production_tonnes"]) else round(float(r["shrimp_production_tonnes"]), 0)}
        for _, r in df.iterrows()
    ]

@router.get("/regional/options")
def regional_options():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Master dataset is not available.")
    df = pd.read_csv(DATA_PATH, usecols=["region", "country"])
    return [{"region": r, "country": c} for r, c in df.drop_duplicates().sort_values("region").itertuples(index=False)]
