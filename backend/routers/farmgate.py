"""Farm-gate APIs for every available country/species/size."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ml.farmgate import forecast, history, options, walk_forward

router = APIRouter(prefix="/api/farmgate")
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "farmgate_weekly.csv"


def load_df():
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Farm-gate dataset is not built. Run scripts/run_pipeline.py first.",
        )
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@router.get("/options")
def farmgate_options():
    return options(load_df())


@router.get("/history")
def farmgate_history(
    country_code: str = Query(...),
    species: str = Query(...),
    size: float = Query(..., ge=1),
):
    try:
        return history(load_df(), country_code, species, size)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/forecast")
def farmgate_forecast(
    country_code: str = Query(...),
    species: str = Query(...),
    size: float = Query(..., ge=1),
    horizon_weeks: int = Query(26, ge=1, le=52),
    temperature_c: float | None = None,
    rainfall_mm: float | None = None,
    disease_events: float | None = None,
    disease_severity: float | None = None,
    production_tonnes: float | None = None,
    usd_inr: float | None = None,
    wti_usd_bbl: float | None = None,
    us_cpi: float | None = None,
    global_price_usd_kg: float | None = None,
    export_price_usd_kg: float | None = None,
):
    scenario = {
        k: v
        for k, v in {
            "temperature_c": temperature_c,
            "rainfall_mm": rainfall_mm,
            "disease_events": disease_events,
            "disease_severity": disease_severity,
            "production_tonnes": production_tonnes,
            "usd_inr": usd_inr,
            "wti_usd_bbl": wti_usd_bbl,
            "us_cpi": us_cpi,
            "global_price_usd_kg": global_price_usd_kg,
            "export_price_usd_kg": export_price_usd_kg,
        }.items()
        if v is not None
    }

    try:
        return forecast(
            load_df(),
            country_code,
            species,
            size,
            horizon_weeks=horizon_weeks,
            scenario=scenario,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/metrics")
def farmgate_metrics():
    df = load_df()
    try:
        best, scores, baseline = walk_forward(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "model": best,
        "models": scores,
        "latest_price_baseline": baseline,
        "countries": int(df["country_code"].nunique()),
        "species": int(df["species"].nunique()),
        "sizes": int(df["size"].nunique()),
        "weeks": int(df["date"].nunique()),
        "rows": int(len(df)),
    }
