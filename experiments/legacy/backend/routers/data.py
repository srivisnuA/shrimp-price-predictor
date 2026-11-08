"""Data upload routes."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, UploadFile

import data_store
from ml import advanced

router = APIRouter(prefix="/api/data")

MONTHLY_MARKERS = {"month", "region", "size_class", "price_usd_kg"}


@router.post("/upload")
async def upload_csv(file: UploadFile, mode: str = Query("merge", pattern="^(merge|replace)$")):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    raw = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    cols = {c.strip().lower() for c in df.columns}
    if "year" not in cols and cols & MONTHLY_MARKERS:
        raise HTTPException(
            status_code=400,
            detail="This looks like the old monthly format. Please upload YEARLY data — "
                   "required columns: year, avg_price_usd_kg; optional: avg_temp_c, rainfall_mm, "
                   "disease_outbreak_severity, disease_name, global_production_tonnes.",
        )
    if "month" in cols and "avg_price_usd_kg" not in cols:
        raise HTTPException(
            status_code=400,
            detail="Found a 'month' column but no 'avg_price_usd_kg'. The app now uses yearly rows — "
                   "required columns: year, avg_price_usd_kg (+ optional weather/disease columns).",
        )

    try:
        added, updated = data_store.merge_yearly(df, mode=mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # synchronous retrain of the advanced model on the updated yearly data
    metrics = {}
    model = None
    try:
        out = advanced.train_and_forecast(horizon=1)
        metrics = out["metrics"]
        model = out["model"]
    except ValueError:
        pass
    return {
        "rows_added": added,
        "rows_updated": updated,
        "columns_used": data_store.YEARLY_COLUMNS,
        "retrained": model is not None,
        "model": model,
        "metrics": metrics,
    }
