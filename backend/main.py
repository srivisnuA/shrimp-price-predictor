"""Shrimp Price Predictor — FastAPI application."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import data_store
from routers import data, forecast, multivariate

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Shrimp Price Predictor", version="1.0.0")
app.include_router(forecast.router)
app.include_router(data.router)
app.include_router(multivariate.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/prices")
def prices(region: str | None = None, size_class: str | None = None):
    df = data_store.load()
    if region:
        df = df[df["region"] == region]
    if size_class:
        df = df[df["size_class"] == size_class]
    return data_store.records(df)


@app.get("/api/dimensions")
def dimensions():
    return data_store.dimensions()


@app.get("/api/prices/yearly")
def prices_yearly(year_from: int | None = None, year_to: int | None = None):
    rows = data_store.yearly_records(year_from=year_from, year_to=year_to)
    # sanitize: NaN/Inf cannot be serialized to JSON
    import math
    for row in rows:
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    return rows


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    @app.get("/")
    def index():  # fallback so the dashboard always serves
        return FileResponse(STATIC_DIR / "index.html")
