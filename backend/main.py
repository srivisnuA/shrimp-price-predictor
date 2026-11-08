"""FastAPI serving boundary. Training and promotion run exclusively through the CLI."""

from __future__ import annotations

import csv
import hmac
import io
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.limits import UploadLimitMiddleware
from backend.settings import Settings
from shrimp_forecast import __version__
from shrimp_forecast.data import atomic_bytes, canonical_bytes, fingerprint, json_bytes, quality, read_csv
from shrimp_forecast.engine import forecast
from shrimp_forecast.registry import Registry

logger = logging.getLogger("shrimp.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
STATIC = Path(__file__).parent / "static"


class Dataset(StrEnum):
    farmgate = "farmgate"
    benchmark = "benchmark"


class Prediction(BaseModel):
    date: str
    horizon: int
    prediction: float = Field(gt=0, allow_inf_nan=False)
    lower: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    upper: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    dataset: Dataset
    series_id: str
    model: str
    run_id: str
    predictions: list[Prediction]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    registry = Registry(settings.artifact_dir)
    app = FastAPI(title="Shrimp Forecast API", version=__version__)
    app.state.registry, app.state.settings = registry, settings
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(UploadLimitMiddleware, max_bytes=settings.max_upload_bytes + 65536)
    collectors = CollectorRegistry()
    requests = Counter(
        "shrimp_http_requests_total", "HTTP responses", ["method", "route", "status"], registry=collectors
    )
    latency = Histogram(
        "shrimp_http_request_seconds", "HTTP request duration", ["route"], registry=collectors
    )

    @app.middleware("http")
    async def request_context(request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = settings.max_upload_bytes + 65537
        if request.method == "POST" and length > settings.max_upload_bytes + 65536:
            response = JSONResponse(status_code=413, content={"detail": "Upload exceeds size limit"})
        else:
            response = await call_next(request)
        response.headers.update(
            {
                "X-Request-ID": request_id,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "same-origin",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'",
            }
        )
        if request.url.path in {"/docs", "/docs/oauth2-redirect", "/redoc"}:
            # FastAPI's interactive documentation loads its renderer from jsDelivr.
            # Keep this allowance scoped to documentation; the dashboard remains self-contained.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "style-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com 'unsafe-inline'; "
                "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https://fastapi.tiangolo.com; "
                "object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
            )
        route = getattr(request.scope.get("route"), "path", "unmatched")
        duration = time.perf_counter() - started
        requests.labels(request.method, route, str(response.status_code)).inc()
        latency.labels(route).observe(duration)
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            )
        )
        return response

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        logger.error("Unhandled request failure", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal service error", "request_id": request.state.request_id},
        )

    def load(dataset: Dataset):
        try:
            return registry.load(dataset.value)
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            logger.warning("Artifact unavailable for %s: %s", dataset.value, exc)
            raise HTTPException(
                503, "Trained model is unavailable. Check readiness and the training run."
            ) from exc

    def authorize(authorization: str | None = Header(default=None)):
        expected = settings.admin_token.get_secret_value() if settings.admin_token else ""
        if not expected:
            raise HTTPException(403, "Data intake is disabled")
        supplied = (authorization or "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "Invalid administrator token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/ready")
    def ready():
        status = {}
        for dataset in Dataset:
            try:
                bundle = registry.load(dataset.value)
                status[dataset.value] = {"ready": True, "run_id": bundle["run_id"]}
            except (FileNotFoundError, ValueError, KeyError, OSError):
                status[dataset.value] = {"ready": False}
        ok = all(value["ready"] for value in status.values())
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ready" if ok else "degraded", "datasets": status},
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus():
        return Response(generate_latest(collectors), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/datasets")
    def datasets():
        return [
            {
                "id": "farmgate",
                "name": "Farm-gate observations",
                "frequency": "weekly",
                "description": "Committed download; country, species and size retained. Unverified provenance.",
            },
            {
                "id": "benchmark",
                "name": "Monthly sample",
                "frequency": "monthly",
                "description": "Illustrative existing data, not verified regional export prices.",
            },
        ]

    @app.get("/api/v1/datasets/{dataset}/series")
    def series(dataset: Dataset):
        return quality(load(dataset)["history"], dataset.value)["series_details"]

    @app.get("/api/v1/datasets/{dataset}/quality")
    def data_quality(dataset: Dataset):
        bundle = load(dataset)
        return {
            **quality(bundle["history"], dataset.value),
            "run_id": bundle["run_id"],
            "trained_at": bundle["report"]["trained_at"],
            "max_horizon": bundle["config"].max_horizon,
        }

    @app.get("/api/v1/datasets/{dataset}/history")
    def history(
        dataset: Dataset,
        series_id: str = Query(min_length=1, max_length=120),
        limit: int = Query(120, ge=1, le=520),
    ):
        df = load(dataset)["history"]
        selected = df[df.series_id.eq(series_id)].sort_values("date").tail(limit)
        if selected.empty:
            raise HTTPException(404, "Unknown series")
        return [
            {"date": str(row.date.date()), "price": row.price_usd_kg}
            for row in selected.itertuples(index=False)
        ]

    @app.get("/api/v1/datasets/{dataset}/forecast", response_model=ForecastResponse)
    def get_forecast(
        dataset: Dataset,
        series_id: str = Query(min_length=1, max_length=120),
        horizon: int = Query(4, ge=1, le=12),
    ):
        bundle = load(dataset)
        try:
            result = forecast(bundle, series_id, horizon)
        except KeyError as exc:
            raise HTTPException(404, "Unknown series") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {**result, "run_id": bundle["run_id"]}

    @app.get("/api/v1/datasets/{dataset}/evaluation")
    def evaluation(dataset: Dataset):
        bundle = load(dataset)
        return {**bundle["report"], "run_id": bundle["run_id"]}

    @app.get("/api/v1/datasets/{dataset}/forecast.csv")
    def download(
        dataset: Dataset,
        series_id: str = Query(min_length=1, max_length=120),
        horizon: int = Query(4, ge=1, le=12),
    ):
        result = get_forecast(dataset, series_id, horizon)
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream, fieldnames=["date", "horizon", "prediction", "lower", "upper", "run_id"]
        )
        writer.writeheader()
        writer.writerows({**row, "run_id": result["run_id"]} for row in result["predictions"])
        return Response(
            stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="shrimp-forecast.csv"'},
        )

    @app.post("/api/v1/datasets/{dataset}/uploads", dependencies=[Depends(authorize)])
    async def upload(
        dataset: Dataset,
        file: UploadFile = File(),
        dry_run: bool = True,
        source: str = Query(min_length=1, max_length=200),
    ):
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(415, "A UTF-8 .csv file is required")
        raw = bytearray()
        while chunk := await file.read(65536):
            raw.extend(chunk)
            if len(raw) > settings.max_upload_bytes:
                raise HTTPException(413, "Upload exceeds size limit")

        def validate():
            frame = read_csv(bytes(raw), dataset.value)
            if len(frame) > settings.max_upload_rows:
                raise ValueError("Upload exceeds row limit")
            if frame.available_at.max().date() > datetime.now(UTC).date():
                raise ValueError("Upload contains future/unreleased observations")
            return frame

        try:
            frame = await run_in_threadpool(validate)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        report = quality(frame, dataset.value)
        if not dry_run:
            digest = fingerprint(frame)
            destination = settings.upload_dir / dataset.value / f"{digest}.csv"
            await run_in_threadpool(atomic_bytes, destination, canonical_bytes(frame))
            await run_in_threadpool(
                atomic_bytes,
                destination.with_suffix(".json"),
                json_bytes(
                    {"source": source, "received_at": datetime.now(UTC).isoformat(), "quality": report}
                ),
            )
        return {
            "dry_run": dry_run,
            "staged": not dry_run,
            "sha256": report["sha256"],
            "quality": report,
            "active_model_changed": False,
        }

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")
    return app


app = create_app()
