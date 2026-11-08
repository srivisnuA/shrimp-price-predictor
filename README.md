# Shrimp Forecast

An offline-trained, auditable shrimp-price forecasting platform with a FastAPI
service and a responsive dashboard. It uses the existing repository data until
verified sources are supplied. **The engineering foundation is production-oriented;
the current forecasts remain a research preview, not validated market intelligence.**

## What changed in v3

- One forecasting package replaces duplicate model implementations and broken import paths.
- Strict, atomic CSV intake rejects invalid rows, duplicate keys and invalid publication dates.
- Farm-gate prices and the illustrative monthly sample have separate models and labels.
- Calendar-aware lags retain gaps; unavailable future targets never enter features.
- Recursive, multi-horizon forecasts use the same implementation in backtests and serving.
- Latest-price, seasonal naïve and damped-drift baselines compete with Ridge, histogram gradient boosting and optional LightGBM.
- Chronological validation selects the model; later, disjoint calibration and final-test blocks measure intervals and generalization.
- Immutable local artifacts record data hashes, source hashes, dependency versions, model configuration and test results. Promotion is explicit.
- Requests load cached artifacts; they never fit a model or run cross-validation.
- Versioned endpoints, typed forecast responses, authenticated bounded data staging, request IDs, Prometheus metrics and readiness checks support operations.
- Tests, pinned runtime dependencies, CI and a non-root container replace syntax-only checks and automatic source scraping.

## Quick start

Use **Python 3.12 or 3.13** from the repository root. The local preview can use port 8766.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
shrimp build-data
shrimp train --dataset farmgate --lightgbm --promote --report reports/local/farmgate.json
shrimp train --dataset benchmark --lightgbm --promote --report reports/local/benchmark.json
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8766
```

On Linux/macOS, activate with `source .venv/bin/activate`. Open
[the dashboard](http://127.0.0.1:8766/) and [API docs](http://127.0.0.1:8766/docs).
`python -m shrimp_forecast.cli` can replace the `shrimp` console command.
Data preparation uses committed CSVs and makes **no network requests**.
Omit `--lightgbm` to use only scikit-learn candidates. Omit `--promote` to evaluate
a run without changing the active model. Retrain after changing forecasting
source code or dependency versions: incompatible artifacts fail closed.

## Existing data and measured limits

| Dataset | Current coverage | Meaning |
|---|---|---|
| Farm-gate | 1,000 rows, 64 series, 22 weekly dates, May–September 2026 | Existing download labeled as Shrimp Insights; provenance and release dates have not been independently verified |
| Monthly sample | 144 rows, 4 series, 96 distinct months, 2016–2023 | Illustrative repository sample; **not verified regional export prices** |

Coverage is irregular, and individual series can be much shorter than the full
panel. The dashboard exposes missing periods, observation age and forecast origin.
Forecast dates start at the saved origin; stale data does not silently become a
forecast from today. Weather, disease, production and macro scenario controls are
excluded until verified observations and publication vintages are available.

The first v3 farm-gate run selected **latest-price persistence** over the learned
challengers. Its independent final block had **MAE ≈ $0.201/kg** across 165 matched
observations. Nominal 90% intervals covered **69.7%**, demonstrating substantial
undercoverage. This is a diagnostic of the current data and split, not a promise
of future accuracy. See [the evaluation snapshot](reports/baseline-evaluation.md).

## Validation and uncertainty

With the default four-period horizon, training needs at least 20 distinct dates:
eight initial training dates plus three four-period blocks. Up to six expanding
validation origins are sampled; each produces a recursive forecast without using
intermediate realized prices. Preprocessing fits only on that origin's training
data. Targets from validation, calibration and test never overlap across stages.
Models are ranked by pooled MAE. A challenger must improve validation MAE by at
least 2% to replace persistence. The final test is never used to reselect a model.

One subsequent origin supplies absolute log-error calibration separately by
horizon; another supplies the untouched final audit. Reports include MAE, RMSE,
WAPE, bias, per-series and per-horizon metrics, baseline performance and interval
coverage. Operational models are then refit on all available history, while the
stored metrics remain from the earlier audit.

The interval uses a finite-sample corrected residual quantile. If too few samples
exist for a finite bound, `lower` and `upper` are `null`. Series are correlated and
the calibration window is short: **there is no conformal coverage guarantee**.
The monthly sample has only four matched final-test targets and insufficient
calibration samples for finite 90% bands. Default serving is limited to four
periods; a longer horizon requires retraining, enough history and a new audit.

## Replacing the data

See [the observation contract](docs/data-contract.md). Prepare a complete,
normalized snapshot with dates, publication dates, series identity and USD/kg
prices. Use a distinct dataset for different price bases; never replace missing
farm-gate prices with export proxies. Preserve source files and redistribution
permissions outside the public repository.

```powershell
shrimp train --dataset farmgate --data path/to/normalized.csv --lightgbm --report reports/local/candidate.json
# Inspect the new report, then explicitly train and activate the reviewed snapshot:
shrimp train --dataset farmgate --data path/to/normalized.csv --lightgbm --promote
```

For HTTP intake, configure `SHRIMP_ADMIN_TOKEN` and use
`POST /api/v1/datasets/farmgate/uploads?source=SOURCE&dry_run=true` with
`Authorization: Bearer TOKEN` and a multipart CSV. Validation is the default.
`dry_run=false` stages a content-addressed snapshot under `data/uploads`; it does
not overwrite observations, train models or change the active model. Treat
staged input as unverified until reviewed. Intake is disabled without a token.

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | Process liveness |
| `GET /api/ready` | Both required datasets have loadable promoted artifacts |
| `GET /api/v1/datasets` | Dataset descriptions and price-basis distinctions |
| `GET /api/v1/datasets/{dataset}/series` | Available series, gaps and age |
| `GET /api/v1/datasets/{dataset}/history?series_id=...` | Bounded historical observations |
| `GET /api/v1/datasets/{dataset}/forecast?series_id=...&horizon=4` | Forecast, intervals, origin and run identifier |
| `GET /api/v1/datasets/{dataset}/forecast.csv?series_id=...` | Download traceable forecasts |
| `GET /api/v1/datasets/{dataset}/quality` | Data fingerprint and coverage |
| `GET /api/v1/datasets/{dataset}/evaluation` | Validation, calibration and final audit |
| `POST /api/v1/datasets/{dataset}/uploads` | Authenticated validation/staging |
| `GET /metrics` | Prometheus request counts and latency |

The old unversioned forecast/upload routes were retired. Prototype models,
scrapers, tests and yearly sample remain in `experiments/legacy` for reference;
they are not imported or scheduled. `scripts/run_pipeline.py` now builds existing
data offline when invoked from an installed checkout.

## Tests and deployment

```powershell
ruff check .
ruff format --check shrimp_forecast backend tests scripts
node --check backend/static/app.js
pytest --cov=shrimp_forecast --cov=backend --cov-fail-under=85
docker compose up --build
```

The Docker build trains both datasets from committed data and serves as a
non-root user. Compose binds to loopback with a read-only filesystem. See
[the operations guide](docs/operations.md) before exposing a deployment.
CI tests Python 3.12/3.13 on Windows and Linux and builds/smokes the container.
The manual training workflow uploads reports without publishing artifacts or
committing data. There is no automatic external refresh.

Time-aware evaluation follows the principles illustrated in the
[scikit-learn lagged-forecast example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
The container uses the process and shutdown approach described in
[FastAPI's container guidance](https://fastapi.tiangolo.com/deployment/docker/).
