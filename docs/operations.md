# Operating the service

## Build and promote

Install the pinned runtime from `requirements.lock` and the project with
`pip install --no-deps -e .`. Runtime currently targets Python 3.12/3.13.
The lock includes LightGBM and all resolved runtime dependencies. Dev tools are
separately pinned in `requirements-dev.lock`. Updates require a new resolution,
test run and retraining; artifact loading enforces exact model-library versions,
the Python minor version and forecasting-source hash.

Training writes immutable `artifacts/{dataset}/{run_id}/` directories. Each has
a checksum-protected `model.joblib` and a JSON manifest with environment,
configuration, data hash, source hash, git revision/dirty status and evaluation.
Only `--promote` atomically updates the dataset's `CURRENT.json` pointer, under
a file lock. Serving caches each active run and switches on the next request
after the pointer changes. A bad new pointer or artifact returns 503; it does
not silently substitute another model. Liveness remains 200; readiness fails.

Rollback: restore a previous valid `CURRENT.json` pointer using an atomic
filesystem replacement while holding `.registry.lock`. Restart serving after
changing Python, source code or library versions. Do not edit artifact files.

Joblib is executable serialization. Its checksum detects accidental changes,
not a malicious actor who can rewrite the manifest. **Only trusted trainers
may write the artifact directory.** Never download/load arbitrary pickle files.
HTTP uploads accept CSV only and stage separately. Containers run non-root and
their default read-only filesystem prevents runtime artifact modifications.

## Deployment boundary

`docker compose up --build` builds/trains both preview datasets and exposes
`http://127.0.0.1:8766`. The image serves on 8000 internally. Container building
and dependency installation require network access; data preparation/training
does not download external data. A fresh build produces new run identifiers.

Before exposing the service beyond loopback, configure an HTTPS reverse proxy
with request timeouts, body limits and rate limits; set `SHRIMP_ALLOWED_HOSTS`
to the exact public hostnames; and restrict metrics and intake at the proxy.
The app intentionally does not trust proxy-forwarded headers or enable CORS.
The default public endpoints are read-only. Administrator intake is opt-in via
a long random `SHRIMP_ADMIN_TOKEN` passed in the Authorization header, never a
URL. Supply secrets through the deployment environment, not Git or build args.

The app bounds POST bytes before multipart parsing, then bounds the file and
row count again. Uploads are limited to 2 MB of CSV and 50,000 rows by default;
multipart transport overhead gets a separate 64 KiB allowance. Limits do not
replace reverse-proxy rate/concurrency controls. Configure a writable upload
volume only if intake is needed; the default Compose deployment is read-only.

Training is intentionally a separate process, not a request handler or an
in-process background job. For larger histories, run training on a worker with
bounded BLAS/OpenMP threads and share reviewed, immutable artifacts with the
serving deployment. Current training uses pandas feature construction and fits
small tabular models; benchmark and replace the feature builder before claiming
large-scale training capacity. No distributed queue or cluster is required for
the existing dataset.

## Monitoring and recovery

- `/api/health`: process liveness; use for restart checks.
- `/api/ready`: 200 only when both datasets have compatible active artifacts.
- `/metrics`: low-cardinality route-template counts and latency histograms.
- API logs: JSON request ID, method, route template, status and duration. Request
  bodies, upload contents, tokens and arbitrary query strings are not logged by
  the application. Configure proxy/access logs with equivalent redaction.
- Dataset quality: track age, gaps, row counts, schema failures and source hashes.
  Model reports track MAE, bias, baseline delta and actual interval coverage.

Back up trusted source snapshots, manifests, active pointers and model artifacts.
After a new verified feed arrives, audit the source, normalize its vintages,
validate it, train an unpromoted run and inspect its report before promotion.
Do not use the existing small final test indefinitely for repeated model tuning;
reserve a fresh future block for later model changes. Interval undercoverage is
a visible limitation requiring more independent calibration history.

## Checks and present limits

Local verification uses pytest, Ruff, JavaScript syntax validation and interactive
browser checks. CI includes the Linux container build and readiness/forecast
smoke test. A machine without Docker cannot establish that the image builds;
read the CI result before deploying it. Neither CI nor a successful build proves
market accuracy, source licensing, security certification or production SLOs.
