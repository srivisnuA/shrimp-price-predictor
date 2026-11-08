FROM python:3.13-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

FROM base AS trained
COPY pyproject.toml README.md ./
COPY shrimp_forecast/ shrimp_forecast/
COPY backend/ backend/
COPY data/farmgate/ data/farmgate/
RUN pip install --no-deps . && python -m shrimp_forecast.cli build-data \
    && python -m shrimp_forecast.cli train --dataset farmgate --lightgbm --promote \
    && python -m shrimp_forecast.cli train --dataset benchmark --lightgbm --promote

FROM base AS runtime
COPY --from=trained /app /app
RUN useradd --uid 10001 --create-home shrimp && mkdir -p /app/data/uploads && chown shrimp:shrimp /app/data/uploads
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--no-access-log"]
