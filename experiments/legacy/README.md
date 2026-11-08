# Archived prototype

This directory preserves the previous models, scrapers, yearly illustrative
dataset and tests for reference. It is excluded from the installed package,
test discovery and production execution. Its old relative paths and imports
are historical, not supported entry points.

The prototype mixed price targets and had full-dataset imputation, inconsistent
backtest/forecast features, contemporaneous unreleased context, training-error
intervals and request-time training. Reusing its context scrapers requires a
source and publication-vintage review. No scraper runs automatically in v3.

Use `shrimp build-data`, `shrimp train` and `uvicorn backend.main:app` instead.
