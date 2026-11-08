"""Compatibility entry point for offline processing of committed data."""

from shrimp_forecast.data import build_existing

if __name__ == "__main__":
    for dataset, report in build_existing().items():
        print(f"{dataset}: {report['rows']} rows, {report['series']} series, {report['dates']} dates")
