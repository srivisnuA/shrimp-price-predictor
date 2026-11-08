"""Offline data build, training and promotion commands."""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path

from shrimp_forecast.data import ROOT, atomic_bytes, build_existing, json_bytes, read_csv
from shrimp_forecast.engine import TrainConfig, train
from shrimp_forecast.registry import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-data")
    build.add_argument("--as-of")
    build.add_argument("--output", type=Path, default=ROOT / "data/processed")
    trainer = sub.add_parser("train")
    trainer.add_argument("--dataset", choices=["farmgate", "benchmark"], required=True)
    trainer.add_argument("--data", type=Path)
    trainer.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    trainer.add_argument("--report", type=Path)
    trainer.add_argument("--max-horizon", type=int, default=4)
    trainer.add_argument("--coverage", type=float, default=0.9)
    trainer.add_argument("--lightgbm", action="store_true")
    trainer.add_argument(
        "--promote", action="store_true", help="Explicitly activate this run, including unverified/demo data"
    )
    args = parser.parse_args()
    if args.command == "build-data":
        reports = build_existing(output=args.output, as_of=args.as_of)
        print(
            json.dumps(
                {
                    key: {k: v for k, v in value.items() if k != "series_details"}
                    for key, value in reports.items()
                },
                indent=2,
            )
        )
        return
    path = args.data or ROOT / "data/processed" / f"{args.dataset}.csv"
    df = read_csv(path.read_bytes(), args.dataset)
    from datetime import datetime

    import pandas as pd

    if df.available_at.max() > pd.Timestamp(datetime.now(UTC).date()):
        parser.error("Training data contains unreleased/future observations")
    config = TrainConfig(max_horizon=args.max_horizon, coverage=args.coverage, lightgbm=args.lightgbm)
    bundle = train(df, args.dataset, config)
    run_id = publish(bundle, args.artifacts, promote=args.promote)
    if args.report:
        atomic_bytes(args.report, json_bytes(bundle["report"]))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "promoted": args.promote,
                "model": bundle["name"],
                "test": bundle["report"]["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
