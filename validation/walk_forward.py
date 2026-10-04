"""CLI for leakage-safe walk-forward validation of both price bases."""
import pandas as pd
from models.multivariate import walk_forward

if __name__ == "__main__":
    df = pd.read_csv("data/processed/shrimp_master_monthly.csv", parse_dates=["date"])

    for price_type in ["export", "farm_gate"]:
        try:
            best, scores = walk_forward(df, price_type=price_type)
        except ValueError as exc:
            print(f"\n{price_type}: skipped — {exc}")
            continue

        print(f"\n{price_type} best model: {best}")
        for name, metrics in scores.items():
            print(name, metrics)
