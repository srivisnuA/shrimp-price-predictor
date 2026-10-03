"""CLI for leakage-safe walk-forward validation."""
import pandas as pd
from models.multivariate import walk_forward

if __name__ == "__main__":
    df = pd.read_csv("data/processed/shrimp_master_monthly.csv", parse_dates=["date"])
    best, scores = walk_forward(df)
    print("best model:", best)
    for name, metrics in scores.items(): print(name, metrics)
