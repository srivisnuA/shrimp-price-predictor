"""Validate both the export model and the all-variety farm-gate model."""
import pandas as pd
from models.multivariate import walk_forward as walk_forward_export
from models.farmgate import walk_forward as walk_forward_farmgate

if __name__ == "__main__":
    export_path = "data/processed/shrimp_master_monthly.csv"
    farm_path = "data/processed/farmgate_weekly.csv"

    export_df = pd.read_csv(export_path, parse_dates=["date"])
    best, scores = walk_forward_export(export_df)
    print("export best model:", best)
    for name, metrics in scores.items():
        print("export", name, metrics)

    farm_df = pd.read_csv(farm_path, parse_dates=["date"])
    best_fg, scores_fg, baseline_fg = walk_forward_farmgate(farm_df)
    print("farm-gate best model:", best_fg)
    print("farm-gate latest-price baseline:", baseline_fg)
    for name, metrics in scores_fg.items():
        print("farm-gate", name, metrics)
