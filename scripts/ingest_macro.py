"""Download additional market variables from FRED without an API key."""
from pathlib import Path
import pandas as pd

SERIES = {
    "usd_inr": "DEXINUS",
    "wti_usd_bbl": "DCOILWTICO",
    "us_cpi": "CPIAUCSL",
}

def main():
    out = Path("data/raw/macro")
    out.mkdir(parents=True, exist_ok=True)
    for name, series in SERIES.items():
        df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        if series in {"DCOILWTICO", "DEXINUS"}:
            df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
            df = df.groupby("date", as_index=False)["value"].mean()
        df.rename(columns={"value": name}, inplace=True)
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"saved {name}: {len(df):,} observations")

if __name__ == "__main__":
    main()
