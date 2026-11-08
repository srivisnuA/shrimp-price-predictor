"""Download the IMF/FRED monthly global shrimp benchmark."""
from pathlib import Path
import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PSHRIUSDM"
OUT = Path("data/raw/prices/global_shrimp_fred.csv")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL)
    df.columns = ["date", "price_usd_kg"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_usd_kg"] = pd.to_numeric(df["price_usd_kg"], errors="coerce")
    df = df.dropna().sort_values("date")
    df.to_csv(OUT, index=False)
    print(f"saved {len(df):,} monthly price observations -> {OUT}")


if __name__ == "__main__":
    main()
