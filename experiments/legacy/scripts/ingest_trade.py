"""Download monthly regional shrimp export unit values from UN Comtrade preview."""
from pathlib import Path
import time
import requests
import pandas as pd

REPORTERS = {
    "india_andhra_pradesh": 699,
    "india_tamil_nadu": 699,
    "vietnam_mekong": 704,
    "ecuador_guayas": 218,
    "indonesia_sumatra": 360,
    "thailand_eastern": 764,
}
BASE = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"
YEARS = range(1992, pd.Timestamp.utcnow().year + 1)


def fetch_year(reporter: int, year: int):
    periods = ",".join(f"{year}{m:02d}" for m in range(1, 13))
    params = {
        "reporterCode": reporter,
        "partnerCode": 0,
        "flowCode": "X",
        "cmdCode": "030617",
        "period": periods,
        "maxRecords": 500,
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    rows = []
    for x in r.json().get("data", []):
        weight = x.get("netWgt") or x.get("netWgt_kg")
        value = x.get("primaryValue")
        if weight and value and float(weight) > 0:
            rows.append(
                {
                    "date": pd.to_datetime(str(x["period"]), format="%Y%m"),
                    "trade_unit_value_usd_kg": float(value) / float(weight),
                }
            )
    return pd.DataFrame(rows)


def main():
    out = Path("data/raw/trade")
    out.mkdir(parents=True, exist_ok=True)
    for region, reporter in REPORTERS.items():
        frames = []
        for year in YEARS:
            try:
                part = fetch_year(reporter, year)
                if not part.empty:
                    frames.append(part)
            except requests.HTTPError as exc:
                print(f"warning: {region} {year}: {exc}")
            time.sleep(0.25)
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["date", "trade_unit_value_usd_kg"])
        )
        df.drop_duplicates("date").sort_values("date").to_csv(out / f"{region}.csv", index=False)
        print(f"saved trade proxy: {region} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
