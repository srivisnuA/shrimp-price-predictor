"""Download monthly NASA POWER weather for configured shrimp-producing regions."""
from pathlib import Path
import json
import time
import requests
import pandas as pd

PARAMETERS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS10M"
BASE = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch(region: str, cfg: dict, start: int = 1981, end: int | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.utcnow().year
    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": cfg["longitude"],
        "latitude": cfg["latitude"],
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()["properties"]["parameter"]
    rows = []
    keys = sorted(next(iter(payload.values())).keys())
    for ym in keys:
        year, month = int(ym[:4]), int(ym[4:])
        row = {"region": region, "date": f"{year:04d}-{month:02d}"}
        for k, values in payload.items():
            row[k.lower()] = values.get(ym)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    config = json.loads(Path("config/regions.json").read_text())
    out = Path("data/raw/weather")
    out.mkdir(parents=True, exist_ok=True)
    for region, cfg in config["regions"].items():
        print(f"downloading weather: {region}")
        df = fetch(region, cfg)
        df.to_csv(out / f"{region}.csv", index=False)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
