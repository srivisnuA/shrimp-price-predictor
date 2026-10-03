"""Download the public WAHIS disease-event extract from EcoHealth Alliance's DoltHub mirror.

The mirror documents the tables and preserves source-derived outbreak information. We keep
only aquatic shrimp-relevant diseases during master-dataset construction.
"""
from pathlib import Path
import requests

BASE = "https://www.dolthub.com/csv/ecohealthalliance/wahisdb/main"
TABLES = ["wahis_epi_events", "wahis_outbreaks", "disease_key"]


def main():
    out = Path("data/raw/disease")
    out.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        r = requests.get(f"{BASE}/{table}", timeout=120)
        r.raise_for_status()
        path = out / f"{table}.csv"
        path.write_bytes(r.content)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
