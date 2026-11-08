"""Run the reproducible data pipeline in the correct order."""
import subprocess
import sys

STEPS = [
    "scripts/ingest_price.py",
    "scripts/ingest_trade.py",
    "scripts/ingest_macro.py",
    "scripts/ingest_weather.py",
    "scripts/ingest_disease.py",
    "scripts/ingest_production.py",
    "scripts/ingest_farmgate.py",
    "scripts/build_master.py",
    "scripts/build_farmgate_master.py",
]

for step in STEPS:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, step], check=True)
