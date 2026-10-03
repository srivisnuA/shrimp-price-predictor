"""Build the region-month master table used by the forecasting models."""
from pathlib import Path
import json
import pandas as pd

DISEASES = {
    "white spot disease", "white spot syndrome", "white spot syndrome virus", "wssv",
    "acute hepatopancreatic necrosis disease", "ahpnd", "ems", "early mortality syndrome",
    "hepatopancreatic microsporidiosis", "ehp", "enterocytozoon hepatopenaei",
    "yellow head disease", "taura syndrome", "infectious myonecrosis"
}


def load_weather(region):
    p = Path("data/raw/weather") / f"{region}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices():
    df = pd.read_csv("data/raw/prices/global_shrimp_fred.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_trade(region):
    p = Path("data/raw/trade") / f"{region}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "trade_unit_value_usd_kg"])
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


COUNTRY_CODES = {"India": 356, "Viet Nam": 704, "Ecuador": 218, "Indonesia": 360}

def load_production(country):
    p = Path("data/raw/production/shrimp_aquaculture_fao.csv")
    if not p.exists():
        return pd.DataFrame(columns=["year", "shrimp_production_tonnes"])
    df = pd.read_csv(p)
    df["country_un_code"] = pd.to_numeric(df["country_un_code"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value_tonnes"] = pd.to_numeric(df["value_tonnes"], errors="coerce")
    code = COUNTRY_CODES.get(country)
    if code is None:
        return pd.DataFrame(columns=["year", "shrimp_production_tonnes"])
    out = df[df["country_un_code"].eq(code)].groupby("year", as_index=False)["value_tonnes"].sum()
    return out.rename(columns={"value_tonnes": "shrimp_production_tonnes"})


def disease_events():
    p = Path("data/raw/disease/wahis_epi_events.csv")
    if not p.exists():
        return pd.DataFrame(columns=["date", "country", "disease", "severity"])
    df = pd.read_csv(p, low_memory=False)
    # WAHIS schemas can change between extracts; select columns by semantic names.
    def col(*names):
        lower = {str(c).lower(): c for c in df.columns}
        for n in names:
            if n in lower: return lower[n]
        return None
    date_c = col("event_start_date", "start_date", "startdate")
    disease_c = col("disease_name", "disease", "disease_standardized")
    country_c = col("country", "country_name")
    if not all([date_c, disease_c, country_c]):
        return pd.DataFrame(columns=["date", "country", "disease", "severity"])
    out = pd.DataFrame({"date": pd.to_datetime(df[date_c], errors="coerce"), "country": df[country_c].astype(str), "disease": df[disease_c].astype(str)})
    out = out[out["disease"].str.lower().isin(DISEASES)].copy()
    out["severity"] = 1.0
    return out.dropna(subset=["date"])


def main():
    cfg = json.loads(Path("config/regions.json").read_text())["regions"]
    prices = load_prices()
    disease = disease_events()
    frames = []
    for region, rcfg in cfg.items():
        weather = load_weather(region)
        weather["date"] = weather["date"].dt.to_period("M").dt.to_timestamp()
        d = disease[disease["country"].str.lower().eq(rcfg["country"].lower())].copy()
        if d.empty:
            weather["disease_event_count"] = 0.0
            weather["disease_severity"] = 0.0
        else:
            d["date"] = d["date"].dt.to_period("M").dt.to_timestamp()
            counts = d.groupby("date").agg(disease_event_count=("disease", "size"), disease_severity=("severity", "sum")).reset_index()
            weather = weather.merge(counts, on="date", how="left")
            weather[["disease_event_count", "disease_severity"]] = weather[["disease_event_count", "disease_severity"]].fillna(0)
        merged = weather.merge(prices, on="date", how="left")
        trade = load_trade(region)
        production = load_production(rcfg["country"])
        merged["year"] = merged["date"].dt.year
        merged = merged.merge(production, on="year", how="left")
        merged = merged.merge(trade, on="date", how="left")
        merged["target_price_usd_kg"] = merged["trade_unit_value_usd_kg"].where(
            merged["trade_unit_value_usd_kg"].notna(), merged["price_usd_kg"]
        )
        merged["region"] = region
        merged["country"] = rcfg["country"]
        frames.append(merged)
    master = pd.concat(frames, ignore_index=True).sort_values(["region", "date"])
    # Leakage-safe lags and rolling features.
    for c in ["target_price_usd_kg", "price_usd_kg", "trade_unit_value_usd_kg", "t2m", "prectotcorr", "disease_severity"]:
        if c in master:
            g = master.groupby("region")[c]
            master[f"{c}_lag1"] = g.shift(1)
            master[f"{c}_lag3"] = g.shift(3)
            master[f"{c}_roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    master["month"] = master["date"].dt.month
    master["year"] = master["date"].dt.year
    out = Path("data/processed/shrimp_master_monthly.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False)
    print(f"saved {len(master):,} rows -> {out}")


if __name__ == "__main__":
    main()
