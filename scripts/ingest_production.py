"""Download FAO 2026 Global Production and extract shrimp aquaculture production."""
from pathlib import Path
from io import BytesIO
import zipfile
import requests
import pandas as pd

URL = "https://www.fao.org/fishery/static/Data/GlobalProduction_2026.1.0.zip"
COUNTRIES = {"India", "Viet Nam", "Ecuador", "Indonesia"}

def main():
    out = Path("data/raw/production")
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(BytesIO(requests.get(URL, timeout=180).content))
    names = z.namelist()
    prod_name = next(n for n in names if n.endswith("Global_production_quantity.csv"))
    species_name = next(n for n in names if n.endswith("CL_FI_SPECIES_GROUPS.csv"))
    country_name = next(n for n in names if n.endswith("CL_FI_COUNTRY_GROUPS.csv"))
    prod = pd.read_csv(z.open(prod_name), low_memory=False)
    species = pd.read_csv(z.open(species_name), low_memory=False)
    countries = pd.read_csv(z.open(country_name), low_memory=False)
    species.columns = [str(c).lower() for c in species.columns]
    countries.columns = [str(c).lower() for c in countries.columns]
    prod.columns = [str(c).lower() for c in prod.columns]
    species["name_en"] = species["name_en"].astype(str)
    shrimp = species[species["name_en"].str.contains("shrimp|prawn", case=False, na=False)]["x3a_code"].astype(str)
    prod["species"] = prod["species"].astype(str)
    prod["year"] = pd.to_numeric(prod["year"], errors="coerce")
    prod["value"] = pd.to_numeric(prod["value"], errors="coerce")
    prod = prod[prod["species"].isin(shrimp) & prod["country"].isin(countries[countries["name_en"].isin(COUNTRIES)]["un_code"].astype(str))]
    prod = prod[prod["production_source"].astype(str).str.contains("AQUA", case=False, na=False)] if "production_source" in prod else prod
    keep = [c for c in ["year","country","species","value","unit","production_source"] if c in prod.columns]
    prod[keep].to_csv(out / "shrimp_aquaculture_fao.csv", index=False)
    print(f"saved {len(prod):,} FAO shrimp production rows")

if __name__ == "__main__":
    main()
