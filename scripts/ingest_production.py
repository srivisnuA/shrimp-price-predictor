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
    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    z = zipfile.ZipFile(BytesIO(response.content))
    prod = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith("Global_production_quantity.csv"))), low_memory=False)
    species = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith("CL_FI_SPECIES_GROUPS.csv"))), low_memory=False)
    countries = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith("CL_FI_COUNTRY_GROUPS.csv"))), low_memory=False)
    source = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith("CL_FI_PRODUCTION_SOURCE_DET.csv"))), low_memory=False)
    shrimp_codes = species.loc[species["name_en"].astype(str).str.contains("shrimp|prawn", case=False, na=False), "x3a_code"].astype(str)
    country_codes = countries.loc[countries["name_en"].isin(COUNTRIES), "un_code"].astype(str)
    aqua_codes = source.loc[source["name_en"].astype(str).str.contains("Aquaculture production", case=False, na=False), "code"].astype(str)
    prod["species_alpha_3_code"] = prod["species_alpha_3_code"].astype(str)
    prod["country_un_code"] = prod["country_un_code"].astype(str)
    prod["production_source_det_code"] = prod["production_source_det_code"].astype(str)
    prod = prod[prod["species_alpha_3_code"].isin(shrimp_codes) & prod["country_un_code"].isin(country_codes) & prod["production_source_det_code"].isin(aqua_codes) & prod["measure"].eq("Q_tlw")].copy()
    prod["year"] = pd.to_numeric(prod["period"], errors="coerce")
    prod["value_tonnes"] = pd.to_numeric(prod["value"], errors="coerce")
    prod[["year", "country_un_code", "species_alpha_3_code", "value_tonnes"]].to_csv(out / "shrimp_aquaculture_fao.csv", index=False)
    print(f"saved {len(prod):,} FAO shrimp aquaculture rows")

if __name__ == "__main__":
    main()
