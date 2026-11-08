"""Download FAO 2026 Global Production and extract shrimp aquaculture production."""
from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd
import requests

URL = "https://www.fao.org/fishery/static/Data/GlobalProduction_2026.1.0.zip"
COUNTRIES = {
    "India": 356,
    "Viet Nam": 704,
    "Ecuador": 218,
    "Indonesia": 360,
    "Thailand": 764,
}


def main():
    out = Path("data/raw/production")
    out.mkdir(parents=True, exist_ok=True)

    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        prod_name = next(n for n in z.namelist() if n.endswith("Global_production_quantity.csv"))
        species_name = next(n for n in z.namelist() if n.endswith("CL_FI_SPECIES_GROUPS.csv"))
        prod = pd.read_csv(z.open(prod_name), low_memory=False)
        species = pd.read_csv(z.open(species_name), low_memory=False)

    prod.columns = [str(c).strip().lower() for c in prod.columns]
    species.columns = [str(c).strip().lower() for c in species.columns]

    required = {
        "year",
        "species_alpha_3_code",
        "country_un_code",
        "production_source_det_code",
        "measure",
        "value",
    }
    missing = required - set(prod.columns)
    if missing:
        raise ValueError(f"Unexpected FAO schema; missing columns: {sorted(missing)}")
    if "name_en" not in species.columns:
        raise ValueError("FAO species lookup is missing name_en")

    shrimp_codes = set(
        species.loc[
            species["name_en"].astype(str).str.contains("shrimp|prawn", case=False, na=False),
            "x3a_code",
        ].astype(str)
    )

    prod["year"] = pd.to_numeric(prod["year"], errors="coerce")
    prod["country_un_code"] = pd.to_numeric(prod["country_un_code"], errors="coerce")
    prod["value"] = pd.to_numeric(prod["value"], errors="coerce")
    prod["species_alpha_3_code"] = prod["species_alpha_3_code"].astype(str)
    prod["measure"] = prod["measure"].astype(str)

    out_df = prod[
        prod["species_alpha_3_code"].isin(shrimp_codes)
        & prod["country_un_code"].isin(COUNTRIES.values())
        & prod["measure"].eq("Q_tlw")
    ].copy()

    out_df = (
        out_df.groupby(["year", "country_un_code"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "value_tonnes"})
    )
    out_df.to_csv(out / "shrimp_aquaculture_fao.csv", index=False)
    print(f"saved {len(out_df):,} FAO shrimp production rows")


if __name__ == "__main__":
    main()
