# Data pipeline

The forecasting system uses source-backed data instead of the old hand-built yearly series.

## Sources

- **Shrimp price:** IMF Primary Commodity Prices via FRED series `PSHRIUSDM` (monthly global shrimp benchmark, 1992-present).
- **Weather:** NASA POWER monthly meteorological data (MERRA-2-derived meteorology from 1981 onward).
- **Production:** FAO FishStat global aquaculture production (annual; shrimp species/country extraction is configurable).
- **Disease:** WOAH WAHIS / EcoHealth Alliance WAHISDB public extract for aquatic disease events.

Raw downloads are intentionally not committed automatically when they are large, dynamically updated, or subject to source redistribution terms. Run the ingestion scripts to reproduce them locally.
