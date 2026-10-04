# Data pipeline

The forecasting system uses source-backed data instead of the old hand-built yearly series.

## Price targets

- **Farm-gate:** Shrimp Insights public farm-gate portal, normalized monthly and filtered to configured species/size.
- **Export:** UN Comtrade regional export unit value, with IMF/FRED global shrimp benchmark fallback.

## Other sources

- **Weather:** NASA POWER monthly meteorological data.
- **Production:** FAO FishStat global aquaculture production.
- **Disease:** WOAH WAHIS / EcoHealth Alliance WAHISDB public extract.
- **Macro:** FRED USD/INR, WTI and US CPI.

Raw downloads are intentionally not committed automatically when they are large, dynamically updated, or subject to source redistribution terms. The farm-gate source can be supplied through the portal download file or a direct URL.