# Data provenance

| Dataset | Source | Frequency | Role |
|---|---|---:|---|
| Farm-gate shrimp price | Shrimp Insights Farm Gate Price portal | Weekly source, normalized monthly | Farmer-to-buyer raw shrimp target |
| Global shrimp benchmark | IMF Primary Commodity Prices / FRED PSHRIUSDM | Monthly | Global benchmark |
| Regional shrimp export unit value | UN Comtrade HS 030617 | Monthly | Export-market target/proxy |
| Weather | NASA POWER / MERRA-2 meteorology | Monthly | Temperature, precipitation, humidity, wind |
| Disease | WOAH WAHIS via EcoHealth Alliance WAHISDB extract | Event-based | Shrimp disease pressure |
| Production | FAO FishStat | Annual | Supply-side feature |
| Macro controls | FRED USD/INR, WTI, US CPI | Monthly | Market/macro controls |

## Price-basis separation

The dashboard now has two independent targets:

- **Farm-gate:** observed farmer-to-buyer price series from the Shrimp Insights farm-gate portal, filtered to the configured Vannamei size. Missing farm-gate observations are not replaced by export prices.
- **Export:** regional export unit value where available, with the IMF/FRED global shrimp benchmark used only when regional trade data is unavailable.

The Shrimp Insights portal states that its farm-gate figures are averages from local partners and should be treated as an indication of farm-gate price movement rather than an exact price for every transaction. The portal currently covers major origins including India, Ecuador, Indonesia and Viet Nam, with species/size filters.