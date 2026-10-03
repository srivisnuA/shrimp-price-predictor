# Data provenance

| Dataset | Source | Frequency | Role |
|---|---|---:|---|
| Global shrimp benchmark | IMF Primary Commodity Prices / FRED PSHRIUSDM | Monthly | Global price benchmark |
| Regional shrimp export unit value | UN Comtrade HS 030617 | Monthly | Region-specific price proxy |
| Weather | NASA POWER / MERRA-2 meteorology | Monthly | Temperature, precipitation, humidity, wind |
| Disease | WOAH WAHIS via EcoHealth Alliance WAHISDB extract | Event-based | Shrimp disease pressure |
| Production | FAO FishStat | Annual | Supply-side feature (next ingestion stage) |

## Important distinction

The model does not claim that every source is a farm-gate transaction price. The global IMF/FRED series is a benchmark, while the Comtrade series is an export unit-value proxy. The application records these separately and uses the regional export unit value when available, falling back to the global benchmark only when regional trade data is unavailable.
