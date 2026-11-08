# Observation contract

UTF-8 CSV with these required columns. One file contains one price basis and
frequency. The trainer currently supports `farmgate` (weekly Mondays) and
`benchmark` (month starts). The latter's bundled data is illustrative; supplying
verified monthly data still requires review of its actual price basis.

| Column | Meaning |
|---|---|
| `date` | Observation period, ISO date at midnight; weekly dates are ISO Mondays |
| `available_at` | Actual first publication date; must be on or after `date` |
| `series_id` | Stable unique identifier, 1–120 characters |
| `country` | Country or market label, 1–120 characters |
| `species` | Stable species label, 1–120 characters |
| `size` | Stable count/kg or size-class string, 1–120 characters |
| `price_usd_kg` | Finite positive observation in USD per kilogram |

```csv
date,available_at,series_id,country,species,size,price_usd_kg
2026-05-04,2026-05-06,in:vannamei:60,India,vannamei,60,3.75
2026-05-11,2026-05-13,in:vannamei:60,India,vannamei,60,3.82
```

Two example rows are not enough to train a model. At the default four-period
horizon, at least 20 distinct panel dates and enough released observations per
trainable series are required. Missing dates remain missing. No target
interpolation, automatic duplicate averaging or export substitution occurs.

Every `(series_id, date)` must be unique. Resolve revisions explicitly into an
as-of snapshot before ingestion and retain the original vintages. Publication
dates cannot precede observation dates. The HTTP staging endpoint and training
CLI reject future/unreleased records. Training feature construction excludes
observations not known at each historical forecast origin.

Country, species and size metadata must remain constant for a series. Change
the identifier when the price basis, currency/unit, geography, species or size
definition changes. Convert currencies with a documented, point-in-time FX
source before intake. Do not mix local-currency prices with USD/kg prices.

Required provenance for a reviewed replacement: source name and URL or file,
acquisition date, source license/permission, geography, farm-gate/export basis,
species, size convention, currency/unit conversions, first-publication dates,
revision policy and known gaps. CSV validation checks structure, not whether a
source is reputable. It does not mark uploads verified automatically.

Future context integration should use a separate long table containing
`series_id, event_date, available_at, feature, value`. Join only records whose
event and publication dates precede the forecast origin. This table is a
proposed extension, not currently consumed. Annual production and revised
macro data must not be assigned retroactively to months before publication.
Future scenario values need explicit assumptions and their own backtest.
