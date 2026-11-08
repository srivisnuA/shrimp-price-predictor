# Data used by v3

`shrimp build-data` reads only the committed farm-gate source and the existing
monthly sample, producing canonical CSVs and quality reports in ignored
`data/processed/`. It never fetches from a website.

- `farmgate/shrimp_insights_farmgate.csv`: unchanged source download attributed
  by the prototype to Shrimp Insights. All 64 country/species/size series remain.
- `../backend/data/shrimp_prices_sample.csv`: illustrative monthly sample,
  explicitly separated from farm-gate prices and not presented as export data.
- `uploads/`: ignored, content-addressed CSV staging, opt-in through authenticated
  HTTP intake. Staging does not change trained artifacts.

Raw publication vintages and source authenticity are not available in the
committed files. Bundled `available_at` is therefore assumed equal to `date`.
Both datasets remain unverified. See [provenance](SOURCES.md) and the
[replacement-data contract](../docs/data-contract.md).
