# Existing-data baseline evaluation

Initial v3 diagnostic run, 9 November 2026 (Asia/Kolkata). Deterministic seed 42;
four-period recursive horizon; 90% nominal log-error interval. The full local
JSON reports include per-series and per-horizon metrics. CI can regenerate
reports using the manual `train.yml` workflow.

## Farm-gate observations

1,000 observations across 64 series and 22 dates, 4 May–28 September 2026.
Validation origins: 22 June, 29 June and 6 July. Calibration origin: 3 August.
Final test origin: 31 August. The target dates of these stages do not overlap.

| Candidate | Validation MAE, USD/kg | Validation RMSE |
|---|---:|---:|
| Latest price | 0.152 | 0.274 |
| Seasonal naïve | 0.152 | 0.274 |
| LightGBM | 0.198 | 0.332 |
| Histogram boosting | 0.198 | 0.339 |
| Damped drift | 0.213 | 0.353 |
| Ridge | 0.392 | 0.576 |

Selected: latest price. Seasonal naïve falls back to latest price because a full
weekly annual cycle is absent. Complex models did not improve validation here.
Final test: 165 matched targets, MAE **0.2007**, RMSE **0.3094**, WAPE **4.73%**,
bias **−0.0213** USD/kg. Actual interval coverage: **69.7%**; mean width: **0.5753**
USD/kg. Coverage is substantially below the nominal 90%.

## Illustrative monthly sample

144 observations across four series and 96 distinct dates (2016–2023). Most
non-global series are sparse. Validation uses six expanding origins; calibration
is April 2023 and final test is August 2023. Selected: LightGBM.

The final block contains only **four** matched targets. MAE is **0.0718** USD/kg
versus **0.1875** for persistence. These tiny, illustrative results establish no
real-market skill. There are insufficient per-horizon calibration samples for
finite 90% bounds, so interval limits and measured coverage are `null`.

## Interpretation

The improvement is in the evaluation and serving foundation; the available
farm-gate data does not establish an accuracy improvement over persistence.
Every current source remains unverified. Pooling correlated series and using
one calibration/test origin gives limited independent evidence. Longer,
publication-aware, reputable data is needed before relying on market forecasts.
