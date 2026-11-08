# Provenance and source limits

| Current input | Attribution | Status |
|---|---|---|
| Committed farm-gate CSV | Labeled as a Shrimp Insights portal download in the existing repository | Unverified provenance, 2026 weekly observations, no publication vintages |
| Bundled monthly CSV | Existing illustrative sample | Demo data; not verified regional export observations |
| Old yearly CSV | Prototype illustrative series, preserved under `experiments/legacy/backend/data` | Excluded from v3 training and serving |

Attribution does not independently establish origin, accuracy, license or source
reputation. The original committed inputs are preserved unchanged. Canonical
derivatives record deterministic data hashes and quality counts. No missing
targets are filled, and duplicate series/date keys fail validation.

The previous project listed IMF/FRED, UN Comtrade, NASA POWER, WAHIS, FAO and
macro data as intended external inputs. Those scrapers are archived, excluded
from training and no longer scheduled. Reintegrating a source requires a review
of its geography/price basis, measurement definition, license, transformations,
availability dates and revisions. Regional export unit values and a global
commodity benchmark are different targets and must remain distinguishable.

Future verified files should carry the provenance described in the
[observation contract](../docs/data-contract.md). Keep licensed or private source
files out of Git unless redistribution is authorized.
