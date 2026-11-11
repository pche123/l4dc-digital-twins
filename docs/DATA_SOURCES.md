# External benchmark inputs

The following are the original benchmark publisher pages and known download containers, not a guarantee that these are the exact archived raw versions or that the hosting is currently operational.

| Study | Expected input | Publisher URL | Known hosted archive |
|---|---|---|---|
| Silverbox | `SNLS80mV.csv` | https://www.nonlinearbenchmark.org/benchmarks/silverbox | https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view |
| EMPS | `DATA_EMPS.mat`, `DATA_EMPS_PULSES.mat` | https://www.nonlinearbenchmark.org/benchmarks/emps | https://drive.google.com/file/d/1zwoXYa9-3f8NQ0ohzmjpF7UxbNgRTHkS/view |
| Wiener-Hammerstein | `WienerHammerBenchmark.csv` | https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein | https://drive.google.com/file/d/16ipySVfKfxkwqWmbO9Z19-VjDoC2S6hx/view |
| Cascaded tanks | `dataBenchmark.csv` | https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks | https://doi.org/10.4121/12960104 |

The original publisher warns that some 4TU-hosted datasets are temporarily unavailable. Check dataset content, sampling, splits and preprocessing against each script before declaring the original numerical results reproduced. Public dataset distribution/repackaging may be governed by the original provider's licensing terms.

**Use:**

1. Retrieve datasets from the official pages.
2. Place the exact named files in `external_data/`.
3. Run each matching `code/*_exp.py` according to `REPRODUCIBILITY.md`; write results into `.runs/` and compare against `data/` with `scripts/compare_csv.py`.
4. Record original checksums and the version of any data loader.

The benchmark scripts expect specific input formats; a newer official standardized train/test split is not automatically the historical split used here.
