# Reproducibility and provenance

For an item-by-item map of **every main-paper figure/table, the archived CSV inputs, plotting code and experiment commands**, see [Paper reproduction map](PAPER_REPRODUCTION_MAP.md).

The repository contains the canonical v25 paper and **40 active archived experiment CSVs** under `data/`, with the relevant experiment scripts and frozen protocols available in the source tree. These original data have not been changed or regenerated merely to fit manuscript claims.

In the earlier local audit, 27 active result CSVs were independently rerun and matched to the archived outputs to within a numerical tolerance of 1e-10. The prior 43-file audit also counted one **superseded certificate** and two **older Lorenz protocol outputs**; those three historical files are excluded from this public release as they are not needed for the v25 manuscript.

The following classes of results remain **not independently reproduced from original inputs and original generating code**:

- **9 real-benchmark output CSVs:** original raw source files must be obtained and their versions and preprocessing checked.
- **4 historical frozen maintenance-twin output CSVs:** the original generating implementation was not recovered; related twin screening and exposure experiments are reproducible from their separate sources.

The original audit also included two older Lorenz outputs that are no longer distributed here; their absence does not change the active-paper reproduction status.

The local audit separately recomputed numerous reported table values from archived result CSVs. **Arithmetic verification against archived data does not substitute for executing the entire experimental pipeline.** A closed-loop table also requires a stable-run-only interpretation of some medians; the unmodified canonical paper may still need that reporting clarification before public release.

## Commands

```bash
python scripts/check_release.py --files
python scripts/check_release.py --math
python scripts/reproduce.py --list
python scripts/reproduce.py --group lorenz
python scripts/reproduce.py --group duffing
python scripts/reproduce.py --group verifier
python scripts/reproduce.py --group neural
python scripts/reproduce.py --group twin
python scripts/run_real_benchmarks.py --dataset all
```

Where original raw benchmark files are available, place them in `external_data/` and run `python scripts/run_real_benchmarks.py --dataset all --execute`. Compare the outputs with `data/` using `scripts/compare_csv.py` after checking the historical preprocessing and seed protocols.

The `paper.pdf` is the submitted manuscript artifact; compilation from `main.tex` should match its scientific text and figures but output bytes can vary between TeX distributions.
