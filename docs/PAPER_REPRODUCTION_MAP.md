# Paper-to-code and data reproduction map

This is the navigation index for **When Do Synthetic Trajectories Count as Evidence?** (L4DC 2027, v25). Use the figure and table numbers in the **current** `main.tex` and `supplement.tex`, not labels from earlier drafts.

**Important distinction:** (1) replotting figures from `data/*.csv` checks presentation; (2) recomputing statistics from archived CSVs checks reported arithmetic; (3) rerunning experiment code from the original inputs checks an experimental pipeline. They are not equivalent. The archived `data/` files are never to be overwritten. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for limitations.

## Setup and safe experiment reruns

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/check_release.py --files
python scripts/check_release.py --math
python scripts/reproduce.py --list
python scripts/reproduce.py --group lorenz
python scripts/reproduce.py --group duffing
python scripts/reproduce.py --group verifier
python scripts/reproduce.py --group neural
python scripts/reproduce.py --group theory
python scripts/reproduce.py --group twin
python scripts/reproduce.py --group coverage
python scripts/reproduce.py --group closedloop
python scripts/reproduce.py --group derived
```

Unless `--out` is supplied, rerun results go under `.runs/reproduction/` and **not** into the archived `data/` directory. The `derived` group may seed missing prerequisites from the archived CSVs; it is *postprocessing*, not a fully independent rerun. Some historical twin outputs cannot be regenerated from original implementations (see the reproducibility caveats).

## Main-paper figures

| Paper item | Tracked graphic | Plot script | Archived inputs used by the plotting script | Relevant experiment command |
|---|---|---|---|---|
| **Figure 1**: proposed physics/symmetry, symbolic screen, learner | `figures/overview.pdf` | `python code/make_hero.py` | `figures/physical_system_strip.png`; simulated Lorenz drawn inside the script | Illustration, **not** experimental evidence |
| **Figure 2**: learner responses and noise | `figures/synthetic_comparisons.pdf` | `python code/make_figures.py` | `data/model_class_results.csv`, `data/noise_value_sweep.csv` | `--group lorenz`; `--group earlier` for the noise-value study |
| **Figure 3**: Lorenz gate and separate MLP | `figures/symmetry_verification.pdf` | `python code/make_figures.py` | `data/verify_main_defects.csv`, `data/verify_main_arms.csv`, `data/mlp_lorenz_fresh.csv` | `--group verifier`, `--group neural` |
| **Figure 4**: maintenance twin self-label / law / screening | `figures/twin_experiments.pdf` | `python code/make_figures.py` | `data/digital_twin_self_label.csv`, `data/digital_twin_gains.csv`, `data/twin_screen_laws.csv` | `--group twin` **only covers screening/exposure**; original generators of the historical `digital_twin_*.csv` files are missing |
| **Figure 5**: gate risk, value, interval coverage | `figures/risk_and_coverage.pdf` | `python code/make_fig_limits.py` | `data/gate_riskcov.csv`, `data/twin_screen_arms.csv`, `data/twin_screen_laws.csv`, `data/coverage_twin.csv` | `--group derived`, `--group twin`, `--group coverage` |

Plot scripts read the **archived** `data/` inputs and write into `figures/`. They can overwrite the tracked figure PDF/PNG; commit or preserve the originals before rebuilding. Rebuilding a plot is not an independent experiment reproduction.

## Main-paper tables and quantitative claims

| Paper item | Source | How to check |
|---|---|---|
| **Table 1**: duplicate-adjusted Lorenz / Duffing gains | `data/model_class_results.csv`, `data/duffing_results.csv` | `python scripts/reproduce.py --group lorenz` and `--group duffing`; then compare fresh CSVs with archived versions |
| **Table 2**: what can be claimed under different information sources | Conceptual recommendations derived from Sections 2–6, **not** a data table | Read theory and scope assumptions; no experiment script |
| Lorenz gate decisions / MLP effect | `data/verify_main_defects.csv`, `data/verify_main_arms.csv`, `data/mlp_lorenz_fresh.csv` | `--group verifier`, `--group neural` |
| Twin degradation, wrong-law gain and dose-response | `data/digital_twin_self_label.csv`, `data/digital_twin_gains.csv`, `data/twin_exposure.csv` | `--group twin` regenerates the exposure series, **not all historical twin files** |
| Nominal 90% interval coverage / closed-form law | `data/coverage_twin.csv` and Proposition on naive overconfidence | `--group coverage`; `python scripts/check_release.py --math` for included mathematical checks |
| Gate risk–coverage / conditional failure | `data/gate_riskcov.csv` | `--group derived` (uses archived prerequisites when missing); inspect setting-level risk, not just pooled totals |

## Supplementary material: figure and table families

| Supplement items | Data / experiment | Rebuilding or inspecting |
|---|---|---|
| Duffing symmetry illustration and screening plot | `data/duf_lownoise_defects.csv`, `data/verify_main_defects.csv` | `python code/make_figures.py`; `--group verifier` |
| EIV, finite-sample and selective-safety numerical checks | `data/eiv_check.csv`, `data/finite_bound_check.csv`, `data/selective_safety_check.csv` | `--group theory` |
| All-seed Lorenz/Duffing and fresh neural tables | `data/model_class_results.csv`, `data/duffing_results.csv`, `data/mlp_lorenz_fresh.csv` | `--group lorenz`, `--group duffing`, `--group neural` |
| Gate comparisons, false-reflection interventions, risk–coverage | `data/verify_main_defects.csv`, `data/verify_tilt_defects.csv`, `data/duf_lownoise_defects.csv`, `data/control_lv_defects.csv`, `data/gate_riskcov.csv` | `--group verifier`, `--group derived`; `python code/gate_summary.py data/verify_main_defects.csv` |
| Real Silverbox and direct-imposition tests | `data/silverbox_gate.csv`, `data/silverbox_results.csv`, `data/silverbox_imposed.csv` | `python scripts/run_real_benchmarks.py --dataset silverbox --execute` |
| Real EMPS friction / symmetry | `data/emps_gate.csv`, `data/emps_results.csv` | `--dataset emps --execute`; `python code/emps_summary.py data` |
| Real Wiener–Hammerstein | `data/wh_gate.csv`, `data/wh_results.csv` | `--dataset wiener-hammerstein --execute` |
| Real Cascaded Tanks | `data/tanks_laws.csv`, `data/tanks_results.csv` | `--dataset tanks --execute`; `python code/tanks_summary.py data` |
| Maintenance-twin law screen / exposure plot | `data/twin_screen_laws.csv`, `data/twin_screen_arms.csv`, `data/twin_exposure.csv` | `--group twin`, `python code/make_figures.py` |
| Bayesian twin interpretation | `data/bayes_view.csv` | `code/bayes_view.py` (inspect its usage before execution) |
| Closed-loop LQR table | `data/closed_loop_twin.csv` | `--group closedloop` (**stable-run interpretation is noted in REPRODUCIBILITY.md**) |
| Risk–coverage and confidence tables | `data/gate_riskcov.csv`, `data/coverage_twin.csv` | `--group derived`, `--group coverage` |

The supplementary EMPS illustration is `figures/emps_illustration.pdf`, generated by `python code/make_emps_figure.py` using `figures/emps_illustration_source.jpg`. The maintenance-twin illustration `figures/digital_twin_illustration.pdf` is illustrative artwork, **not a simulation output**. The dose-response supplement figure `figures/exposure_study.pdf` is produced by `python code/make_figures.py` from `data/twin_exposure.csv`.

## Real nonlinear benchmark raw files

| Benchmark | Exact filename(s) expected by this repository | Source |
|---|---|---|
| Silverbox | `SNLS80mV.csv` | https://www.nonlinearbenchmark.org/benchmarks/silverbox |
| EMPS | `DATA_EMPS.mat`, `DATA_EMPS_PULSES.mat` | https://www.nonlinearbenchmark.org/benchmarks/emps |
| Wiener–Hammerstein | `WienerHammerBenchmark.csv` | https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein |
| Cascaded Tanks | `dataBenchmark.csv` | https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks |

To identify links without downloading: `python scripts/fetch_benchmarks.py`. To retrieve the first three official archives using the helper: `python -m pip install -r requirements-download-optional.txt && python scripts/fetch_benchmarks.py --dataset all --download`. Fetch Cascaded Tanks manually from https://doi.org/10.4121/12960104. Verify file contents and the historical split/normalization *before* claiming equivalence.

```bash
python scripts/run_real_benchmarks.py --dataset all          # dry run: shows missing inputs
python scripts/run_real_benchmarks.py --dataset all --execute
```

Results are written to `.runs/real/`. Compare each generated CSV against the appropriate `data/*.csv` using `python scripts/compare_csv.py --help` for supported arguments; do not overwrite the archived `data/` files. Source downloads may be temporarily unavailable.

**Reproduction boundary:** raw external recordings are not included in the repository, and several historical digital-twin generators are unavailable. This index identifies scripts and archived numerical sources; it does **not** claim every result is independently reproducible today.
