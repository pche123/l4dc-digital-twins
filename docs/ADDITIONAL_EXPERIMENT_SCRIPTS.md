# Additional L4DC experiment sources

These four scripts were present in the latest code archive. They have now been added under `code/`. Their source bytes were checked against the uploaded archive.

| Script | Purpose | Invocation | Expected output |
| --- | --- | --- | --- |
| [dose_noise_law.py](../code/dose_noise_law.py) | Noise-to-bias optimal exposure law, under equation-error and sensor-noise designs | `python code/dose_noise_law.py .runs/dose` | `dose_noise_law.csv` |
| [dose_and_defect_check.py](../code/dose_and_defect_check.py) | Randomized checks of dose-bias equivalence, verifier movement bounds, and optimal dosing formulas | `python code/dose_and_defect_check.py` | Console checks |
| [lorenz_neural_reconstruction.py](../code/lorenz_neural_reconstruction.py) | Independent one-step Lorenz neural study with positive/negative wing holdouts | `python code/lorenz_neural_reconstruction.py harness .runs/lorenz_reconstruction` | `lorenz_documented_protocol_results.csv`, `lorenz_documented_protocol_summary.csv` |
| [twin_frozen_study.py](../code/twin_frozen_study.py) | Frozen 40-seed maintenance-twin comparison, matched-input controls and self-label check | `python code/twin_frozen_study.py .runs/twin_frozen` | `digital_twin_results.csv`, `digital_twin_gains.csv`, `digital_twin_self_label.csv`, `digital_twin_summary.csv` |

Create output directories before running. These experiments can be computationally expensive; use a dedicated environment with dependencies from `requirements.txt`. The four sources were syntax-checked. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the existing distinctions between figure recreation, archived arithmetic checks, and independent reproduction from raw inputs.

The current submission separates the main PDF and supplementary PDF on OpenReview. 
