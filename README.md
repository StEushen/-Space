# PPN Model Theme Repository

This repository focuses on the PPN model theme for time-series forecasting, including:

- model architecture implementation
- intrinsic-time (tau-space) modeling
- global-fit then partition-correction training strategy
- reproducible scripts and experiment logs

## What Is Original In This Repo

This repository's original contribution is the PPN-centered mechanism and training workflow:

1. Learn a global fitting relationship on the whole dataset.
2. Apply partitioned correction against the global baseline.
3. Avoid stage-to-stage self-bootstrapping drift.

Correction term:

$$
L_{corr} = \operatorname{mean}(\operatorname{ReLU}(MSE_{partition} - MSE_{global\_baseline} + margin_\mu))
$$

## Latest Optimization Update (2026-04-11)

- Replaced direct single-step style extrapolation with tau-trajectory direct horizon generation in [models/PPN.py](models/PPN.py).
- Added PPN-specific intrinsic-time regularization flags (`--ppn_use_tau_loss`, `--ppn_lambda_tau`).
- Added optimized runnable script [scripts/run_ppn_latest_optimized.ps1](scripts/run_ppn_latest_optimized.ps1) (cosine LR + longer patience).

## Repository Structure

- [run.py](run.py): unified experiment entry.
- [exp/](exp): task-level training/evaluation pipelines.
- [data_provider/](data_provider): dataset factory and loaders.
- [models/](models): model implementations, including PPN and tau-related models.
- [layers/](layers): reusable architecture blocks.
- [utils/](utils): metrics, training tools, and utilities.
- [scripts/](scripts): reproducible scripts for benchmark tasks and PPN experiments.
- [experiments/](experiments): experiment configs and notes.
- [logs/](logs): result tables from completed runs.
- [paper/](paper): manuscript and figure artifacts.

## PPN/Tau Related Files

Core model and mechanism files:

- [models/PPN.py](models/PPN.py)
- [models/tau_mechanism.py](models/tau_mechanism.py)
- [models/TauOnly.py](models/TauOnly.py)
- [models/TauFusion.py](models/TauFusion.py)
- [models/TauClockFixedGRU.py](models/TauClockFixedGRU.py)

Representative scripts:

- [scripts/run_ppn_latest_optimized.ps1](scripts/run_ppn_latest_optimized.ps1)
- [scripts/run_final_tau_structure_prior_suite.ps1](scripts/run_final_tau_structure_prior_suite.ps1)
- [scripts/run_tauonly_robust_baseline_multiseed.ps1](scripts/run_tauonly_robust_baseline_multiseed.ps1)

Representative result summaries:

- [logs/tauonly_robust_baseline_v1_fixed_runs.csv](logs/tauonly_robust_baseline_v1_fixed_runs.csv)
- [logs/tauonly_robust_baseline_v1_fixed_rank.csv](logs/tauonly_robust_baseline_v1_fixed_rank.csv)
- [logs/patchtst_taufusion_small_v1_summary.csv](logs/patchtst_taufusion_small_v1_summary.csv)

## Quick Start

### 1. Environment

Use Python 3.9+ and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data

Place datasets under `./dataset` (legacy benchmark path) or `./datasets` (local managed path), based on your script config.

### 3. Run A PPN Experiment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ppn_latest_optimized.ps1
```

### 4. Run A Tau Baseline Multi-Seed Experiment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_tauonly_robust_baseline_multiseed.ps1 -Tag tauonly_robust_baseline_v1 -Epochs 5 -Ratio 0.1
```

## Acknowledgement and Upstream Credit

This repository is built on a benchmark scaffold derived from Time-Series-Library (TSLib).

- Upstream project: https://github.com/thuml/Time-Series-Library
- Original benchmark framework and many baseline model implementations are credited to TSLib contributors.

PPN-specific design, tau-mechanism extensions, scripts, and experiment organization in this repository are maintained in this project context.

## Citation

If you use this repository in research, please cite both:

1. The upstream TSLib papers (for benchmark framework and included baselines).
2. Your PPN/tau-mechanism manuscript and repository.

## License

See [LICENSE](LICENSE).
