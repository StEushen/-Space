# Axiomatic Tau-Space PPN

A project-focused fork of the Time-Series-Library scaffold for developing and evaluating **Axiomatic Tau-Space PPN**.

This repository is maintained for one goal:
- turn the tau-space idea into a reproducible, benchmarkable forecasting system.

## 1. Project Scope

This codebase centers on:
- Axiomatic intrinsic-time modeling (tau-space)
- Non-bootstrap horizon generation
- Global mapping + cross-adjustment forecast path
- Few-shot stress evaluation under TSLib-compatible settings

It does **not** aim to be a generic model zoo documentation. The base TSLib framework is used as infrastructure.

## 2. Core Method (Axiomatic Tau-Space)

The current PPN implementation follows:
1. Structural Evidence Encoding
2. Causal Monotonic Clock
3. Forecasting in Tau Space
4. Global Mapping and Cross Adjustment
5. Projection Back to Observation Time

In short:
- tau is treated as an intrinsic coordinate system, not a single handcrafted statistic.
- acceleration can be one evidence channel, but is not the definition of tau.

## 3. Axioms Implemented

The implementation targets these properties:
- Causality: tau depends only on past and current observations.
- Monotonicity: intrinsic time is strictly increasing.
- Reparameterization consistency: observation process is modeled through tau-space states.
- Dynamics simplification: tau-space dynamics are encouraged to be flatter.
- Predictive utility: gains must appear in forecasting metrics.
- Recoverability: tau-space representation must reconstruct useful observation content.
- Gauge invariance awareness: practical training focuses on order/structure, not absolute tau scale.

## 4. Key Files

- `models/ppn.py`: Axiomatic tau-space PPN model
- `exp/exp_long_term_forecasting.py`: training/validation logic and tau auxiliary losses
- `run.py`: CLI args and experiment entry
- `utils/print_args.py`: structured runtime argument printout
- `scripts/run_ppn_horizon_residual_fewshot.ps1`: few-shot benchmark launcher
- `scripts/run_ppn_tau_axiom_overnight.ps1`: long-run/resume launcher

## 5. Data and Compatibility

Experiments are run in TSLib-compatible format using local datasets, typically:
- `./dataset/ETT-small/ETTh1.csv` (and related ETT files)

This allows apples-to-apples comparison **if protocol is aligned**:
- same split and preprocessing
- same horizon/task settings
- same training budget

## 6. Quick Start (Windows PowerShell)

### 6.1 Environment

```powershell
conda activate py39env
```

### 6.2 Fast sanity run

```powershell
powershell -ExecutionPolicy Bypass -Command "& {
  .\scripts\run_ppn_horizon_residual_fewshot.ps1 \
    -Epochs 1 -Patience 1 -PredLen 96 \
    -Seeds @(42) -Ratios @(0.1) \
    -IncludeGatedResidual 0 -IncludeWarmupResidual 1 \
    -WarmupEpochs 1 -IncludeTauFieldVariant 0 \
    -Tag tau_axiom_smoke
}"
```

### 6.3 Overnight/resume run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ppn_tau_axiom_overnight.ps1 -Resume 1
```

## 7. Output Files

Each run writes three CSV artifacts in `logs/`:
- run-level: `... .csv`
- grouped summary: `... _summary.csv`
- ranking: `... _rank.csv`

Overnight orchestration log:
- `logs/tau_axiom_overnight_master.log`

## 8. Current Practical Guidance

Based on recent runs in this repository:
- `horizon_residual` and `horizon_residual_warmup` are usually better than `base`.
- ratio `0.05` is significantly less stable than `0.1`.
- phase scheduling is available but should be validated per setting.

## 9. Reproducibility Checklist

When reporting results, include:
- commit hash
- dataset root and file
- seeds and ratios
- epochs/patience
- pred_len
- exact script command

## 10. Acknowledgement

This project builds on the Time-Series-Library ecosystem for benchmark infrastructure.
All method-specific logic and experiments here are focused on Axiomatic Tau-Space PPN evolution.
