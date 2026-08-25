# Experiments

This folder contains runnable experiment scripts that configure the model-building pipeline for specific optimisation runs.

## Ridge Hyperparameter Optimisation

Run from the repository root:

```bash
python src/experiments/linear_hp_optimisation.py
```

`linear_hp_optimisation.py` evaluates the `Linear` model, which is implemented as:

- `sklearn.preprocessing.StandardScaler`
- `sklearn.linear_model.Ridge`

The script tests 30 `alpha` values from `1e-5` to `1e5` with `np.logspace(-5, 5, 30)`. Each value is passed through `LinearModelConfig(alpha=...)`, into `ExperimentConfig.linear_model_config`, and then into the inner `Ridge` estimator.

The configured experiment uses cases `A`, `B`, and `C`, five repeated stratified splits, and these input features:

- `vibration_x`
- `vibration_y`
- `vibration_z`
- `speed`
- `vibration_magnitude`
- `score_mild`
- `score_standard`
- `score_strict`

During the run, the shared model-building pipeline logs averaged cross-validation metrics for each dataset/test-case and alpha. At the end, `log_ridge_optimisation_summary` logs:

- number of unique dataset/test-cases tested
- number of alpha-by-dataset model runs
- number of alpha values tested
- best MAE, including testcase id, alpha, and full metrics
- best macro-F1, including testcase id, alpha, and full metrics
- mean MAE and mean macro-F1 across all alpha-by-dataset runs

Metrics are logged to stdout but are not persisted to disk.
