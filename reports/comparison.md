# Comparison of the 4 experiments

Same algorithm (GradientBoostingRegressor), same athletes, same 80/20 split. Only the feature version
and the hyperparameters change.

| run_name   |   rmse |   mae |     r2 |   train_r2 |   mae_men |   mae_women |   fit_predict_seconds |   energy_kwh |   emissions_mg_co2e |
|:-----------|-------:|------:|-------:|-----------:|----------:|------------:|----------------------:|-------------:|--------------------:|
| v1_small   |  161.4 | 126.2 | 0.6593 |     0.6602 |     135.3 |      102.1  |                 0.963 |    3.164e-06 |               1.169 |
| v1_large   |  160.8 | 125.5 | 0.6618 |     0.6899 |     134.4 |      101.7  |                 4.565 |    1.494e-05 |               5.521 |
| v2_small   |  139.4 | 107.7 | 0.7457 |     0.7539 |     116.7 |       83.84 |                 1.695 |    5.7e-06   |               2.106 |
| v2_large   |  139.1 | 107.2 | 0.747  |     0.7838 |     116.2 |       83.23 |                 8.13  |    2.659e-05 |               9.824 |

Error columns are in lbs on the test athletes. `train_r2` minus `r2` shows overfitting.

## What changed the result?

- **Feature version (v2 vs v1):** RMSE changes by -13.6% with the small settings and -13.5% with the large settings.
- **Hyperparameters (large vs small):** RMSE changes by -0.4% on v1 and -0.3% on v2.
- **Carbon of the large settings vs small:** 4.7x on v1, 4.7x on v2.

## Plots

- `comparison_metrics.png`: error, carbon, and accuracy vs carbon.
- `comparison_pred_vs_actual.png`, `comparison_residuals.png`, `comparison_feature_importance.png`: the same plot for all 4 runs.
- Per-run plots: `reports/runs/<run>/`. Raw numbers: `reports/results.csv`. Carbon log: `reports/emissions/emissions.csv`.

Note: carbon figures are small and noisy (CodeCarbon estimates energy from CPU use over a few seconds). Compare them as ratios between runs, not as absolute values.
