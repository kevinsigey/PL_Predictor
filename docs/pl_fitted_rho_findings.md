# Fitted Dixon-Coles rho for ml_scoreline: negative result (rho stays 0)

Question: `ml_scoreline.predict_grid`/`predict_grids_batch` price the grid with `rho=0.0`. Does a fitted
low-score correlation rho improve the served model?

Method (`src/pl_predictor/evaluate/fitted_rho_walk_forward.py`, raw numbers in `docs/pl_fitted_rho_results.json`):
the repo's walk-forward folds (`prepare_folds()`, 5 validation seasons 2021-22 .. 2025-26, 1,900 matches), shipped
features and hyperparameters. Per fold, ONE scalar rho is fitted on the fold's TRAINING rows only: expanding inner
folds (train on seasons < j, predict season j, for every training season j >= 1) give out-of-fold lambdas, and
`ml_scoreline.fit_rho` maximises their exact-score likelihood (bounded search, [-0.3, 0.3]). The evaluation fold never
touches the fit. Paired bootstrap (2,000 resamples over matches, pooled folds) of per-match loss(rho=0) minus loss(fitted);
positive favours fitted.

Fitted rho by fold: -0.056, -0.021, -0.007, -0.011, -0.008 (small, negative, shrinking as training data grows).

| metric (lower better) | rho=0 | fitted | gain (rho0-fitted) | 95% CI |
|---|---|---|---|---|
| 1X2 RPS (primary) | 0.200517 | 0.200549 | -0.000032 | [-0.000118, +0.000049] |
| 1X2 log-loss | 0.978498 | 0.978525 | -0.000027 | [-0.000664, +0.000675] |
| O/U 2.5 log-loss | 0.682931 | 0.682909 | +0.000022 | [-0.000060, +0.000101] |
| BTTS log-loss | 0.687534 | 0.687581 | -0.000048 | [-0.000366, +0.000264] |
| exact-score log-loss | 2.978609 | 2.979214 | -0.000606 | [-0.001682, +0.000501] |

Draw calibration (pooled): actual draw rate 0.2389; mean predicted draw 0.2284 at rho=0, 0.2330 fitted. The fitted rho
does move the draw mass toward the observed rate, but not in a way any scored metric rewards.

Ship rule (primary-metric CI must exclude zero in the model's favour, no other metric worse): FAILED. The primary CI
straddles zero with a point estimate on the wrong side, and four of five metrics are nominally worse. Result: rho stays
0.0, no manifest field, serving unchanged. `rho`/`fit_rho` exist in `ml_scoreline` (default 0.0, byte-identical) only so
this experiment is reproducible. The draw/narrow-win split is therefore not the cause of the weak 50-60% live band
(37%, 17/46; n is small).

Unverified: only the 5 walk-forward seasons in the repo were tested; a position-dependent or lambda-dependent rho was not.

# Separate fix shipped in the same PR

`scoreline.predict_fixtures_batch` built X with `.reindex(...).fillna(0)` before `_apply_missing_value_encoding`, so that
function's unexpected-NaN warning could never fire on the main serving path. It now passes the reindexed frame straight to
`predict_many_from_rows` (which applies the shared helper). Tests: a NaN in a non-contract column warns on the batch path
(fails on origin/main), and grids for fully-populated rows are `assert_array_equal` to the old expression.
