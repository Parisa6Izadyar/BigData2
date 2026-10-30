# Validation record

## Automated checks

On 30 October 2026, the current suite ran with 23 tests passed and one source-workbook check skipped when A1 was unavailable. There are 24 tests in total. The workbook-specific test verifies the supplied A1 SHA-256, coverage, zeros and chronological partitions when that file is present. A skipped source check is not a successful real-data execution.

The suite checks daily dates, duplicate and missing targets, finite values, chronological boundaries, training-only weather statistics, leap-day fallback, fixed-origin baselines, forecast invariance to held-out data, conditional weather behavior, convergence failures, validation-based selection, metric alignment and CLI outputs. XGBoost checks cover training rows, absence of test-label early stopping and recursive use of predicted target values.

## Notebook and CI

CI installs the pinned requirements, runs the tests and executes all nine notebook cells on deterministic synthetic data. A real-workbook notebook step runs only when the A1 workbook is present. Synthetic scores check software behavior and are not Arno performance measurements. A user-run Codespaces verification of the expanded six-candidate comparison remains pending.

## Recorded A1 evaluation

The supplied workbook has 6,249 complete daily rows from 25 June 2000 to 3 August 2017. The recorded statistical-model comparison reserves 5,519 training days, 365 validation days and 365 test days. Validation MAE selects weather SARIMAX, whose test RMSE is 0.4960966022 versus persistence at 0.6133875306, a 19.1% reduction. Fourier ARMA is nearly identical at 0.4961214549; a large weather-covariate improvement is not supported.

Aggregate metrics are stored under `results/benchmark/`. The full workbook, dated predictions and observed-series plots are not distributed. Re-run the expanded workflow with an authorized local A1 copy before final review. Provider, source units, earlier imputation and redistribution terms remain unspecified.
