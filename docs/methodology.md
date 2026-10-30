# Forecasting protocol

## Chronological partitions

The input is sorted by date and checked against a daily calendar. Duplicate dates require an explicit source correction. Missing target values are not filled for evaluation. The default split holds out the final 365 days for testing and the previous 365 for validation, with at least 365 training days before both intervals.

Every candidate is first fitted on training data. Validation MAE selects the candidate; alphabetical model name resolves exact ties. A naive baseline can win. Candidates with valid validation fits are then refitted on training plus validation and forecast the held-out horizon. Test scores do not choose the model. If the selected candidate cannot produce a valid test fit, the run fails rather than selecting a different candidate using test outcomes.

This is a fixed-origin, multistep evaluation, not a sequence of one-day forecasts. The models do not update from held-out hydrometry within the prediction horizon. A single validation/test division does not establish stability across historical origins or future operating conditions.

## Models

| Candidate | Prediction rule |
| --- | --- |
| Persistence | Repeat the final observed training value |
| Seasonal naive, 7 days | Repeat the last seven training observations |
| Seasonal naive, 365 days | Repeat the last 365 training observations |
| ARMA with Fourier terms | ARMA(1,1), constant trend and two annual sine/cosine harmonics |
| Weather SARIMAX | The same ARMA/Fourier model with standardized temperature and rainfall |
| Recursive XGBoost | Six calendar features and seven lags each of hydrometry, temperature and rainfall |

Both statistical candidates use statsmodels' SARIMAX implementation with order `(1, 0, 1)`. Seasonal order is `(0, 0, 0, 0)` because annual seasonality is represented by explicit Fourier regressors with a 365.25-day period. These settings are a predefined comparison, not a claim of optimal parameter tuning. The 365-day naive baseline uses a fixed day lag rather than exact calendar-date matching around leap years.

Optimizer convergence is checked. Recorded warnings and unavailable candidates remain in the result manifest. No global warning suppression is applied.

## Recursive XGBoost

The recovered notebook used the test interval for early stopping and generated test target lags from observed held-out levels. That score cannot represent the same fixed-origin forecast as the statistical models. The current candidate fits only pre-origin rows, uses 1,000 trees with depth 5 and learning rate 0.03, and performs no early stopping. Later target lags contain the model's own predictions. Weather values follow the same climatology or explicitly observed scenario as SARIMAX. The settings are fixed, not presented as tuned optimum values.

## Weather availability

All weather medians, means, scales and calendar-day averages fit only the data preceding each forecast origin. Validation uses training statistics; the final test forecast uses statistics from training plus validation.

In default `climatology` mode, future weather depends on the future calendar dates and past weather only. An unseen leap day uses the average of February 28 and March 1 when both are available. Other unseen calendar days use training medians. Mutating held-out weather or hydrometry does not change these predictions.

In `observed` mode, future temperature and rainfall come from the held-out observations and must be complete. This is a conditional hindcast, preserving the assumption used in the original notebook. Its errors are not comparable to an operational forecast that lacks future weather. Weather forecasts from an external model would introduce another source of uncertainty that is not evaluated here.

## Outputs and provenance

Each run saves full-precision validation/test MAE and RMSE, selected model, candidate settings, package versions, input hash, date partitions, predictions and unavailable-model messages. The plot separates prior observations from the held-out forecast; in-sample fitted values are not labelled as an independent forecast.

Synthetic runs carry `synthetic: true` in their manifest and a synthetic label on plots. They check software behavior only. Real-data evaluation requires the original input and its preparation history. Training-only preprocessing here does not certify that an earlier workbook was prepared without using later observations.

## Input preparation

The A1 input contains 6,249 contiguous daily rows and 177 zero target observations. Those zeros remain in MAE and RMSE. MAPE would be undefined or misleading, so it is not reported. The first 30 prepared temperature and rainfall values are constant; their upstream preparation is unknown. Training-only feature fitting in this repository does not validate preprocessing performed before the workbook was supplied. A2 and A3 duplicate a subset of A1 and are not combined with it.

## Implementation references

- [statsmodels SARIMAX parameter definitions](https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/tsa/statespace/sarimax.py): the seasonal period counts observations, so 52 observations on daily data represent 52 days.
- [statsmodels out-of-sample exogenous validation](https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/tsa/statespace/mlemodel.py): regression-based forecasts require exogenous values for the requested horizon.
