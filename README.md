# Arno River Level Forecasting

An academic team project evaluating daily hydrometry forecasts with weather covariates. The workflow compares persistence, 7-day and 365-day seasonal naive forecasts with an ARMA model using annual Fourier terms and a weather-augmented SARIMAX model. A recursive XGBoost candidate uses seven lags and calendar features under the same forecast protocol.

[CI checks](https://github.com/Parisa6Izadyar/BigData2/actions/workflows/tests.yml) · [Methods](docs/methodology.md) · [Validation record](docs/validation.md)

## Input data

The analysis expects `data/RIVER_TRAIN_DATASET_A1.xlsx`, with these columns:

| Column | Meaning |
| --- | --- |
| `Date` | Daily observation date |
| `Actual_Hydrometry` | River-level target in the source file's units |
| `Mean_Temp` | Prepared temperature covariate |
| `Mean_Rainfall` | Prepared rainfall covariate |

The supplied A1 workbook contains **6,249 daily observations from 25 June 2000 to 3 August 2017**, with no missing values or duplicate dates. A2 and A3 contain the same 1,283-row subset of A1; they are not additional independent observations. [Source inventory](data/source_inventory.json) records file hashes and coverage.

The workbook is not distributed here. Use an authorized local copy. Its provider, hydrometry units, upstream preparation and redistribution terms have not been supplied. The 177 zero target values are retained; no missing-value interpretation is assumed. [Input notes](data/README.md) describe the accepted format.

## Recorded A1 benchmark

The six-candidate comparison uses 5,519 training days, a separate 365-day validation interval and a 365-day test ending on 3 August 2017. Validation MAE selects SARIMAX before the test is scored.

| Model | Validation MAE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: |
| Persistence | 0.4225 | 0.3419 | 0.6134 |
| Seasonal naive 7 days | 0.4341 | 0.3584 | 0.6309 |
| Seasonal naive 365 days | 0.4424 | 0.4521 | 0.7040 |
| ARMA with annual Fourier terms | 0.3290 | 0.3343 | 0.4961 |
| Weather SARIMAX | **0.3289** | **0.3343** | **0.4961** |
| Recursive XGBoost | 0.3414 | 0.3445 | 0.5078 |

The selected model's test RMSE is **19.1% lower than persistence** (0.4961 versus 0.6134). The two statistical models have nearly identical errors, so this does not establish a material benefit from weather covariates. These errors use the source hydrometry units; they are not percentage accuracy. See [full-precision validation MAE and test metrics](results/benchmark/metrics.json).

The complete comparison was run in GitHub Codespaces with the local A1 workbook. SARIMAX had lower validation MAE and lower test MAE/RMSE than recursive XGBoost on this split. New run outputs go to `results/reproduced/`. This is a single historical split of a prepared input file; upstream preprocessing remains unspecified.

## Run in Codespaces

[Open in GitHub Codespaces](https://codespaces.new/Parisa6Izadyar/BigData2?ref=main)

Select the pull request branch when reviewing changes. The dev container uses Python 3.12 and installs the dependencies. Run the installation command explicitly if setup has not finished or the Codespace was created before changing branches.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m arno_forecasting demo
```

The demo uses deterministic synthetic data and checks the complete training/evaluation workflow. Its scores are not results on the Arno River. It produces `results/demo/metrics.json`, dated predictions, split dates and a forecast plot, ending with `Finished. Results: ...`.

After placing the original workbook in `data/`, run:

```bash
python -m arno_forecasting run
```

An explicit input path is also supported:

```bash
python -m arno_forecasting run --input /path/to/RIVER_TRAIN_DATASET_A1.xlsx
```

Real-data outputs go to `results/reproduced/`. The workbook, generated outputs and cached kernels are ignored by Git. No Google Colab upload or API key is required.

## Evaluation

The default protocol reserves the final 365 days for testing and the preceding 365 days for validation. At least 365 earlier training days are required. All candidate settings are fixed before evaluation; validation MAE selects the model, including the naive baselines. Candidates are then refitted on all pre-test observations and evaluated once at the held-out forecast origin.

Each prediction spans the full horizon without receiving observed river levels from within that horizon. MAE and RMSE retain the hydrometry column's source units; no percentage accuracy is reported. MAPE is omitted because the target contains zeros. Nonconverged statistical fits are reported rather than hidden.

Annual Fourier terms use a 365.25-day period with two harmonics. The earlier notebook used a seasonal period of 52 on daily data, which represents 52 days. The current formulation represents annual seasonality without fitting a large 365-state seasonal component.

The default weather mode constructs future temperature and rainfall from calendar-day averages fitted only to past observations. Imputation and scaling also fit past data only. To evaluate the separate, conditional scenario in which future weather is supplied from observations:

```bash
python -m arno_forecasting run --weather-mode observed --output results/reproduced/observed_weather
```

This mode is a conditional hindcast. Its scores must not be presented as forecasts made without knowledge of future weather. Neither mode establishes flood-warning performance or forecasts water discharge.

## Notebook

`notebooks/H&S_SARIMAX.ipynb` uses the same tested functions as the command-line run. Open it in the Python environment installed above, or execute all its cells with:

```bash
python scripts/execute_notebook.py
```

For an installation check independent of the original workbook:

```bash
python scripts/execute_notebook.py --demo
```

The executor writes an executed copy under `results/notebook_check/` and leaves the source notebook free of stale output. Set `ARNO_DATA` to select another workbook in the notebook.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m arno_forecasting demo
```

On Windows, activate with `.venv\Scripts\activate`; Windows is not covered by CI. The earlier filename `requirement.txt` remains as an alias that installs `requirements.txt`.

## Repository

- `arno_forecasting/`: input validation, chronological splits, weather features, model fitting and evaluation.
- `notebooks/`: worked analysis using the same implementation.
- `tests/`: regression and integration checks.
- `docs/`: methods, validation and historical figures.
- `scripts/execute_notebook.py`: notebook execution with the selected Python interpreter.

Historical figures are preserved under [docs/legacy_figures](docs/legacy_figures/README.md). The available notebooks include SARIMAX and XGBoost workflows. The LSTM implementation and training configuration were not included with the archived materials, so LSTM is outside the reproducible comparison.
