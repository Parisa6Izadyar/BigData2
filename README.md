# Arno River Level Forecasting

An academic time-series project evaluating daily hydrometry forecasts with weather covariates. The workflow compares persistence, 7-day and 365-day seasonal naive forecasts with an ARMA model using annual Fourier terms and a weather-augmented SARIMAX model.

[CI checks](https://github.com/Parisa6Izadyar/BigData2/actions/workflows/tests.yml) · [Methods](docs/methodology.md) · [Validation record](docs/validation.md)

## Input data

The analysis expects `data/RIVER_TRAIN_DATASET_A1.xlsx`, with these columns:

| Column | Meaning |
| --- | --- |
| `Date` | Daily observation date |
| `Actual_Hydrometry` | River-level target in the source file's units |
| `Mean_Temp` | Prepared temperature covariate |
| `Mean_Rainfall` | Prepared rainfall covariate |

The original workbook is not included in the repository or its history. The dataset's source, units and preparation steps need to accompany the workbook before a real-data benchmark can be verified. [Input notes](data/README.md) describe the accepted format. Existing images alone do not establish reproducible model performance.

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

Each prediction spans the full horizon without receiving observed river levels from within that horizon. MAE and RMSE retain the hydrometry column's source units; no percentage accuracy is reported. Nonconverged statistical fits are reported rather than hidden.

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

Historical figures are preserved under [docs/legacy_figures](docs/legacy_figures/README.md). LSTM and XGBoost training code is absent from the original repository; those images are not used as evidence for the current model comparison.
