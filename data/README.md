# Input workbook

Place `RIVER_TRAIN_DATASET_A1.xlsx` here, or pass another CSV/XLSX path with `--input`. The required columns are `Date`, `Actual_Hydrometry`, `Mean_Temp` and `Mean_Rainfall`.

Dates may be native Excel dates, ISO `YYYY-MM-DD` strings or `DD/MM/YYYY` strings. Mixed formats, duplicate dates, infinite values and missing target days are rejected. Weather gaps are filled from training-only medians. Zero river levels are retained as observations; they are not automatically treated as missing.

Before publishing measured results, document the data provider, observation period, measurement units, redistribution terms and any earlier aggregation or imputation. The old notebook reads a prepared workbook and does not contain its preparation code. The new workflow cannot establish how that earlier file was created.

The demo creates synthetic values in memory. It does not download, reconstruct or replace the missing Arno observations.
