# Validation record

## Local checks

On 30 October 2026, all 20 automated tests passed with Python 3.12.14 and the pinned requirements. The synthetic command-line run fitted all five predefined candidates, evaluated a 90-day validation interval and a separate 90-day test horizon, and saved predictions, split dates, metrics and a plot. Synthetic scores are not Arno River performance measurements.

The suite checks chronological partition boundaries, daily-date handling, missing/duplicate target dates, train-only weather statistics, leap-day fallback, multistep baselines, forecast invariance to held-out data, conditional observed-weather behavior, convergence failures, validation-based model selection, metric alignment and command-line outputs.

## Notebook and CI

The source notebook calls the same implementation as the CLI. CI installs the pinned packages, runs the tests and executes all nine notebook cells on the synthetic fixture, using a workspace-local kernel specification. It requires no external dataset or service credentials.

A local Jupyter kernel launch in the restricted execution environment was blocked by its socket permissions. Notebook execution therefore needs to be verified by the hosted CI run; the CLI and numerical tests ran locally. A user-run hosted Codespaces session has not yet been verified.

## Real-data evaluation

The original workbook is absent from the repository and its history. No new real-data accuracy or improvement claim is made. The notebook's saved historical outputs and images do not establish a reproducible benchmark. The observation source, units, upstream preprocessing and redistribution terms need to be documented with the input before a measured result can be added.
