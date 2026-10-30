# Historical figures

These images were supplied with the original project. They are retained for reference and use a different or undocumented evaluation setup.

The original repository notebook contains SARIMAX code and lists LSTM under its next steps. A later supplied notebook includes LSTM imports but no model definition, training loop or prediction code. The title of `sarimax_lstm_xgboost.png` mentions LSTM, while its legend labels only training data, test data, SARIMAX and XGBoost. The LSTM training code and configuration are unavailable in these materials.

The later notebook also contains XGBoost training, with test-label early stopping and observed held-out target lags. Those results use a different evaluation setup from a fixed-origin forecast. The old full-series figure combines in-sample fitted values and a held-out forecast; those portions should not be interpreted as one independent test.

Use plots and metrics generated together by the current workflow for new comparisons. The recorded statistical-model benchmark is documented in the main README. The workbook's upstream preparation remains unspecified.
