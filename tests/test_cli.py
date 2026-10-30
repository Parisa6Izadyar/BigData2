import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_real_models_and_writes_consistent_outputs(tmp_path):
    result = subprocess.run([sys.executable, "-m", "arno_forecasting", "demo", "--output", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "metrics.json").read_text())
    assert manifest["synthetic"] is True
    assert "sarimax_weather" in manifest["test_metrics"]
    assert manifest["partitions"]["test"]["days"] == 90
    assert (tmp_path / "forecast_comparison.png").stat().st_size > 1000
    assert len((tmp_path / "test_predictions.csv").read_text().splitlines()) == 91


def test_missing_data_file_returns_actionable_error(tmp_path):
    result = subprocess.run([sys.executable, "-m", "arno_forecasting", "run", "--input", str(tmp_path / "missing.xlsx")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Input file not found" in result.stderr
    assert "Traceback" not in result.stderr
