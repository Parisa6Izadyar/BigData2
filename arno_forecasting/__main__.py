import argparse
from pathlib import Path
import sys

from .data import load_data, synthetic_data
from .experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate chronological river-level forecasts against naive baselines.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, default=ROOT / "data/RIVER_TRAIN_DATASET_A1.xlsx")
    run.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    run.add_argument("--horizon", type=int, default=365)
    run.add_argument("--validation-days", type=int, default=365)
    run.add_argument("--weather-mode", choices=["climatology", "observed"], default="climatology")
    run.add_argument("--maxiter", type=int, default=200)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, default=ROOT / "results/demo")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            frame = synthetic_data()
            return run_experiment(frame, args.output, {"source": "Deterministic synthetic installation fixture", "seed": 42},
                                  horizon=90, validation_days=90, synthetic=True)
        frame, quality = load_data(args.input)
        return run_experiment(frame, args.output, quality, args.horizon, args.validation_days, args.weather_mode, args.maxiter)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return None


if __name__ == "__main__":
    raise SystemExit(0 if main() is not None else 1)
