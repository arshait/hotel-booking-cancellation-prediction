"""Train the hotel cancellation model and write reports.

Example:
    python train.py --data data/hotel_bookings.csv --model models/model.joblib --reports reports
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hotel_ml import run_experiment


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the hotel cancellation classifier.")
    parser.add_argument("--data", type=Path, default=root / "data" / "hotel_bookings.csv")
    parser.add_argument("--model", type=Path, default=root / "models" / "model.joblib")
    parser.add_argument("--reports", type=Path, default=root / "reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = run_experiment(args.data, args.model, args.reports)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    selected = summary["selected_model"]
    test_auc = summary["test"]["roc_auc"]
    print(f"Selected model: {selected}")
    print(f"Validation ROC AUC: {summary['validation']['roc_auc']:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")
    print(f"Operating threshold: {summary['threshold']:.2f}")
    print(f"Saved model: {args.model}")
    print(f"Saved metrics: {args.reports / 'metrics.json'}")


if __name__ == "__main__":
    main()
