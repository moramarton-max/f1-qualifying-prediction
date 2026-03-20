"""
Generate qualifying predictions for an in-progress race weekend.

Usage:
    python scripts/predict_weekend.py --year 2026 --round 5 --target Q
    python scripts/predict_weekend.py --year 2026 --round 5 --target Q --sessions FP1 FP2
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.build_dataset import build_weekend_features
from src.models.predict import predict
from src.utils.logging_config import get_logger

logger = get_logger("predict_weekend")


def main():
    parser = argparse.ArgumentParser(description="Predict qualifying results for a live weekend.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--sessions", nargs="+", default=None,
                        help="Which sessions are available (e.g. FP1 FP2). Default: all preceding sessions.")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--raw-dir", default="data/raw/")
    parser.add_argument("--models-dir", default="models/artifacts/")
    args = parser.parse_args()

    features = build_weekend_features(
        args.year,
        args.round_number,
        target=args.target,
        available_sessions=args.sessions,
        cache_dir=args.cache_dir,
        raw_dir=args.raw_dir,
    )

    if features is None or features.empty:
        logger.error("Could not build features for this weekend.")
        sys.exit(1)

    result = predict(features, models_dir=args.models_dir)

    print(f"\n{'='*45}")
    print(f"  Predicted {args.target} results — {args.year} Round {args.round_number}")
    print(f"{'='*45}")
    for _, row in result.iterrows():
        print(f"  P{row['PredictedRank']:>2}  {row['Driver']:<4}  {row['Team']}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
