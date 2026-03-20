"""
Train the qualifying prediction model from processed features.

Usage:
    python scripts/train_model.py --target Q
    python scripts/train_model.py --target Q --trials 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_processed
from src.models.train import train
from src.utils.logging_config import get_logger

logger = get_logger("train_model")


def main():
    parser = argparse.ArgumentParser(description="Train qualifying prediction model.")
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials per fold")
    parser.add_argument("--processed-dir", default="data/processed/")
    parser.add_argument("--models-dir", default="models/artifacts/")
    args = parser.parse_args()

    df = load_processed(args.target, args.processed_dir)
    if df is None:
        logger.error(f"No processed data found for target '{args.target}'. Run build_features.py first.")
        sys.exit(1)

    if "QualiPos" not in df.columns:
        logger.error("Column 'QualiPos' not found. Ensure your feature table includes actual qualifying results.")
        sys.exit(1)

    # Only train on the training split — never touch the holdout
    if "Split" in df.columns:
        test_rows = (df["Split"] == "test").sum()
        df = df[df["Split"] == "train"].reset_index(drop=True)
        logger.info(f"Train split: {len(df)} rows  |  Holdout (excluded): {test_rows} rows")
    else:
        logger.info(f"Loaded feature table: {df.shape}")

    train(
        df,
        target_col="QualiPos",
        n_optuna_trials=args.trials,
        models_dir=args.models_dir,
        save=True,
    )


if __name__ == "__main__":
    main()
