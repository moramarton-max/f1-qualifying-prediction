"""
Evaluate the trained model on the held-out test set.

Test set: 2025 R21-24 + 2026 R1-2 (defined in build_dataset.TEST_WEEKENDS)

Usage:
    python scripts/evaluate_holdout.py
    python scripts/evaluate_holdout.py --target Q --models-dir models/artifacts/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import load_processed
from src.models.evaluate import evaluate_all
from src.models.predict import load_latest_artifact
from src.utils.logging_config import get_logger

logger = get_logger("evaluate_holdout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--processed-dir", default="data/processed/")
    parser.add_argument("--models-dir", default="models/artifacts/")
    args = parser.parse_args()

    df = load_processed(args.target, args.processed_dir)
    if df is None:
        logger.error("No processed data found. Run build_features.py first.")
        sys.exit(1)

    if "Split" not in df.columns:
        logger.error("No 'Split' column found. Rebuild the feature table.")
        sys.exit(1)

    test = df[df["Split"] == "test"].copy()
    if test.empty:
        logger.error("Test split is empty.")
        sys.exit(1)

    if test["QualiPos"].isna().any():
        missing = test[test["QualiPos"].isna()][["Year", "Round"]].drop_duplicates()
        logger.warning(f"Dropping rows with missing QualiPos:\n{missing.to_string()}")
        test = test[test["QualiPos"].notna()]

    artifact = load_latest_artifact(args.models_dir)
    feature_cols = artifact["feature_cols"]
    model = artifact["model"]

    results = []
    for (year, rnd), group in test.groupby(["Year", "Round"]):
        X = group.reindex(columns=feature_cols)
        scores = model.predict(X)
        metrics = evaluate_all(group["QualiPos"].values, scores)
        metrics["Year"] = year
        metrics["Round"] = rnd
        metrics["n_drivers"] = len(group)
        results.append(metrics)
        logger.info(
            f"  {year} R{rnd:>2}  "
            f"Spearman={metrics['spearman']:+.3f}  "
            f"Top5={metrics['top5_accuracy']:.2f}  "
            f"MAPE={metrics['mape_pos']:.2f}  "
            f"P1={'Y' if metrics['p1_hit_rate'] else 'N'}"
        )

    summary = pd.DataFrame(results)
    print("\n" + "=" * 55)
    print("  Holdout evaluation summary")
    print("=" * 55)
    print(summary[["Year", "Round", "spearman", "top5_accuracy", "mape_pos", "p1_hit_rate"]].to_string(index=False))
    print("-" * 55)
    print(f"  Mean  Spearman : {summary['spearman'].mean():+.3f}")
    print(f"  Mean  Top-5    : {summary['top5_accuracy'].mean():.2f}")
    print(f"  Mean  MAPE     : {summary['mape_pos'].mean():.2f}")
    print(f"  P1 hit rate    : {summary['p1_hit_rate'].mean():.0%}")
    print("=" * 55)


if __name__ == "__main__":
    main()
