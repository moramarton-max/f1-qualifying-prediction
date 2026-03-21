"""
Baseline: predict qualifying order by ranking drivers on their
fastest-lap median from the last available practice session.

No ML — just "whoever was fastest in practice starts at the front."
Evaluated with the same CV structure as the trained model so results
are directly comparable.

Usage:
    python scripts/baseline_practice_rank.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data.loader import load_processed
from src.models.evaluate import evaluate_all
from src.utils.logging_config import get_logger

logger = get_logger("baseline")

# Session priority: use the latest available session's lap time
STANDARD_SESSION_PRIORITY = ["LapTime_median_FP3", "LapTime_median_FP2", "LapTime_median_FP1"]
SPRINT_SESSION_PRIORITY   = ["LapTime_median_SQ",  "LapTime_median_FP1"]


def best_practice_time(row: pd.Series) -> float:
    """Return the lap time from the last available practice session for this driver."""
    for col in STANDARD_SESSION_PRIORITY + SPRINT_SESSION_PRIORITY:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return float("nan")


def evaluate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Year, Round) group, rank drivers by practice lap time
    and compute metrics vs actual QualiPos.
    Returns a DataFrame with one row per weekend.
    """
    records = []
    for (year, rnd), group in df.groupby(["Year", "Round"]):
        group = group.copy()
        group["practice_time"] = group.apply(best_practice_time, axis=1)

        # Drop drivers with no practice time at all
        valid = group.dropna(subset=["practice_time", "QualiPos"])
        if len(valid) < 3:
            continue

        # Rank by practice time ascending (lower = better)
        pred_ranks = valid["practice_time"].rank(method="min").values
        true_pos   = valid["QualiPos"].values

        metrics = evaluate_all(true_pos, -valid["practice_time"].values)  # negate: lower time = higher score
        metrics["Year"]      = year
        metrics["Round"]     = rnd
        metrics["n_drivers"] = len(valid)
        records.append(metrics)

    return pd.DataFrame(records)


def main():
    df = load_processed("Q")
    if df is None:
        logger.error("No processed data found. Run build_features.py first.")
        sys.exit(1)

    df = df[df["QualiPos"].notna()].copy()

    splits = {
        "train": df[df["Split"] == "train"],
        "test":  df[df["Split"] == "test"],
    }

    for split_name, split_df in splits.items():
        if split_df.empty:
            continue

        print(f"\n{'='*60}")
        print(f"  Baseline (last practice session rank) -- {split_name.upper()} split")
        print(f"{'='*60}")

        weekend_metrics = evaluate_baseline(split_df)

        # Per-season summary to match CV structure
        for season, season_df in weekend_metrics.groupby("Year"):
            print(
                f"  {season}  "
                f"Spearman={season_df['spearman'].mean():+.3f}  "
                f"Top5={season_df['top5_accuracy'].mean():.2f}  "
                f"MAPE_pos={season_df['mape_pos'].mean():.2f}  "
                f"P1={season_df['p1_hit_rate'].mean():.0%}  "
                f"({len(season_df)} rounds)"
            )

        print("-"*60)
        print(
            f"  MEAN  "
            f"Spearman={weekend_metrics['spearman'].mean():+.3f}  "
            f"Top5={weekend_metrics['top5_accuracy'].mean():.2f}  "
            f"MAPE_pos={weekend_metrics['mape_pos'].mean():.2f}  "
            f"P1={weekend_metrics['p1_hit_rate'].mean():.0%}"
        )
        print("="*60)


if __name__ == "__main__":
    main()
