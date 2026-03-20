"""
Build the processed feature table from all raw data on disk.

Usage:
    python scripts/build_features.py --seasons 2022 2023 2024 --target Q
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import list_available_weekends
from src.features.build_dataset import build_dataset
from src.utils.logging_config import get_logger

logger = get_logger("build_features")


def main():
    parser = argparse.ArgumentParser(description="Build feature table from raw session data.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--target", default="Q", choices=["Q", "SQ"])
    parser.add_argument("--raw-dir", default="data/raw/")
    parser.add_argument("--processed-dir", default="data/processed/")
    args = parser.parse_args()

    available = list_available_weekends(args.raw_dir)
    weekends = sorted(set(
        (year, rnd)
        for year, rnd, session in available
        if year in args.seasons
    ))

    logger.info(f"Found {len(weekends)} weekends for seasons {args.seasons}")

    df = build_dataset(
        weekends,
        target=args.target,
        raw_dir=args.raw_dir,
        save=True,
        processed_dir=args.processed_dir,
    )

    logger.info(f"Feature table shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
