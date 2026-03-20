"""Persistence helpers: save/load raw session DataFrames as parquet."""

import os
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _raw_path(raw_dir: str, year: int, round_number: int, session: str) -> str:
    return os.path.join(raw_dir, f"{year}_R{round_number:02d}_{session}.parquet")


def save_raw(df: pd.DataFrame, year: int, round_number: int, session: str, raw_dir: str = "data/raw/") -> None:
    os.makedirs(raw_dir, exist_ok=True)
    path = _raw_path(raw_dir, year, round_number, session)
    df.to_parquet(path, index=False)
    logger.info(f"Saved raw data: {path}")


def load_raw(year: int, round_number: int, session: str, raw_dir: str = "data/raw/") -> Optional[pd.DataFrame]:
    path = _raw_path(raw_dir, year, round_number, session)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def is_raw_available(year: int, round_number: int, session: str, raw_dir: str = "data/raw/") -> bool:
    return os.path.exists(_raw_path(raw_dir, year, round_number, session))


def list_available_weekends(raw_dir: str = "data/raw/") -> List[Tuple[int, int, str]]:
    """Return list of (year, round, session) tuples available on disk."""
    if not os.path.exists(raw_dir):
        return []
    results = []
    for fname in os.listdir(raw_dir):
        if not fname.endswith(".parquet"):
            continue
        try:
            parts = fname.replace(".parquet", "").split("_")
            year = int(parts[0])
            round_number = int(parts[1][1:])
            session = parts[2]
            results.append((year, round_number, session))
        except (IndexError, ValueError):
            pass
    return results


def save_processed(df: pd.DataFrame, target: str, processed_dir: str = "data/processed/") -> None:
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, f"features_{target}.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed features: {path}")


def load_processed(target: str, processed_dir: str = "data/processed/") -> Optional[pd.DataFrame]:
    path = os.path.join(processed_dir, f"features_{target}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)
