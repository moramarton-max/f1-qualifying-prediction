"""
Pipeline orchestrator: raw laps → flat feature DataFrame ready for modelling.

Usage (training):
    df = build_dataset(weekends, target="Q")

Usage (live prediction for one weekend):
    df = build_weekend_features(year, round_number, target="Q",
                                available_sessions=["FP1", "FP2"])
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.loader import load_raw, save_raw, load_processed, save_processed
from src.data.session_config import get_preceding_sessions
from src.features.delta_features import build_delta_features
from src.features.lap_features import extract_lap_features
from src.features.modifier_features import attach_modifiers
from src.utils.logging_config import get_logger
from src.utils.regulation_era import get_era, get_sample_weight

logger = get_logger(__name__)

# Temporal holdout split — do not change once training has started
TEST_WEEKENDS = {
    (2025, 21), (2025, 22), (2025, 23), (2025, 24),  # last 4 rounds of 2025
    (2026, 1), (2026, 2),                              # first 2 rounds of 2026
}


def get_split(year: int, round_number: int) -> str:
    return "test" if (year, round_number) in TEST_WEEKENDS else "train"


def _load_quali_results(year: int, round_number: int, raw_dir: str) -> Optional[pd.DataFrame]:
    """Load saved qualifying results (Driver, QualiPos) for one weekend."""
    path = os.path.join(raw_dir, f"{year}_R{round_number:02d}_Q_results.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)[["Driver", "QualiPos"]]


def _build_weekend_row(
    year: int,
    round_number: int,
    target: str,
    available_sessions: Optional[List[str]] = None,
    raw_dir: str = "data/raw/",
    include_labels: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Build feature rows for one weekend. Loads raw data from disk.
    Returns a DataFrame with one row per driver, or None if data unavailable.
    include_labels=True attaches QualiPos (needed for training, not for live prediction).
    """
    session_order = get_preceding_sessions(year, round_number, target)
    if available_sessions is not None:
        session_order = [s for s in session_order if s in available_sessions]

    session_features: Dict[str, pd.DataFrame] = {}
    for session in session_order:
        laps = load_raw(year, round_number, session, raw_dir)
        if laps is None:
            logger.debug(f"Raw data not found: {year} R{round_number} {session} — will be NaN")
            session_features[session] = pd.DataFrame()
            continue
        feat = extract_lap_features(laps)
        session_features[session] = feat

    delta_df = build_delta_features(session_features, session_order)
    if delta_df.empty:
        logger.warning(f"No features for {year} R{round_number} {target}")
        return None

    # Attach team info from the first available session
    team_map: Dict[str, str] = {}
    for session in session_order:
        laps = load_raw(year, round_number, session, raw_dir)
        if laps is not None and "Driver" in laps.columns and "Team" in laps.columns:
            for _, row in laps[["Driver", "Team"]].drop_duplicates().iterrows():
                team_map[row["Driver"]] = row["Team"]
            break

    delta_df["Team"] = delta_df["Driver"].map(team_map)
    delta_df = attach_modifiers(delta_df)

    # Weekend metadata
    delta_df["Year"] = year
    delta_df["Round"] = round_number
    delta_df["Target"] = target
    delta_df["Era"] = get_era(year)
    delta_df["SampleWeight"] = get_sample_weight(year)
    delta_df["Split"] = get_split(year, round_number)

    # Attach qualifying results as training labels
    if include_labels:
        results = _load_quali_results(year, round_number, raw_dir)
        if results is not None:
            delta_df = delta_df.merge(results, on="Driver", how="left")
        else:
            logger.warning(f"No quali results found for {year} R{round_number} — QualiPos will be NaN")
            delta_df["QualiPos"] = float("nan")

    return delta_df


def build_dataset(
    weekends: List[Tuple[int, int]],
    target: str = "Q",
    raw_dir: str = "data/raw/",
    save: bool = True,
    processed_dir: str = "data/processed/",
) -> pd.DataFrame:
    """
    Build the full training dataset from a list of (year, round) tuples.
    Only includes weekends where quali results are available (skips rounds
    without labels so the training set stays clean).
    Saves processed features to disk by default.
    """
    frames = []
    skipped = 0
    for year, round_number in weekends:
        df = _build_weekend_row(year, round_number, target, raw_dir=raw_dir, include_labels=True)
        if df is None:
            continue
        if df["QualiPos"].isna().all():
            logger.warning(f"Skipping {year} R{round_number} — no quali results on disk")
            skipped += 1
            continue
        frames.append(df)

    if not frames:
        logger.error("No data was available for any of the requested weekends.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Built dataset: {len(result)} driver-weekend rows, "
        f"{result.shape[1]} features, {skipped} weekends skipped (no labels)"
    )

    if save:
        save_processed(result, target, processed_dir)

    return result


def build_weekend_features(
    year: int,
    round_number: int,
    target: str = "Q",
    available_sessions: Optional[List[str]] = None,
    cache_dir: str = "cache/",
    raw_dir: str = "data/raw/",
) -> Optional[pd.DataFrame]:
    """
    Fetch + build features for a live (in-progress) weekend.
    Does not attach QualiPos (unknown at prediction time).
    """
    session_order = get_preceding_sessions(year, round_number, target)
    if available_sessions is not None:
        session_order = [s for s in session_order if s in available_sessions]

    from src.data.fetcher import fetch_session
    from src.data.loader import is_raw_available
    for session in session_order:
        if not is_raw_available(year, round_number, session, raw_dir):
            laps = fetch_session(year, round_number, session, cache_dir)
            if laps is not None:
                save_raw(laps, year, round_number, session, raw_dir)

    return _build_weekend_row(
        year, round_number, target, available_sessions, raw_dir, include_labels=False
    )
