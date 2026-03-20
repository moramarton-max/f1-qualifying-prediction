"""
Extract per-driver lap features from a raw laps DataFrame.

Features produced (per driver per session):
  - LapTime_P5      : 5th-percentile lap time (seconds) over clean laps
  - PaceRank        : position rank by LapTime_P5 within the session (1 = fastest)
  - SpeedI1         : median SpeedI1 (intermediate speed trap 1) over clean laps
  - SpeedI2         : median SpeedI2 (intermediate speed trap 2) over clean laps
  - SpeedFL         : median SpeedFL (finish-line speed) over clean laps
  - SpeedST         : median SpeedST (speed trap) over clean laps — proxy for top straight speed
"""

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

LAP_PERCENTILE = 5
MIN_LAPS = 3

SPEED_COLS = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]


def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep only accurate, non-pit-out, non-wet laps with a valid lap time."""
    mask = (
        laps["IsAccurate"].fillna(False).astype(bool)
        & ~laps.get("PitOutLap", pd.Series(False, index=laps.index)).fillna(False).astype(bool)
        & laps["LapTime"].notna()
    )
    # Drop wet sessions entirely — speed values are not comparable
    if laps.get("IsWet", pd.Series(False, index=laps.index)).any():
        logger.debug("Skipping wet session laps")
        return laps.iloc[0:0]
    return laps[mask].copy()


def _lap_time_seconds(laps: pd.DataFrame) -> pd.Series:
    """Convert LapTime (timedelta or float) to seconds."""
    lt = laps["LapTime"]
    if pd.api.types.is_timedelta64_dtype(lt):
        return lt.dt.total_seconds()
    return lt.astype(float)


def extract_lap_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Given a raw laps DataFrame for one session, return a tidy DataFrame
    indexed by Driver with the features listed in the module docstring.
    """
    clean = _clean_laps(laps)
    if clean.empty:
        logger.warning("No clean laps available for feature extraction.")
        return pd.DataFrame()

    clean = clean.copy()
    clean["LapTime_s"] = _lap_time_seconds(clean)

    # Keep only drivers with enough clean laps
    lap_counts = clean.groupby("Driver")["LapTime_s"].count()
    valid_drivers = lap_counts[lap_counts >= MIN_LAPS].index
    clean = clean[clean["Driver"].isin(valid_drivers)]

    if clean.empty:
        return pd.DataFrame()

    records = []
    for driver, group in clean.groupby("Driver"):
        row: dict = {"Driver": driver}

        # 5th-percentile lap time
        row["LapTime_P5"] = float(np.percentile(group["LapTime_s"].dropna(), LAP_PERCENTILE))

        # Speed traps: median over clean laps
        for col in SPEED_COLS:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                row[col] = float(vals.median()) if len(vals) > 0 else np.nan
            else:
                row[col] = np.nan

        records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # Add pace rank (1 = fastest)
    result = result.sort_values("LapTime_P5").reset_index(drop=True)
    result["PaceRank"] = result["LapTime_P5"].rank(method="min").astype(int)

    return result
