"""
Extract per-driver lap features from a pre-filtered laps DataFrame.

The fetcher already keeps only the TOP_N fastest clean laps per driver,
so this module just aggregates those into a single row per driver.

Features produced:
  - LapTime_median  : median lap time (seconds) over the stored fastest laps
  - LapTime_best    : single fastest lap time (seconds)
  - PaceRank        : rank by LapTime_median within the session (1 = fastest)
  - SpeedI1         : median SpeedI1 over stored laps
  - SpeedI2         : median SpeedI2 over stored laps
  - SpeedFL         : median SpeedFL over stored laps
  - SpeedST         : median SpeedST over stored laps (top straight speed proxy)
"""

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SPEED_COLS = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]

# Minimum number of laps required to include a driver
MIN_LAPS = 2


def _lap_time_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return series.astype(float)


def extract_lap_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pre-filtered laps into one row per driver.
    Input is expected to already contain only the fastest clean laps
    (as saved by the fetcher).
    """
    if laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    # Wet sessions were already dropped in the fetcher; double-check here
    if laps.get("IsWet", pd.Series(False, index=laps.index)).any():
        logger.debug("Skipping wet session.")
        return pd.DataFrame()

    laps = laps.copy()
    laps["LapTime_s"] = _lap_time_seconds(laps["LapTime"])

    # Drop drivers with too few laps
    counts = laps.groupby("Driver")["LapTime_s"].count()
    valid = counts[counts >= MIN_LAPS].index
    laps = laps[laps["Driver"].isin(valid)]
    if laps.empty:
        return pd.DataFrame()

    records = []
    for driver, group in laps.groupby("Driver"):
        row: dict = {"Driver": driver}
        times = group["LapTime_s"].dropna()
        row["LapTime_median"] = float(times.median())
        row["LapTime_best"] = float(times.min())

        for col in SPEED_COLS:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                row[col] = float(vals.median()) if len(vals) > 0 else np.nan
            else:
                row[col] = np.nan

        records.append(row)

    result = pd.DataFrame(records)
    result["PaceRank"] = result["LapTime_median"].rank(method="min").astype(int)
    result = result.sort_values("LapTime_median").reset_index(drop=True)
    return result
