"""FastF1 data fetcher. Wraps session loading and returns tidy DataFrames."""

import os
from typing import Dict, Optional

import fastf1
import pandas as pd

from src.data.session_config import SESSION_IDENTIFIERS, get_preceding_sessions
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Only the columns we actually use downstream — keeps parquet files small
# and avoids saving telemetry/timing blobs we never read.
# Only the columns we actually use downstream
KEEP_COLS = [
    "Driver",
    "Team",
    "LapTime",
    "IsAccurate",
    "PitOutTime",   # NaT = not a pit-out lap
    "SpeedI1",      # intermediate speed trap 1
    "SpeedI2",      # intermediate speed trap 2
    "SpeedFL",      # finish-line speed
    "SpeedST",      # speed trap (main straight) — proxy for top straight speed
    "TrackStatus",  # used to detect safety car / wet track codes
]

# How many of the fastest clean laps to keep per driver.
# 5 laps → ~100 rows/session instead of ~1000+, while still giving
# enough data to compute a robust median lap time and speed trap values.
TOP_N_LAPS = 5


def _keep_fastest_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    For each driver, keep only the TOP_N_LAPS fastest accurate, non-pit-out laps.
    Wet sessions are dropped entirely (speed values not comparable).
    """
    if laps.get("IsWet", pd.Series(False, index=laps.index)).any():
        return laps.iloc[0:0]

    is_accurate = laps["IsAccurate"].fillna(False).astype(bool)
    has_laptime = laps["LapTime"].notna()

    if "PitOutTime" in laps.columns:
        is_pit_out = laps["PitOutTime"].notna()
    else:
        is_pit_out = pd.Series(False, index=laps.index)

    clean = laps[is_accurate & has_laptime & ~is_pit_out].copy()

    # Convert to seconds for sorting
    lt = clean["LapTime"]
    clean["_lt_s"] = lt.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(lt) else lt.astype(float)

    # Keep top N fastest per driver
    clean = (
        clean.sort_values("_lt_s")
        .groupby("Driver", sort=False)
        .head(TOP_N_LAPS)
        .drop(columns=["_lt_s"])
        .reset_index(drop=True)
    )
    return clean


def enable_cache(cache_dir: str = "cache/") -> None:
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def fetch_session(
    year: int,
    round_number: int,
    session_name: str,
    cache_dir: str = "cache/",
) -> Optional[pd.DataFrame]:
    """
    Fetch laps for one session and return a pruned DataFrame.

    Only KEEP_COLS are retained plus Year/Round/Session/IsWet metadata.
    No telemetry is loaded — speed trap values (SpeedI1/I2/FL/ST) come
    directly from the laps DataFrame and require no extra API calls.

    Returns None if the session cannot be loaded.
    """
    enable_cache(cache_dir)
    identifier = SESSION_IDENTIFIERS.get(session_name, session_name)
    try:
        session = fastf1.get_session(year, round_number, identifier)
        # weather=True adds one extra lightweight API call but gives us
        # rainfall data to flag wet sessions — worth keeping.
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        laps = session.laps.copy()
        if laps.empty:
            logger.warning(f"No laps found: {year} R{round_number} {session_name}")
            return None

        # Add session metadata before pruning so they're always present
        laps["Year"] = year
        laps["Round"] = round_number
        laps["Session"] = session_name

        # Mark session as wet — single bool, not per-lap
        if session.weather_data is not None and not session.weather_data.empty:
            laps["IsWet"] = bool(session.weather_data["Rainfall"].any())
        else:
            laps["IsWet"] = False

        # Prune to only the columns we need, then keep only the fastest laps
        present = [c for c in KEEP_COLS if c in laps.columns]
        meta_cols = ["Year", "Round", "Session", "IsWet"]
        laps = laps[present + meta_cols]
        laps = _keep_fastest_laps(laps)

        if laps.empty:
            logger.warning(f"No clean laps after filtering: {year} R{round_number} {session_name}")
            return None

        logger.info(
            f"Fetched {len(laps)} laps ({TOP_N_LAPS} fastest/driver), "
            f"{len(laps.columns)} cols: {year} R{round_number} {session_name}"
        )
        return laps

    except Exception as exc:
        logger.error(f"Failed to fetch {year} R{round_number} {session_name}: {exc}")
        return None


def fetch_weekend(
    year: int,
    round_number: int,
    target: str = "Q",
    cache_dir: str = "cache/",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all sessions preceding the target for one race weekend.
    Returns a dict mapping session_name -> laps DataFrame (missing sessions omitted).
    """
    sessions = get_preceding_sessions(year, round_number, target)
    result = {}
    for s in sessions:
        df = fetch_session(year, round_number, s, cache_dir)
        if df is not None:
            result[s] = df
    return result
