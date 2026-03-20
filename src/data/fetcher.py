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

        # Prune to only the columns we need
        present = [c for c in KEEP_COLS if c in laps.columns]
        missing = set(KEEP_COLS) - set(present)
        if missing:
            logger.debug(f"Columns not present in this session (will be NaN): {missing}")
        meta_cols = ["Year", "Round", "Session", "IsWet"]
        laps = laps[present + meta_cols]

        logger.info(
            f"Fetched {len(laps)} laps, {len(laps.columns)} cols: "
            f"{year} R{round_number} {session_name}"
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
