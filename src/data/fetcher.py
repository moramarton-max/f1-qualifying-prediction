"""FastF1 data fetcher. Wraps session loading and returns tidy DataFrames."""

import os
from typing import Dict, Optional

import fastf1
import pandas as pd

from src.data.session_config import SESSION_IDENTIFIERS, get_preceding_sessions
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


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
    Fetch laps + basic telemetry for one session.
    Returns a DataFrame with columns including Driver, Team, LapTime, IsAccurate,
    PitOutLap, SpeedI1, SpeedI2, SpeedFL, SpeedST, and the session metadata columns
    Year, Round, Session added.
    Returns None if the session cannot be loaded.
    """
    enable_cache(cache_dir)
    identifier = SESSION_IDENTIFIERS.get(session_name, session_name)
    try:
        session = fastf1.get_session(year, round_number, identifier)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        laps = session.laps.copy()
        if laps.empty:
            logger.warning(f"No laps found: {year} R{round_number} {session_name}")
            return None

        # Attach session metadata
        laps["Year"] = year
        laps["Round"] = round_number
        laps["Session"] = session_name

        # Attach weather: mark laps as wet if rain was observed
        if session.weather_data is not None and not session.weather_data.empty:
            rain_observed = session.weather_data["Rainfall"].any()
            laps["IsWet"] = bool(rain_observed)
        else:
            laps["IsWet"] = False

        logger.info(f"Fetched {len(laps)} laps: {year} R{round_number} {session_name}")
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
