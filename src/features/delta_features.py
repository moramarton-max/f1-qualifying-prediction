"""
Compute session-to-session delta features for each driver within a weekend.

Given a dict of {session_name: lap_features_df}, produces per-driver columns:
  - LapTime_P5_{session}       : raw pace per session
  - PaceRank_{session}         : pace rank per session
  - SpeedST_{session}          : top straight speed per session
  - LapTime_improvement        : best_session_P5 - first_session_P5 (negative = improvement)
  - PaceRank_improvement       : first_session_PaceRank - best_session_PaceRank (positive = improved)
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_delta_features(
    session_features: Dict[str, pd.DataFrame],
    session_order: List[str],
) -> pd.DataFrame:
    """
    session_features : {session_name: DataFrame from extract_lap_features}
    session_order    : ordered list of sessions (e.g. ["FP1", "FP2", "FP3"])

    Returns a wide DataFrame with one row per driver.
    """
    all_drivers: set = set()
    for df in session_features.values():
        if not df.empty and "Driver" in df.columns:
            all_drivers.update(df["Driver"].tolist())

    if not all_drivers:
        return pd.DataFrame()

    result = pd.DataFrame({"Driver": sorted(all_drivers)})

    for session in session_order:
        df = session_features.get(session)
        if df is None or df.empty:
            # Session not available — leave as NaN (XGBoost handles this natively)
            result[f"LapTime_P5_{session}"] = np.nan
            result[f"PaceRank_{session}"] = np.nan
            result[f"SpeedST_{session}"] = np.nan
            continue

        sub = df[["Driver", "LapTime_P5", "PaceRank"]].copy()
        sub = sub.rename(columns={
            "LapTime_P5": f"LapTime_P5_{session}",
            "PaceRank": f"PaceRank_{session}",
        })
        if "SpeedST" in df.columns:
            sub[f"SpeedST_{session}"] = df["SpeedST"].values
        else:
            sub[f"SpeedST_{session}"] = np.nan

        result = result.merge(sub, on="Driver", how="left")

    # Improvement deltas: use first and last available sessions
    available = [s for s in session_order if f"LapTime_P5_{s}" in result.columns
                 and result[f"LapTime_P5_{s}"].notna().any()]

    if len(available) >= 2:
        first_s, last_s = available[0], available[-1]
        result["LapTime_improvement"] = (
            result[f"LapTime_P5_{last_s}"] - result[f"LapTime_P5_{first_s}"]
        )
        result["PaceRank_improvement"] = (
            result[f"PaceRank_{first_s}"] - result[f"PaceRank_{last_s}"]
        )
    else:
        result["LapTime_improvement"] = np.nan
        result["PaceRank_improvement"] = np.nan

    return result
