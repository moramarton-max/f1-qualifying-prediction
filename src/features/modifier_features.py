"""
Compute team modifier, driver modifier, and power-unit grouping features.

These are historical modifiers derived from prior races (not the current weekend),
so they do not leak future information.

Features produced:
  - team_quali_modifier      : team's historical mean (QualiPos - ExpectedPosByPace)
  - driver_quali_vs_teammate : driver's historical mean quali delta vs teammate
  - pu_family                : power unit family label
  - SpeedST_pu_zscore        : driver's SpeedST z-score within their PU family (current weekend)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Power unit family assignments (updated for 2025 season)
PU_FAMILY: dict = {
    # Mercedes PU
    "Mercedes": "Mercedes",
    "Aston Martin": "Mercedes",
    "Williams": "Mercedes",
    # Ferrari PU
    "Ferrari": "Ferrari",
    "Haas F1 Team": "Ferrari",
    "Sauber": "Ferrari",
    # Honda/RBPT PU
    "Red Bull Racing": "Honda",
    "RB": "Honda",
    # Renault/Alpine PU
    "Alpine": "Renault",
}


def get_pu_family(team: str) -> str:
    return PU_FAMILY.get(team, "Unknown")


def compute_pu_zscore(df: pd.DataFrame, speed_col: str = "SpeedST_latest") -> pd.Series:
    """
    Given a DataFrame with columns [Driver, Team, speed_col],
    compute z-score of speed within each PU family.
    """
    if speed_col not in df.columns or "Team" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    df = df.copy()
    df["pu_family"] = df["Team"].map(get_pu_family)

    def zscore_group(g: pd.DataFrame) -> pd.Series:
        vals = pd.to_numeric(g[speed_col], errors="coerce")
        mean, std = vals.mean(), vals.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=g.index)
        return (vals - mean) / std

    return df.groupby("pu_family", group_keys=False).apply(zscore_group)


def compute_team_modifiers(historical_results: pd.DataFrame) -> pd.DataFrame:
    """
    historical_results must have columns: Year, Round, Team, QualiPos, PaceRankByCar
    (PaceRankByCar = rank by best practice lap time, as a proxy for car pace).

    Returns DataFrame[Team, team_quali_modifier].
    """
    required = {"Team", "QualiPos", "PaceRankByCar"}
    if not required.issubset(historical_results.columns):
        logger.warning(f"Missing columns for team modifier: {required - set(historical_results.columns)}")
        return pd.DataFrame(columns=["Team", "team_quali_modifier"])

    df = historical_results.copy()
    df["delta"] = df["QualiPos"] - df["PaceRankByCar"]
    modifiers = df.groupby("Team")["delta"].mean().reset_index()
    modifiers.columns = ["Team", "team_quali_modifier"]
    return modifiers


def compute_driver_modifiers(historical_results: pd.DataFrame) -> pd.DataFrame:
    """
    historical_results must have columns: Year, Round, Driver, Team, QualiPos, TeammateQualiPos.

    Returns DataFrame[Driver, driver_quali_vs_teammate].
    """
    required = {"Driver", "QualiPos", "TeammateQualiPos"}
    if not required.issubset(historical_results.columns):
        logger.warning(f"Missing columns for driver modifier: {required - set(historical_results.columns)}")
        return pd.DataFrame(columns=["Driver", "driver_quali_vs_teammate"])

    df = historical_results.copy()
    df["delta"] = df["QualiPos"] - df["TeammateQualiPos"]
    modifiers = df.groupby("Driver")["delta"].mean().reset_index()
    modifiers.columns = ["Driver", "driver_quali_vs_teammate"]
    return modifiers


def attach_modifiers(
    feature_df: pd.DataFrame,
    team_modifiers: Optional[pd.DataFrame] = None,
    driver_modifiers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge pre-computed modifier tables into the main feature DataFrame.
    Missing drivers/teams get NaN (handled by XGBoost natively).
    """
    df = feature_df.copy()

    if team_modifiers is not None and not team_modifiers.empty and "Team" in df.columns:
        df = df.merge(team_modifiers, on="Team", how="left")

    if driver_modifiers is not None and not driver_modifiers.empty and "Driver" in df.columns:
        df = df.merge(driver_modifiers, on="Driver", how="left")

    # Add PU family label
    if "Team" in df.columns:
        df["pu_family"] = df["Team"].map(get_pu_family)

    # Add PU z-score for latest available SpeedST
    speed_cols = [c for c in df.columns if c.startswith("SpeedST_")]
    if speed_cols:
        latest_speed_col = speed_cols[-1]
        df["SpeedST_pu_zscore"] = compute_pu_zscore(df, speed_col=latest_speed_col)

    return df
