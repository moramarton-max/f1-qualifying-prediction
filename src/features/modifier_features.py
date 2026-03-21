"""
Compute team, driver, and power-unit modifiers from historical Q results.

All modifiers are derived exclusively from races PRIOR to the target weekend
(no leakage). Uses a rolling window of the last N races.

Features produced:
  - team_avg_quali_pos       : team's rolling mean QualiPos (lower = stronger car)
  - driver_avg_quali_pos     : driver's rolling mean QualiPos
  - driver_vs_teammate       : driver's mean (own_pos - teammate_pos); negative = beats teammate
  - pu_family                : power unit family label
  - SpeedST_pu_zscore        : driver's SpeedST z-score within their PU family this weekend
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# How many prior races to include in rolling averages
ROLLING_N_RACES = 20

# Power unit family assignments
PU_FAMILY: dict = {
    "Mercedes": "Mercedes",
    "Aston Martin": "Mercedes",
    "Williams": "Mercedes",
    "Ferrari": "Ferrari",
    "Haas F1 Team": "Ferrari",
    "Sauber": "Ferrari",
    "Red Bull Racing": "Honda",
    "RB": "Honda",
    "Alpine": "Renault",
}


def get_pu_family(team: str) -> str:
    return PU_FAMILY.get(team, "Unknown")


def _prior_races(
    results: pd.DataFrame,
    before_year: int,
    before_round: int,
    n: int = ROLLING_N_RACES,
) -> pd.DataFrame:
    """Return the last N Q_results rows that occurred before (before_year, before_round)."""
    prior = results[
        (results["Year"] < before_year) |
        ((results["Year"] == before_year) & (results["Round"] < before_round))
    ].copy()

    # Sort chronologically and take last N unique (Year, Round) pairs
    prior = prior.sort_values(["Year", "Round"])
    rounds = prior[["Year", "Round"]].drop_duplicates().tail(n)
    return prior.merge(rounds, on=["Year", "Round"])


def build_historical_modifiers(
    results: pd.DataFrame,
    before_year: int,
    before_round: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute team and driver modifiers using only races prior to (before_year, before_round).

    Returns:
        team_mods   : DataFrame[Team, team_avg_quali_pos]
        driver_mods : DataFrame[Driver, driver_avg_quali_pos, driver_vs_teammate]
    """
    prior = _prior_races(results, before_year, before_round)

    if prior.empty:
        return (
            pd.DataFrame(columns=["Team", "team_avg_quali_pos"]),
            pd.DataFrame(columns=["Driver", "driver_avg_quali_pos", "driver_vs_teammate"]),
        )

    # --- Team modifier ---
    team_mods = (
        prior.groupby("Team")["QualiPos"]
        .mean()
        .reset_index()
        .rename(columns={"QualiPos": "team_avg_quali_pos"})
    )

    # --- Driver modifiers ---
    driver_avg = (
        prior.groupby("Driver")["QualiPos"]
        .mean()
        .reset_index()
        .rename(columns={"QualiPos": "driver_avg_quali_pos"})
    )

    # Driver vs teammate: for each (Year, Round, Team), compute pos difference
    teammate_deltas = []
    for (year, rnd, team), group in prior.groupby(["Year", "Round", "Team"]):
        if len(group) != 2:
            continue
        d1, d2 = group.iloc[0], group.iloc[1]
        teammate_deltas.append({"Driver": d1["Driver"], "delta": d1["QualiPos"] - d2["QualiPos"]})
        teammate_deltas.append({"Driver": d2["Driver"], "delta": d2["QualiPos"] - d1["QualiPos"]})

    if teammate_deltas:
        vs_tm = (
            pd.DataFrame(teammate_deltas)
            .groupby("Driver")["delta"]
            .mean()
            .reset_index()
            .rename(columns={"delta": "driver_vs_teammate"})
        )
    else:
        vs_tm = pd.DataFrame(columns=["Driver", "driver_vs_teammate"])

    driver_mods = driver_avg.merge(vs_tm, on="Driver", how="left")

    return team_mods, driver_mods


def attach_modifiers(
    feature_df: pd.DataFrame,
    team_modifiers: Optional[pd.DataFrame] = None,
    driver_modifiers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge modifier tables into the feature DataFrame."""
    df = feature_df.copy()

    if team_modifiers is not None and not team_modifiers.empty and "Team" in df.columns:
        df = df.merge(team_modifiers, on="Team", how="left")

    if driver_modifiers is not None and not driver_modifiers.empty and "Driver" in df.columns:
        df = df.merge(driver_modifiers, on="Driver", how="left")

    if "Team" in df.columns:
        df["pu_family"] = df["Team"].map(get_pu_family)

    # PU z-score on latest available SpeedST
    speed_cols = [c for c in df.columns if c.startswith("SpeedST_") and not c.endswith("zscore")]
    if speed_cols and "Team" in df.columns:
        latest = speed_cols[-1]
        df["SpeedST_pu_zscore"] = _pu_zscore(df, latest)

    return df


def _pu_zscore(df: pd.DataFrame, speed_col: str) -> pd.Series:
    df = df.copy()
    df["_pu"] = df["Team"].map(get_pu_family)

    def zscore(g):
        vals = pd.to_numeric(g[speed_col], errors="coerce")
        std = vals.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=g.index)
        return (vals - vals.mean()) / std

    return df.groupby("_pu", group_keys=False).apply(zscore)
