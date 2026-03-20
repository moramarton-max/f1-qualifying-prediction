import numpy as np
import pandas as pd
import pytest

from src.features.lap_features import extract_lap_features


def _make_laps(n_drivers=5, laps_per_driver=10) -> pd.DataFrame:
    rows = []
    for i in range(n_drivers):
        driver = f"DR{i}"
        for j in range(laps_per_driver):
            rows.append({
                "Driver": driver,
                "Team": f"Team{i % 3}",
                "LapTime": pd.Timedelta(seconds=90 + i * 0.5 + j * 0.01),
                "IsAccurate": True,
                "PitOutLap": False,
                "SpeedI1": 200.0 + i,
                "SpeedI2": 220.0 + i,
                "SpeedFL": 210.0 + i,
                "SpeedST": 310.0 + i,
                "IsWet": False,
            })
    return pd.DataFrame(rows)


def test_extract_returns_one_row_per_driver():
    laps = _make_laps(n_drivers=5)
    result = extract_lap_features(laps)
    assert len(result) == 5
    assert set(result["Driver"]) == {"DR0", "DR1", "DR2", "DR3", "DR4"}


def test_pace_rank_is_1_for_fastest():
    laps = _make_laps(n_drivers=3)
    result = extract_lap_features(laps)
    fastest = result.loc[result["LapTime_P5"].idxmin(), "Driver"]
    assert result.loc[result["Driver"] == fastest, "PaceRank"].values[0] == 1


def test_wet_session_returns_empty():
    laps = _make_laps(n_drivers=3)
    laps["IsWet"] = True
    result = extract_lap_features(laps)
    assert result.empty


def test_insufficient_laps_excluded():
    laps = _make_laps(n_drivers=3, laps_per_driver=1)  # only 1 lap per driver, min is 3
    result = extract_lap_features(laps)
    assert result.empty


def test_speed_columns_present():
    laps = _make_laps()
    result = extract_lap_features(laps)
    for col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
        assert col in result.columns
