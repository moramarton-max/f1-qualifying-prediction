import numpy as np
import pandas as pd
import pytest

from src.features.delta_features import build_delta_features


def _make_session_features(drivers, base_time=90.0, offset=0.0):
    rows = [
        {
            "Driver": d,
            "LapTime_median": base_time + i * 0.3 + offset,
            "PaceRank": i + 1,
            "SpeedST": 310.0 - i,
        }
        for i, d in enumerate(drivers)
    ]
    return pd.DataFrame(rows)


DRIVERS = ["VER", "NOR", "LEC", "HAM", "RUS"]


def test_columns_created_for_each_session():
    fp1 = _make_session_features(DRIVERS)
    fp2 = _make_session_features(DRIVERS, offset=-0.2)
    result = build_delta_features({"FP1": fp1, "FP2": fp2}, ["FP1", "FP2"])
    assert "LapTime_median_FP1" in result.columns
    assert "LapTime_median_FP2" in result.columns
    assert "PaceRank_FP1" in result.columns


def test_missing_session_gives_nan():
    fp1 = _make_session_features(DRIVERS)
    result = build_delta_features({"FP1": fp1}, ["FP1", "FP2", "FP3"])
    assert result["LapTime_median_FP2"].isna().all()
    assert result["LapTime_median_FP3"].isna().all()


def test_improvement_is_negative_when_faster():
    fp1 = _make_session_features(DRIVERS, base_time=91.0)
    fp2 = _make_session_features(DRIVERS, base_time=90.0)  # faster
    result = build_delta_features({"FP1": fp1, "FP2": fp2}, ["FP1", "FP2"])
    # All drivers improved → improvement should be negative
    assert (result["LapTime_improvement"] < 0).all()


def test_one_driver_per_row():
    fp1 = _make_session_features(DRIVERS)
    result = build_delta_features({"FP1": fp1}, ["FP1"])
    assert len(result) == len(DRIVERS)
    assert result["Driver"].nunique() == len(DRIVERS)
