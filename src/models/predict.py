"""Load a saved model artifact and produce ranked predictions for a weekend."""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_latest_artifact(models_dir: str = "models/artifacts/") -> dict:
    files = sorted(
        [f for f in os.listdir(models_dir) if f.startswith("quali_predictor_v") and f.endswith(".joblib")]
    )
    if not files:
        raise FileNotFoundError(f"No model artifact found in {models_dir}")
    path = os.path.join(models_dir, files[-1])
    logger.info(f"Loading model: {path}")
    return joblib.load(path)


def predict(
    feature_df: pd.DataFrame,
    artifact: Optional[dict] = None,
    models_dir: str = "models/artifacts/",
) -> pd.DataFrame:
    """
    feature_df : output of build_weekend_features() for the current weekend
    Returns a DataFrame with columns: Driver, Team, PredictedRank, Score
    sorted by PredictedRank ascending (1 = predicted pole).
    """
    if artifact is None:
        artifact = load_latest_artifact(models_dir)

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # Align columns — fill missing features with NaN
    X = feature_df.reindex(columns=feature_cols)

    scores = model.predict(X)
    result = pd.DataFrame({
        "Driver": feature_df["Driver"].values,
        "Team": feature_df.get("Team", pd.Series(["Unknown"] * len(feature_df))).values,
        "Score": scores,
    })
    result["PredictedRank"] = result["Score"].rank(ascending=False).astype(int)
    result = result.sort_values("PredictedRank").reset_index(drop=True)
    return result
