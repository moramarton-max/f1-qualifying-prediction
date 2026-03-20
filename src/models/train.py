"""
Train the qualifying prediction model.

Strategy:
  - Leave-one-season-out cross-validation (never leak future into past)
  - Era-weighted samples (2026 regulations are a distribution shift)
  - Optuna hyperparameter search within each CV fold
  - Final model trained on all available data

Output: saved artifact at models/artifacts/quali_predictor_v{n}.joblib
"""

import os
from typing import List, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut

from src.models.evaluate import evaluate_all
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columns that are not features
NON_FEATURE_COLS = ["Driver", "Team", "Year", "Round", "Target", "Era", "SampleWeight",
                    "QualiPos", "pu_family"]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _make_xgb(params: dict) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        objective="rank:pairwise",
        random_state=42,
        n_jobs=-1,
        **params,
    )


def _optuna_objective(trial: optuna.Trial, X_train, y_train, groups_train, weights_train) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    model = _make_xgb(params)

    # Inner CV: leave-one-group-out on the training fold
    logo = LeaveOneGroupOut()
    scores = []
    for inner_train, inner_val in logo.split(X_train, y_train, groups_train):
        if len(np.unique(groups_train[inner_train])) < 2:
            continue
        qids_train = _group_sizes(groups_train[inner_train])
        model.fit(
            X_train.iloc[inner_train], y_train[inner_train],
            qid=qids_train,
            sample_weight=weights_train[inner_train],
            eval_set=[(X_train.iloc[inner_val], y_train[inner_val])],
            verbose=False,
        )
        metrics = evaluate_all(y_train[inner_val], model.predict(X_train.iloc[inner_val]))
        scores.append(metrics["spearman"])

    return float(np.mean(scores)) if scores else 0.0


def _group_sizes(groups: np.ndarray) -> np.ndarray:
    """Convert group labels array to XGBoost qid format (group sizes)."""
    _, counts = np.unique(groups, return_counts=True)
    return counts


def train(
    df: pd.DataFrame,
    target_col: str = "QualiPos",
    n_optuna_trials: int = 50,
    models_dir: str = "models/artifacts/",
    save: bool = True,
) -> xgb.XGBRanker:
    """
    Train with leave-one-season-out CV, then fit final model on all data.
    Returns the final fitted model.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df[target_col].values.astype(float)
    groups = df["Year"].values
    weights = df["SampleWeight"].values

    logo = LeaveOneGroupOut()
    fold_metrics = []

    logger.info(f"Starting leave-one-season-out CV with {len(np.unique(groups))} seasons.")

    for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
        val_season = np.unique(groups[val_idx])[0]
        logger.info(f"Fold {fold_idx + 1}: validating on season {val_season}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = weights[train_idx]
        g_train = groups[train_idx]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _optuna_objective(trial, X_train, y_train, g_train, w_train),
            n_trials=n_optuna_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        model = _make_xgb(best_params)
        qids = _group_sizes(g_train)
        model.fit(X_train, y_train, qid=qids, sample_weight=w_train, verbose=False)

        metrics = evaluate_all(y_val, model.predict(X_val))
        metrics["val_season"] = val_season
        fold_metrics.append(metrics)
        logger.info(f"  Spearman={metrics['spearman']:.3f}  Top5={metrics['top5_accuracy']:.2f}  MAPE={metrics['mape']:.2f}")

    metrics_df = pd.DataFrame(fold_metrics)
    logger.info(f"\nCV summary:\n{metrics_df.describe()}")

    # Final model on all data
    logger.info("Training final model on all data...")
    study_final = optuna.create_study(direction="maximize")
    study_final.optimize(
        lambda trial: _optuna_objective(trial, X, y, groups, weights),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
    )
    final_model = _make_xgb(study_final.best_params)
    final_model.fit(X, y, qid=_group_sizes(groups), sample_weight=weights, verbose=False)

    if save:
        os.makedirs(models_dir, exist_ok=True)
        # Version the artifact
        existing = [f for f in os.listdir(models_dir) if f.startswith("quali_predictor_v")]
        version = len(existing) + 1
        path = os.path.join(models_dir, f"quali_predictor_v{version}.joblib")
        joblib.dump({"model": final_model, "feature_cols": feature_cols, "cv_metrics": metrics_df}, path)
        logger.info(f"Saved model: {path}")

    return final_model
