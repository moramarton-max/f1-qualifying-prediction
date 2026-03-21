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
from typing import List

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut

from src.models.evaluate import evaluate_all
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columns that are not model features
NON_FEATURE_COLS = [
    "Driver", "Team", "Year", "Round", "Target", "Era",
    "SampleWeight", "QualiPos", "pu_family", "Split",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _make_xgb(params: dict, n_jobs: int = -1) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        objective="rank:pairwise",
        random_state=42,
        n_jobs=n_jobs,
        **params,
    )


def _make_qid(df: pd.DataFrame) -> np.ndarray:
    """
    Sorted integer query-ID array for XGBoost ranking.
    Each (Year, Round) pair = one query. Data must be sorted by (Year, Round) first.
    """
    pairs = list(zip(df["Year"].values, df["Round"].values))
    unique_pairs = sorted(set(pairs))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    return np.array([pair_to_id[p] for p in pairs], dtype=np.int32)


def _group_weights(qid: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    XGBoost ranking requires one weight per query group.
    All drivers in a weekend share the same SampleWeight, so take the first.
    """
    _, first_idx = np.unique(qid, return_index=True)
    return weights[first_idx]


def _spearman_per_weekend(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    qid: np.ndarray,
) -> float:
    """
    Average Spearman correlation computed separately for each race weekend.
    XGBoost ranking scores are only comparable within a query group, not across.
    Higher score = predicted better = should correlate negatively with position (1=best).
    Returns the mean of per-weekend Spearman values (sign-flipped so +1 = perfect model).
    """
    corrs = []
    for g in np.unique(qid):
        mask = qid == g
        if mask.sum() < 3:
            continue
        corr, _ = spearmanr(y_true[mask], y_pred_scores[mask])
        corrs.append(-float(corr))  # negate: y_true is position (1=best), score is inverted
    return float(np.mean(corrs)) if corrs else 0.0


def _optuna_objective(
    trial: optuna.Trial,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    qid_tr: np.ndarray,
    cv_groups: np.ndarray,
    w_tr: np.ndarray,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    model = _make_xgb(params, n_jobs=1)

    logo = LeaveOneGroupOut()
    scores = []
    for inner_tr, inner_val in logo.split(X_tr, y_tr, cv_groups):
        if len(np.unique(cv_groups[inner_tr])) < 2:
            continue
        qid_i = qid_tr[inner_tr]
        model.fit(
            X_tr.iloc[inner_tr], y_tr[inner_tr],
            qid=qid_i,
            sample_weight=_group_weights(qid_i, w_tr[inner_tr]),
            verbose=False,
        )
        preds = model.predict(X_tr.iloc[inner_val])
        spearman = _spearman_per_weekend(y_tr[inner_val], preds, qid_tr[inner_val])
        scores.append(spearman)

    return float(np.mean(scores)) if scores else 0.0


def _eval_fold(
    model: xgb.XGBRanker,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    qid_val: np.ndarray,
) -> dict:
    """Evaluate model on a validation fold, aggregating per-weekend metrics."""
    preds = model.predict(X_val)
    per_weekend = []
    for g in np.unique(qid_val):
        mask = qid_val == g
        if mask.sum() < 3:
            continue
        per_weekend.append(evaluate_all(y_val[mask], preds[mask]))

    if not per_weekend:
        return {"spearman": 0.0, "top5_accuracy": 0.0, "mape": 0.0, "p1_hit_rate": 0.0}

    return {
        k: float(np.mean([m[k] for m in per_weekend]))
        for k in per_weekend[0]
    }


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

    # Drop rows with missing labels
    before = len(df)
    df = df[df[target_col].notna()].copy()
    if len(df) < before:
        logger.info(f"Dropped {before - len(df)} rows with NaN {target_col}")

    # Sort by (Year, Round) — required by XGBoost ranking (qid must be non-decreasing)
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]

    # XGBoost rank:pairwise treats higher label = better/more relevant.
    # QualiPos=1 is pole (best), so invert: relevance = max_pos + 1 - QualiPos.
    # Within each weekend max_pos is typically 20, but use group-wise max to handle
    # weekends with fewer classified drivers.
    max_pos = df.groupby(["Year", "Round"])[target_col].transform("max")
    y = (max_pos + 1 - df[target_col]).values.astype(float)  # pole -> highest value
    y_pos = df[target_col].values.astype(float)  # original positions, for evaluation
    qid = _make_qid(df)
    cv_groups = df["Year"].values
    weights = df["SampleWeight"].values

    logo = LeaveOneGroupOut()
    fold_metrics = []

    logger.info(
        f"Starting leave-one-season-out CV -- {len(np.unique(cv_groups))} seasons, "
        f"{len(np.unique(qid))} weekends, {len(df)} driver rows."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X, y, cv_groups)):
        val_season = np.unique(cv_groups[val_idx])[0]
        logger.info(f"Fold {fold_idx + 1}: validating on season {val_season}")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = weights[train_idx]
        qid_tr = qid[train_idx]
        qid_val = qid[val_idx]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _optuna_objective(
                trial, X_tr, y_tr, qid_tr, cv_groups[train_idx], w_tr
            ),
            n_trials=n_optuna_trials,
            show_progress_bar=False,
        )

        model = _make_xgb(study.best_params)
        model.fit(X_tr, y_tr, qid=qid_tr, sample_weight=_group_weights(qid_tr, w_tr), verbose=False)

        metrics = _eval_fold(model, X_val, y_pos[val_idx], qid_val)
        metrics["val_season"] = val_season
        fold_metrics.append(metrics)
        logger.info(
            f"  Spearman={metrics['spearman']:+.3f}  "
            f"Top5={metrics['top5_accuracy']:.2f}  "
            f"MAPE={metrics['mape']:.2f}  "
            f"P1={metrics['p1_hit_rate']:.0%}"
        )

    metrics_df = pd.DataFrame(fold_metrics)
    logger.info(
        f"\nCV summary:\n"
        f"{metrics_df[['val_season','spearman','top5_accuracy','mape','p1_hit_rate']].to_string()}"
    )

    # Final model on all data
    logger.info("Training final model on full training set...")
    study_final = optuna.create_study(direction="maximize")
    study_final.optimize(
        lambda trial: _optuna_objective(trial, X, y, qid, cv_groups, weights),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
    )
    final_model = _make_xgb(study_final.best_params)
    final_model.fit(X, y, qid=qid, sample_weight=_group_weights(qid, weights), verbose=False)

    if save:
        os.makedirs(models_dir, exist_ok=True)
        existing = [f for f in os.listdir(models_dir) if f.startswith("quali_predictor_v")]
        version = len(existing) + 1
        path = os.path.join(models_dir, f"quali_predictor_v{version}.joblib")
        joblib.dump({"model": final_model, "feature_cols": feature_cols, "cv_metrics": metrics_df}, path)
        logger.info(f"Saved model: {path}")

    return final_model
