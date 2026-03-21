"""Evaluation metrics for rank prediction."""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between true and predicted positions."""
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def top_n_accuracy(y_true: np.ndarray, y_pred_ranks: np.ndarray, n: int = 5) -> float:
    """Fraction of actual top-N drivers that appear in predicted top-N."""
    actual_top = set(np.argsort(y_true)[:n])
    predicted_top = set(np.argsort(y_pred_ranks)[:n])
    return len(actual_top & predicted_top) / n


def mean_absolute_position_error(y_true: np.ndarray, y_pred_ranks: np.ndarray) -> float:
    """Average number of grid positions the prediction is off by (lower = better)."""
    return float(np.mean(np.abs(y_true - y_pred_ranks)))


def p1_hit_rate(y_true: np.ndarray, y_pred_ranks: np.ndarray) -> float:
    """1.0 if the model correctly predicted pole/P1, else 0.0."""
    return float(np.argmin(y_true) == np.argmin(y_pred_ranks))


def evaluate_all(y_true: np.ndarray, y_pred_scores: np.ndarray) -> Dict[str, float]:
    """
    y_true        : actual qualifying positions (1-based, lower = better)
    y_pred_scores : model output scores (higher = predicted better)
    """
    # Convert scores to predicted ranks (1-based)
    pred_ranks = pd.Series(y_pred_scores).rank(ascending=False).values

    return {
        "spearman": spearman_correlation(y_true, pred_ranks),
        "top5_accuracy": top_n_accuracy(y_true, pred_ranks, n=5),
        "mape_pos": mean_absolute_position_error(y_true, pred_ranks),
        "p1_hit_rate": p1_hit_rate(y_true, pred_ranks),
    }
