import numpy as np
import pytest

from src.models.evaluate import (
    evaluate_all,
    mean_absolute_position_error,
    p1_hit_rate,
    spearman_correlation,
    top_n_accuracy,
)

TRUE_RANKS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
PERFECT_SCORES = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # higher = better predicted rank
REVERSE_SCORES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # perfect reverse prediction


def test_spearman_perfect():
    assert spearman_correlation(TRUE_RANKS, TRUE_RANKS) == pytest.approx(1.0)


def test_spearman_reverse():
    assert spearman_correlation(TRUE_RANKS, TRUE_RANKS[::-1]) == pytest.approx(-1.0)


def test_top5_accuracy_perfect():
    assert top_n_accuracy(TRUE_RANKS, TRUE_RANKS, n=5) == pytest.approx(1.0)


def test_top5_accuracy_zero():
    # True top-5 are positions 1-5; predicted top-5 should be positions 6-10
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = np.array([6, 7, 8, 9, 10, 1, 2, 3, 4, 5])
    assert top_n_accuracy(y_true, y_pred, n=5) == pytest.approx(0.0)


def test_mape_zero_for_perfect():
    assert mean_absolute_position_error(TRUE_RANKS, TRUE_RANKS) == pytest.approx(0.0)


def test_p1_hit_rate_correct():
    assert p1_hit_rate(TRUE_RANKS, TRUE_RANKS) == pytest.approx(1.0)


def test_p1_hit_rate_wrong():
    wrong = TRUE_RANKS.copy()
    wrong[0] = 20  # move P1 driver to last
    assert p1_hit_rate(TRUE_RANKS, wrong) == pytest.approx(0.0)


def test_evaluate_all_returns_all_keys():
    metrics = evaluate_all(TRUE_RANKS, PERFECT_SCORES)
    assert set(metrics.keys()) == {"spearman", "top5_accuracy", "mape", "p1_hit_rate"}
