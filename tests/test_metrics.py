import numpy as np

from moral_bench import metrics


def test_auroc_perfect_and_inverted():
    y = np.array([True, True, False, False])
    assert metrics.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert metrics.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_auroc_ties_count_half():
    y = np.array([True, False])
    assert metrics.auroc(y, np.array([0.5, 0.5])) == 0.5


def test_auroc_single_class_is_nan():
    assert np.isnan(metrics.auroc(np.array([True, True]), np.array([0.5, 0.6])))


def test_accuracy_brier_ece():
    y = np.array([True, False, True, False])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert metrics.accuracy(y, p) == 1.0
    assert metrics.brier(y, p) == 0.0
    assert metrics.ece(y, p) == 0.0
