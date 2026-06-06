import pytest

pytest.importorskip("torch")

from main import (
    compute_classification_metrics,
    compute_regression_metrics,
    prefix_metrics,
)


def test_compute_regression_metrics_returns_mae_rmse_and_r2():
    metrics = compute_regression_metrics(
        predictions=[1.0, 2.0, 3.0],
        targets=[1.0, 2.0, 4.0],
    )

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(11 / 14)


def test_compute_classification_metrics_returns_core_metrics():
    metrics = compute_classification_metrics(
        predictions=[[3.0, 0.0], [0.0, 3.0], [0.0, 2.0], [2.0, 0.0]],
        targets=[0, 1, 0, 0],
        task_config={"class_names": ["off", "on"]},
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert metrics["f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert metrics["confusion_matrix"] == [[2, 1], [0, 1]]
    assert metrics["per_class/off/precision"] == pytest.approx(1.0)
    assert metrics["per_class/on/recall"] == pytest.approx(1.0)


def test_prefix_metrics_keeps_only_scalar_values():
    metrics = prefix_metrics(
        {
            "loss": 0.5,
            "confusion_matrix": [[1, 0], [0, 1]],
            "class_names": ["a", "b"],
            "auc": None,
        },
        "test",
    )

    assert metrics == {"test/loss": 0.5}
