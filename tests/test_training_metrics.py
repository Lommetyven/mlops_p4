import pytest

pytest.importorskip("torch")

from main import (
    compute_classification_metrics,
    compute_regression_metrics,
    prefix_metrics,
    render_model_card,
    write_model_card,
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


def _model_card_inputs():
    config = {
        "experiment": {"model_version": "gru-test"},
        "task": {"type": "regression"},
        "model": {
            "input_size": 16,
            "hidden_size": 32,
            "num_layers": 1,
            "output_size": 1,
        },
        "data": {
            "processed_path": "data/processed/example.csv",
            "target_column": "target",
            "feature_columns": ["feature_a", "feature_b"],
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "sequence_length": 12,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "loss": "mse",
            "seed": 42,
        },
    }
    tracking_metadata = {
        "dataset/version": "abc123",
        "dataset/hash_sha256": "dataset-sha",
        "training/device": "cuda:0",
        "training/distributed": 0,
        "training/world_size": 1,
        "training/gpu_count": 1,
        "training/precision": "float32",
        "model/parameter_count": 1234,
        "git/commit_hash": "commit-sha",
        "config/path": "reports/runtime_train_config.yaml",
    }
    final_summary = {
        "model/best_epoch": 1,
        "model/best_metric": 0.25,
        "model/best_checkpoint": "models/gru_model.pt",
        "model/best_checkpoint_sha256": "checkpoint-sha",
    }
    test_metrics = {"loss": 0.25, "rmse": 0.5, "confusion_matrix": [[1]]}
    carbon_summary = {"carbontracker/actual_energy_kwh": 0.01}
    return config, tracking_metadata, final_summary, test_metrics, carbon_summary


def test_render_model_card_contains_core_sections():
    markdown = render_model_card(*_model_card_inputs())

    assert "# Model Card: gru-test" in markdown
    assert "## Dataset" in markdown
    assert "data/processed/example.csv" in markdown
    assert "checkpoint-sha" in markdown
    assert "actual_energy_kwh" in markdown
    assert "confusion_matrix" not in markdown


def test_write_model_card_creates_markdown_file(tmp_path):
    output_path = tmp_path / "model_card.md"

    model_card_path = write_model_card(*_model_card_inputs(), output_path=output_path)

    assert model_card_path == output_path
    assert output_path.read_text(encoding="utf-8").startswith("# Model Card: gru-test")
