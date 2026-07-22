import json

import pytest

from scripts.inference_reporting import (
    build_inference_metrics,
    calculate_rmse,
    count_predictions,
    tail_text,
    write_metrics,
)


def test_build_inference_metrics_counts_windows_and_throughput(tmp_path):
    input_path = tmp_path / "input.csv"
    predictions_path = tmp_path / "predictions.txt"
    targets_path = tmp_path / "targets.csv"
    input_path.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n", encoding="utf-8")
    predictions_path.write_text("0.1\n0.2\n0.3\n", encoding="utf-8")
    targets_path.write_text("target\n0.1\n0.4\n0.5\n", encoding="utf-8")

    metrics = build_inference_metrics(
        input_path=input_path,
        predictions_path=predictions_path,
        targets_path=targets_path,
        precision="FP16",
        batch_size=32,
        sequence_length=2,
        runtime_seconds=1.5,
        status="success",
        runner="AI_LAB",
        backend="python-torchscript",
        device="Test GPU",
        model_version="report-model",
    )

    assert metrics["inference/input_rows"] == 4
    assert metrics["inference/expected_windows"] == 3
    assert metrics["inference/prediction_count"] == 3
    assert metrics["inference/target_count"] == 3
    assert metrics["inference/rmse"] == pytest.approx((0.08 / 3) ** 0.5)
    assert metrics["inference/throughput_windows_per_second"] == 2.0
    assert metrics["inference/completed"] == 1


def test_prediction_count_ignores_runtime_log_lines(tmp_path):
    predictions_path = tmp_path / "predictions.txt"
    predictions_path.write_text(
        "Finished release target\n1.5\nerror text\n2.5\n",
        encoding="utf-8",
    )

    assert count_predictions(predictions_path) == 2


def test_calculate_rmse_rejects_mismatched_counts(tmp_path):
    predictions_path = tmp_path / "predictions.txt"
    targets_path = tmp_path / "targets.csv"
    predictions_path.write_text("1.0\n2.0\n", encoding="utf-8")
    targets_path.write_text("target\n1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="counts differ"):
        calculate_rmse(predictions_path, targets_path)


def test_write_metrics_and_error_tail(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    error_path = tmp_path / "error.log"
    error_path.write_text("old detail\nfinal error\n", encoding="utf-8")

    write_metrics(metrics_path, {"inference/status": "failed"})

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "inference/status": "failed"
    }
    assert tail_text(error_path, max_characters=12).endswith("final error\n")
