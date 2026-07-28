import copy

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_monitor import (
    apply_feature_shift,
    build_evidently_snapshot,
    build_monitoring_markdown,
    build_shift_vector,
    calculate_regression_quality,
    extract_evidently_summary,
    finalize_monitoring_rows,
    save_evidently_snapshot,
    select_subset_indices,
    validate_drift_monitoring_config,
    write_monitoring_csv,
    write_monitoring_dashboard,
    write_monitoring_json,
    write_monitoring_markdown,
)


class IndexedSubset:
    def __init__(self, indices):
        self.indices = indices


def valid_config():
    return {
        "reference_sample_count": 10,
        "current_sample_count": 10,
        "drift_share_threshold": 0.5,
        "performance_rmse_relative_threshold": 0.1,
        "shifted_features": ["feature_a"],
        "windows": [
            {"name": "normal", "shift_std": 0},
            {"name": "drift", "shift_std": 1},
            {"name": "recovery", "shift_std": 0},
        ],
    }


def monitoring_rows():
    return [
        {
            "window_index": 0,
            "window_name": "normal",
            "window_label": "Normal",
            "shift_std": 0.0,
            "sample_count": 100,
            "drifted_feature_count": 0,
            "drifted_feature_share": 0.0,
            "prediction_drift_score": 0.8,
            "target_drift_score": 0.9,
            "rmse": 1.0,
            "mae": 0.8,
            "r2": 0.7,
            "failed_test_count": 0,
            "drifted_features": [],
        },
        {
            "window_index": 1,
            "window_name": "drift",
            "window_label": "Drift",
            "shift_std": 1.0,
            "sample_count": 100,
            "drifted_feature_count": 1,
            "drifted_feature_share": 0.5,
            "prediction_drift_score": 0.01,
            "target_drift_score": 0.9,
            "rmse": 1.2,
            "mae": 1.0,
            "r2": 0.3,
            "failed_test_count": 2,
            "drifted_features": ["feature_a"],
        },
    ]


def test_config_validation_rejects_invalid_monitoring_designs():
    features = ["feature_a", "feature_b"]
    assert validate_drift_monitoring_config(valid_config(), features)

    invalid = copy.deepcopy(valid_config())
    invalid["shifted_features"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown features"):
        validate_drift_monitoring_config(invalid, features)

    invalid = copy.deepcopy(valid_config())
    invalid["windows"] = [
        {"name": "one", "shift_std": 1},
        {"name": "two", "shift_std": 2},
        {"name": "three", "shift_std": 3},
    ]
    with pytest.raises(ValueError, match="unshifted control"):
        validate_drift_monitoring_config(invalid, features)

    invalid = copy.deepcopy(valid_config())
    invalid["drift_share_threshold"] = 0
    with pytest.raises(ValueError, match="interval"):
        validate_drift_monitoring_config(invalid, features)


def test_sample_selection_and_feature_shift_are_deterministic():
    subset = IndexedSubset(np.arange(30))
    first = select_subset_indices(subset, sample_count=8, seed=42)
    second = select_subset_indices(subset, sample_count=8, seed=42)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 8

    vector = build_shift_vector(
        feature_columns=["feature_a", "feature_b"],
        reference_standard_deviations={
            "feature_a": 2.0,
            "feature_b": 4.0,
        },
        shifted_features=["feature_a"],
        shift_std=1.5,
    )
    shifted = apply_feature_shift(np.zeros((2, 3, 2)), vector)
    assert vector.tolist() == pytest.approx([3.0, 0.0])
    assert shifted[:, :, 0] == pytest.approx(3.0)
    assert shifted[:, :, 1] == pytest.approx(0.0)


def test_regression_metrics_and_reaction_policy():
    quality = calculate_regression_quality(
        predictions=[0.0, 4.0],
        targets=[0.0, 2.0],
    )
    assert quality["rmse"] == pytest.approx(np.sqrt(2))
    assert quality["mae"] == pytest.approx(1.0)
    assert quality["r2"] == pytest.approx(-1.0)

    rows = finalize_monitoring_rows(
        monitoring_rows(),
        drift_share_threshold=0.5,
        performance_rmse_relative_threshold=0.1,
    )
    assert rows[0]["reaction"] == "monitor"
    assert rows[1]["dataset_drift_detected"] == 1
    assert rows[1]["performance_degraded"] == 1
    assert rows[1]["reaction"] == "investigate_and_retrain"
    assert rows[1]["rmse_change_percent_from_baseline"] == pytest.approx(20.0)


def test_evidently_snapshot_detects_known_shift(tmp_path):
    generator = np.random.default_rng(7)
    reference = pd.DataFrame(
        {
            "feature_a": generator.normal(0, 1, 400),
            "feature_b": generator.normal(0, 1, 400),
        }
    )
    current = pd.DataFrame(
        {
            "feature_a": generator.normal(4, 1, 400),
            "feature_b": generator.normal(0, 1, 400),
        }
    )
    for frame in (reference, current):
        frame["target"] = frame["feature_b"] * 0.5
        frame["prediction"] = frame["target"] + generator.normal(
            0.1,
            0.01,
            len(frame),
        )

    snapshot = build_evidently_snapshot(
        reference_frame=reference,
        current_frame=current,
        feature_columns=["feature_a", "feature_b"],
        window_name="known-shift",
        timestamp=None,
        model_version="test",
        drift_share_threshold=0.5,
        drift_method="ks",
        drift_threshold=0.05,
    )
    summary = extract_evidently_summary(
        snapshot,
        ["feature_a", "feature_b"],
    )
    assert summary["drifted_feature_count"] >= 1
    assert "feature_a" in summary["drifted_features"]
    assert summary["rmse"] == pytest.approx(0.1, abs=0.01)
    html_path, json_path = save_evidently_snapshot(
        snapshot,
        tmp_path / "known_shift",
    )
    assert html_path.stat().st_size > 0
    assert json_path.stat().st_size > 0


def test_monitoring_evidence_writers(tmp_path):
    rows = finalize_monitoring_rows(
        monitoring_rows(),
        drift_share_threshold=0.5,
        performance_rmse_relative_threshold=0.1,
    )
    report = {
        "model_version": "test-model",
        "device": "CPU",
        "configuration": {
            "drift_method": "ks",
            "drift_threshold": 0.05,
            "drift_share_threshold": 0.5,
            "performance_rmse_relative_threshold": 0.1,
            "feature_columns": ["feature_a", "feature_b"],
            "shifted_features": ["feature_a"],
        },
        "windows": rows,
    }

    markdown = build_monitoring_markdown(report)
    assert "Evidently drift monitoring experiment" in markdown
    assert "investigate_and_retrain" in markdown
    assert write_monitoring_json(tmp_path / "summary.json", report).is_file()
    assert write_monitoring_csv(tmp_path / "summary.csv", rows).is_file()
    assert write_monitoring_markdown(
        tmp_path / "summary.md",
        report,
    ).is_file()
    plot = write_monitoring_dashboard(
        tmp_path / "dashboard.png",
        rows,
        report["configuration"],
    )
    assert plot.is_file()
    assert plot.stat().st_size > 0
