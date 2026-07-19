import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.tune_batch_size import (
    build_candidate_batch_sizes,
    select_best_result,
    update_runtime_config_batch_size,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_script_entrypoint_can_import_project_modules():
    result = subprocess.run(
        [sys.executable, "scripts/tune_batch_size.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_build_candidate_batch_sizes_includes_powers_configured_and_maximum():
    candidates = build_candidate_batch_sizes(
        configured=48,
        minimum=16,
        maximum=100,
    )

    assert candidates == [16, 32, 48, 64, 100]


def test_build_candidate_batch_sizes_handles_small_training_shard():
    candidates = build_candidate_batch_sizes(
        configured=32,
        minimum=16,
        maximum=7,
    )

    assert candidates == [7]


def test_select_best_result_uses_throughput_and_ignores_failed_candidates():
    selected = select_best_result(
        [
            {
                "batch_size_per_gpu": 32,
                "status": "ok",
                "aggregate_samples_per_second": 1200.0,
            },
            {
                "batch_size_per_gpu": 64,
                "status": "ok",
                "aggregate_samples_per_second": 1800.0,
            },
            {
                "batch_size_per_gpu": 128,
                "status": "memory_limit",
                "aggregate_samples_per_second": 2000.0,
            },
        ]
    )

    assert selected["batch_size_per_gpu"] == 64


def test_select_best_result_prefers_smaller_batch_on_throughput_tie():
    selected = select_best_result(
        [
            {
                "batch_size_per_gpu": 64,
                "status": "ok",
                "aggregate_samples_per_second": 1800.0,
            },
            {
                "batch_size_per_gpu": 32,
                "status": "ok",
                "aggregate_samples_per_second": 1800.0,
            },
        ]
    )

    assert selected["batch_size_per_gpu"] == 32


def test_update_runtime_config_batch_size_preserves_other_values(tmp_path):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"hidden_size": 800},
                "training": {"batch_size": 32, "epochs": 50},
            }
        ),
        encoding="utf-8",
    )

    update_runtime_config_batch_size(config_path, 256)

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated == {
        "model": {"hidden_size": 800},
        "training": {"batch_size": 256, "epochs": 50},
    }


def test_write_report_creates_json_parent_directory(tmp_path):
    report_path = tmp_path / "reports" / "batch_size_tuning.json"

    write_report(report_path, {"selected_batch_size_per_gpu": 128})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "selected_batch_size_per_gpu": 128
    }
