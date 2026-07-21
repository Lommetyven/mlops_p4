import pytest

from monitoring.carbon_tracking import (
    collect_carbontracker_summary,
    merged_carbon_tracking_config,
)
from monitoring.wandb_monitor import (
    WandbMonitorConfig,
    _to_float,
    collect_dataset_metadata,
    collect_hardware_metrics,
)


def test_wandb_monitor_config_loads_defaults_from_missing_file(tmp_path):
    config = WandbMonitorConfig.from_yaml(tmp_path / "missing.yaml")

    assert config.project == "MLOps"
    assert config.entity == "tobiasr-aalborg-universitet"
    assert config.mode == "online"
    assert config.watch_log == "all"


def test_wandb_monitor_config_loads_yaml_values(tmp_path):
    config_path = tmp_path / "monitoring.yaml"
    config_path.write_text(
        """
monitoring:
  project: "test-project"
  entity: "test-team"
  run_name: "unit-test"
  notes: "unit notes"
  tags:
    - "local"
    - "gpu"
  mode: "offline"
  hardware_log_freq: 7
""",
        encoding="utf-8",
    )

    config = WandbMonitorConfig.from_yaml(config_path)

    assert config.project == "test-project"
    assert config.entity == "test-team"
    assert config.run_name == "unit-test"
    assert config.notes == "unit notes"
    assert config.tags == ["local", "gpu"]
    assert config.mode == "offline"
    assert config.hardware_log_freq == 7


def test_collect_hardware_metrics_returns_metric_dict():
    metrics = collect_hardware_metrics(started_at=0.0)

    assert isinstance(metrics, dict)
    assert "hardware/runtime_seconds" in metrics
    assert "hardware/gpu_available" in metrics


def test_to_float_handles_invalid_values():
    assert _to_float("12.5") == 12.5
    assert _to_float("not-a-number") is None


def test_collect_dataset_metadata_includes_file_git_and_dvc_sections(tmp_path):
    dataset_path = tmp_path / "processed.csv"
    dataset_path.write_text("feature,target\n1.0,2.0\n", encoding="utf-8")

    metadata = collect_dataset_metadata(dataset_path)

    assert metadata["dataset/path"] == str(dataset_path)
    assert metadata["dataset/exists"] is True
    assert metadata["dataset/size_bytes"] > 0
    assert len(metadata["dataset/sha256"]) == 64
    assert "git" in metadata
    assert "dvc" in metadata


def test_collect_carbontracker_summary_flattens_latest_log(tmp_path):
    output_log = tmp_path / "lokalt_run_123_2026-06-05T120000Z_carbontracker_output.log"
    standard_log = tmp_path / "lokalt_run_123_2026-06-05T120000Z_carbontracker.log"

    output_log.write_text(
        """
2026-06-05 12:00:00 - CarbonTracker:
Actual consumption for 50 epoch(s):
    Time:   0:00:02
    Energy: 0.000100000000 kWh
    CO2eq:  0.014300000000 g
""",
        encoding="utf-8",
    )
    standard_log.write_text(
        "\n"
        "2026-06-05 12:00:00 - The following components were found: "
        "GPU with device(s) NVIDIA GeForce RTX 3070. "
        "CPU with device(s) Intel.\n"
        "2026-06-05 12:00:01 - Epoch 1:\n"
        "2026-06-05 12:00:02 - Duration: 0:00:02.00\n"
        "2026-06-05 12:00:03 - Average power usage (W) for gpu: 100.0\n"
        "2026-06-05 12:00:03 - Average power usage (W) for cpu: 50.0\n",
        encoding="utf-8",
    )

    summary = collect_carbontracker_summary(tmp_path)

    assert summary["carbontracker/actual_epochs"] == 50
    assert summary["carbontracker/actual_energy_kwh"] == 0.0001
    assert summary["carbontracker/actual_co2eq_g"] == 0.0143
    assert summary["carbontracker/gpu_avg_power_watts"] == 100.0
    assert summary["carbontracker/cpu_avg_power_watts"] == 50.0


def test_carbon_tracking_defaults_to_gpu_only():
    assert merged_carbon_tracking_config({})["components"] == "gpu"


def test_collect_carbontracker_summary_sums_multiple_gpus(tmp_path):
    output_log = tmp_path / "training_456_2026-07-19T171929Z_carbontracker_output.log"
    standard_log = tmp_path / "training_456_2026-07-19T171929Z_carbontracker.log"
    gpu_count = 4
    per_device_power_watts = 100.0
    duration_seconds = 2.0
    pue = 1.58
    carbon_intensity = 143.3
    raw_energy_kwh = per_device_power_watts * duration_seconds / 3_600_000 * pue
    raw_co2eq_g = raw_energy_kwh * carbon_intensity

    output_log.write_text(
        f"""
2026-07-19 17:19:29 - CarbonTracker:
Actual consumption for 1 epoch(s):
    Time:   0:00:02
    Energy: {raw_energy_kwh:.12f} kWh
    CO2eq:  {raw_co2eq_g:.12f} g
""",
        encoding="utf-8",
    )
    standard_log.write_text(
        "\n"
        "2026-07-19 17:19:29 - The following components were found: "
        "GPU with device(s) NVIDIA L4, NVIDIA L4, NVIDIA L4, NVIDIA L4.\n"
        "2026-07-19 17:19:30 - Epoch 1:\n"
        "2026-07-19 17:19:31 - Duration: 0:00:02.00\n"
        "2026-07-19 17:19:31 - Average power usage (W) for gpu: 100.0\n",
        encoding="utf-8",
    )

    summary = collect_carbontracker_summary(tmp_path)

    direct_gpu_energy_kwh = (
        gpu_count * per_device_power_watts * duration_seconds / 3_600_000
    )
    assert summary["carbontracker/gpu_device_count"] == gpu_count
    assert summary["carbontracker/gpu_avg_power_per_device_watts"] == 100.0
    assert summary["carbontracker/gpu_avg_power_watts"] == 400.0
    assert summary["carbontracker/gpu_energy_joules"] == pytest.approx(800.0)
    assert summary["carbontracker/gpu_energy_kwh"] == pytest.approx(
        direct_gpu_energy_kwh
    )
    assert summary["carbontracker/gpu_co2eq_g"] == pytest.approx(
        direct_gpu_energy_kwh * carbon_intensity
    )
    assert summary["carbontracker/actual_energy_kwh"] == pytest.approx(
        direct_gpu_energy_kwh * pue
    )
    assert summary["carbontracker/actual_co2eq_g"] == pytest.approx(
        direct_gpu_energy_kwh * pue * carbon_intensity
    )
    assert summary["carbontracker/gpu_consumption_correction_factor"] == (
        pytest.approx(gpu_count)
    )
    assert summary["carbontracker/raw_actual_energy_kwh"] == pytest.approx(
        raw_energy_kwh
    )
    assert not any(key.startswith("carbontracker/cpu") for key in summary)
