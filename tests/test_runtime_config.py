from scripts.write_runtime_config import build_monitoring_config, build_runtime_config


def test_build_runtime_config_applies_parameter_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("EPOCHS", "7")
    monkeypatch.setenv("BATCH_SIZE", "128")
    monkeypatch.setenv("FLOAT_PRECISION", "float16")
    monkeypatch.setenv("DATASET_PATH", "data/processed/custom.csv")
    monkeypatch.setenv("DO_TEST", "false")
    monkeypatch.setenv("CARBON_TRACKING", "true")

    config = build_runtime_config(
        {
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "precision": "float32",
            },
            "data": {"processed_path": "data/processed/default.csv"},
            "monitoring": {},
            "carbon_tracking": {"enabled": False},
        },
        monitoring_config_path=tmp_path / "monitoring.yaml",
    )

    assert config["training"]["epochs"] == 7
    assert config["training"]["batch_size"] == 128
    assert config["training"]["precision"] == "float16"
    assert config["training"]["run_test"] is False
    assert config["data"]["processed_path"] == "data/processed/custom.csv"
    assert config["carbon_tracking"]["enabled"] is True


def test_build_monitoring_config_applies_run_name_and_hardware(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_NAME", "jenkins-ddp")
    monkeypatch.setenv("HARDWARE_TRACKING", "false")

    config = build_monitoring_config({"monitoring": {"project": "MLOps"}})

    assert config["monitoring"]["run_name"] == "jenkins-ddp"
    assert config["monitoring"]["hardware_tracking_enabled"] is False
