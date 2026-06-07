from pathlib import Path

import pytest
import yaml


def test_train_config_yaml_has_required_sections():
    with open("configs/train_config.yaml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert set(config) == {
        "model",
        "experiment",
        "task",
        "data",
        "training",
        "checkpoint",
        "monitoring",
        "data_versioning",
        "carbon_tracking",
    }
    assert config["model"] == {
        "input_size": 16,
        "hidden_size": 800,
        "num_layers": 1,
        "output_size": 1,
    }
    assert config["experiment"]["model_version"] == "gru-1.0.0"
    assert config["task"]["type"] == "regression"
    assert len(config["data"]["feature_columns"]) == config["model"]["input_size"]
    assert config["data"]["target_column"] == "target_next_hour"
    assert config["data_versioning"]["enabled"] is True
    assert config["data_versioning"]["artifact_type"] == "dataset"
    assert config["carbon_tracking"]["enabled"] is True
    assert config["carbon_tracking"]["log_dir"] == "reports/carbontracker"

    training = config["training"]
    for key in (
        "epochs",
        "batch_size",
        "sequence_length",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "loss",
        "gradient_clip_norm",
        "validation_split",
        "test_split",
        "shuffle",
        "num_workers",
        "seed",
        "device",
        "distributed",
        "precision",
        "run_train",
        "run_validation",
        "run_test",
    ):
        assert key in training


def test_load_train_config_merges_defaults(tmp_path):
    pytest.importorskip("torch")

    from main import load_train_config

    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        """
training:
  epochs: 3
  batch_size: 4
  learning_rate: 0.01
""",
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config["training"]["epochs"] == 3
    assert config["training"]["batch_size"] == 4
    assert config["training"]["learning_rate"] == 0.01
    assert config["training"]["sequence_length"] == 12
    assert config["model"]["input_size"] == 16
    assert config["task"]["type"] == "regression"
    assert config["checkpoint"]["output_path"] == "models/gru_model.pt"
    assert config["data_versioning"]["artifact_name"] == "household-power-gru"
    assert config["carbon_tracking"]["enabled"] is True


def test_default_train_config_file_exists():
    assert Path("configs/train_config.yaml").is_file()


def test_build_criterion_supports_configured_losses():
    pytest.importorskip("torch")

    from main import build_criterion

    assert build_criterion("mse").__class__.__name__ == "MSELoss"
    assert build_criterion("mae").__class__.__name__ == "L1Loss"


def test_build_optimizer_supports_configured_optimizer():
    torch = pytest.importorskip("torch")

    from main import build_optimizer

    model = torch.nn.Linear(1, 1)
    optimizer = build_optimizer(
        model,
        {
            "optimizer": "adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
    )

    assert optimizer.__class__.__name__ == "Adam"
