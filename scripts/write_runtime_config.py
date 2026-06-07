import os
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import yaml

TRAINING_OVERRIDES = {
    "MODEL_VERSION": ("experiment", "model_version", str),
    "MODEL_HIDDEN_SIZE": ("model", "hidden_size", int),
    "MODEL_NUM_LAYERS": ("model", "num_layers", int),
    "EPOCHS": ("training", "epochs", int),
    "BATCH_SIZE": ("training", "batch_size", int),
    "SEQUENCE_LENGTH": ("training", "sequence_length", int),
    "LEARNING_RATE": ("training", "learning_rate", float),
    "WEIGHT_DECAY": ("training", "weight_decay", float),
    "FLOAT_PRECISION": ("training", "precision", str),
    "DATASET_PATH": ("data", "processed_path", str),
    "VALIDATION_SPLIT": ("training", "validation_split", float),
    "TEST_SPLIT": ("training", "test_split", float),
    "RANDOM_SEED": ("training", "seed", int),
    "AI_LAB_CPUS": ("training", "num_workers", int),
}


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(data, config_file, sort_keys=False)


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_override(config, env_name, section, key, caster):
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return
    config.setdefault(section, {})[key] = caster(raw_value)


def build_runtime_config(base_config, monitoring_config_path):
    config = deepcopy(base_config)

    for env_name, (section, key, caster) in TRAINING_OVERRIDES.items():
        apply_override(config, env_name, section, key, caster)

    config.setdefault("training", {})["run_train"] = env_bool("DO_TRAIN", True)
    config["training"]["run_validation"] = env_bool("DO_VALIDATE", True)
    config["training"]["run_test"] = env_bool("DO_TEST", True)
    config["training"]["distributed"] = env_bool("TRAIN_DISTRIBUTED", True)
    config["training"]["amp_enabled"] = env_bool(
        "AUTOMATIC_MIXED_PRECISION",
        bool(config["training"].get("amp_enabled", False)),
    )
    if config["training"]["amp_enabled"] and not os.getenv("FLOAT_PRECISION"):
        config["training"]["precision"] = "float16"

    config.setdefault("monitoring", {})["enabled"] = env_bool("WANDB_ENABLED", True)
    config["monitoring"]["config_path"] = str(monitoring_config_path)
    config.setdefault("carbon_tracking", {})["enabled"] = env_bool(
        "CARBON_TRACKING",
        True,
    )

    return config


def build_monitoring_config(base_monitoring_config):
    config = deepcopy(base_monitoring_config)
    monitoring = config.setdefault("monitoring", {})

    run_name = os.getenv("WANDB_RUN_NAME")
    if run_name:
        monitoring["run_name"] = run_name

    project = os.getenv("WANDB_PROJECT")
    if project:
        monitoring["project"] = project

    entity = os.getenv("WANDB_ENTITY")
    if entity:
        monitoring["entity"] = entity

    monitoring["hardware_tracking_enabled"] = env_bool("HARDWARE_TRACKING", True)
    return config


def main():
    parser = ArgumentParser()
    parser.add_argument("--base-config", default="configs/train_config.yaml")
    parser.add_argument(
        "--base-monitoring-config",
        default="configs/monitering_config.yaml",
    )
    parser.add_argument("--output", default="reports/runtime_train_config.yaml")
    parser.add_argument(
        "--monitoring-output",
        default="reports/runtime_monitoring_config.yaml",
    )
    args = parser.parse_args()

    runtime_config = build_runtime_config(
        read_yaml(args.base_config),
        monitoring_config_path=args.monitoring_output,
    )
    monitoring_config = build_monitoring_config(read_yaml(args.base_monitoring_config))

    write_yaml(args.output, runtime_config)
    write_yaml(args.monitoring_output, monitoring_config)

    print(f"Wrote runtime training config to {args.output}")
    print(f"Wrote runtime monitoring config to {args.monitoring_output}")


if __name__ == "__main__":
    main()
