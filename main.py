import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from monitoring import (
    carbontracker_log_files,
    finish_carbon_tracker,
    start_carbon_tracker_if_enabled,
)
from monitoring.wandb_monitor import (
    collect_dataset_metadata,
    collect_git_metadata,
    file_sha256,
)
from train.dataset import (
    build_sequence_dataset,
    load_processed_dataframe,
    split_train_val_test_dataset,
)
from train.gru_model import GruModel

DEFAULT_FEATURE_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

DEFAULT_CONFIG = {
    "model": {
        "input_size": 16,
        "hidden_size": 800,
        "num_layers": 1,
        "output_size": 1,
    },
    "experiment": {
        "model_version": "gru-1.0.0",
    },
    "task": {
        "type": "regression",
        "class_names": None,
        "positive_class_threshold": 0.5,
    },
    "data": {
        "processed_path": "data/processed/household_power_gru.csv",
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "target_column": "target_next_hour",
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "sequence_length": 12,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "loss": "mse",
        "gradient_clip_norm": 1.0,
        "validation_split": 0.2,
        "test_split": 0.1,
        "shuffle": True,
        "num_workers": 0,
        "seed": 42,
        "device": "auto",
        "distributed": True,
        "amp_enabled": False,
        "precision": "float32",
        "run_train": True,
        "run_validation": True,
        "run_test": True,
    },
    "checkpoint": {
        "output_path": "models/gru_model.pt",
        "torchscript_output_path": "models/gru_model_torchscript.pt",
        "save_best_only": True,
    },
    "monitoring": {
        "enabled": True,
        "config_path": "configs/monitering_config.yaml",
    },
    "data_versioning": {
        "enabled": True,
        "artifact_name": "household-power-gru",
        "artifact_type": "dataset",
        "upload_dataset": False,
        "aliases": ["latest"],
    },
    "carbon_tracking": {
        "enabled": True,
        "log_dir": "reports/carbontracker",
        "log_file_prefix": "training",
        "epochs_before_pred": 1,
        "monitor_epochs": -1,
        "update_interval": 1,
        "interpretable": True,
        "stop_and_confirm": False,
        "ignore_errors": True,
        "components": "all",
        "devices_by_pid": False,
        "verbose": 1,
        "decimal_precision": 12,
    },
}


def load_train_config(config_path="configs/train_config.yaml"):
    config = deepcopy(DEFAULT_CONFIG)
    path = Path(config_path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as config_file:
            loaded_config = yaml.safe_load(config_file) or {}
        _merge_config(config, loaded_config)

    return config


def _merge_config(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed_if_requested(training_config):
    distributed_requested = bool(training_config.get("distributed", True))
    distributed_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed_requested or not distributed_env:
        return {
            "distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main": True,
        }

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return {
        "distributed": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed(distributed_context):
    if distributed_context.get("distributed") and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(device_name, distributed_context=None):
    distributed_context = distributed_context or {}
    if distributed_context.get("distributed") and torch.cuda.is_available():
        return torch.device(f"cuda:{distributed_context['local_rank']}")

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def unwrap_model(model):
    return (
        model.module
        if isinstance(model, (nn.DataParallel, DistributedDataParallel))
        else model
    )


def maybe_distribute_model(model, device, distributed_context):
    if not distributed_context.get("distributed"):
        return model

    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[distributed_context["local_rank"]],
            output_device=distributed_context["local_rank"],
        )

    return DistributedDataParallel(model)


def build_criterion(loss_name):
    loss_key = loss_name.lower()
    losses = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "l1": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
    }

    if loss_key not in losses:
        raise ValueError(f"Unsupported loss '{loss_name}'. Options: {sorted(losses)}")

    return losses[loss_key]()


def build_optimizer(model, training_config):
    optimizer_name = training_config["optimizer"].lower()
    optimizer_kwargs = {
        "lr": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
    }
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    if optimizer_name not in optimizers:
        raise ValueError(
            f"Unsupported optimizer '{training_config['optimizer']}'. "
            f"Options: {sorted(optimizers)}"
        )

    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = float(training_config.get("momentum", 0.0))

    return optimizers[optimizer_name](model.parameters(), **optimizer_kwargs)


def build_dataloaders(config, distributed_context=None):
    data_config = config["data"]
    training_config = config["training"]
    distributed_context = distributed_context or {}

    dataframe = load_processed_dataframe(data_config["processed_path"])
    dataset = build_sequence_dataset(
        dataframe=dataframe,
        feature_columns=data_config["feature_columns"],
        target_column=data_config["target_column"],
        sequence_length=int(training_config["sequence_length"]),
    )
    train_dataset, val_dataset, test_dataset = split_train_val_test_dataset(
        dataset=dataset,
        validation_split=float(training_config["validation_split"]),
        test_split=float(training_config.get("test_split", 0.0)),
        seed=int(training_config["seed"]),
    )

    loader_kwargs = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=distributed_context["world_size"],
            rank=distributed_context["rank"],
            shuffle=bool(training_config["shuffle"]),
            seed=int(training_config["seed"]),
        )
        if distributed_context.get("distributed")
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(training_config["shuffle"]) if train_sampler is None else False,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        if test_dataset is not None
        else None
    )

    split_sizes = {
        "train": len(train_dataset),
        "validation": len(val_dataset) if val_dataset is not None else 0,
        "test": len(test_dataset) if test_dataset is not None else 0,
    }

    return train_loader, val_loader, test_loader, len(dataset), split_sizes


def start_monitoring_if_enabled(config, model):
    monitoring_config = config.get("monitoring", {})
    if not monitoring_config.get("enabled", False):
        return None

    try:
        from monitoring import init_wandb_monitor

        return init_wandb_monitor(
            model=model,
            training_config=config,
            config_path=monitoring_config.get(
                "config_path",
                "configs/monitering_config.yaml",
            ),
        )
    except Exception as exc:
        print(f"W&B monitoring disabled: {exc}")
        return None


def log_dataset_version_if_enabled(config, monitor):
    if monitor is None:
        return

    data_versioning_config = config.get("data_versioning", {})
    if not data_versioning_config.get("enabled", False):
        return

    data_config = config["data"]
    monitor.log_dataset_artifact(
        dataset_path=data_config["processed_path"],
        artifact_name=data_versioning_config.get(
            "artifact_name",
            "household-power-gru",
        ),
        artifact_type=data_versioning_config.get("artifact_type", "dataset"),
        upload_dataset=bool(data_versioning_config.get("upload_dataset", False)),
        aliases=data_versioning_config.get("aliases", ["latest"]),
        metadata={
            "feature_columns": data_config["feature_columns"],
            "target_column": data_config["target_column"],
            "sequence_length": config["training"]["sequence_length"],
        },
    )


def build_tracking_metadata(
    config,
    config_path,
    model,
    sequence_count,
    split_sizes,
    device,
    distributed_context=None,
):
    distributed_context = distributed_context or {
        "distributed": False,
        "rank": 0,
        "world_size": 1,
    }
    dataset_path = config["data"]["processed_path"]
    dataset_metadata = collect_dataset_metadata(dataset_path)
    git_metadata = dataset_metadata.get("git") or collect_git_metadata()
    dvc_metadata = dataset_metadata.get("dvc") or {}
    parameter_breakdown = unwrap_model(model).parameter_breakdown()
    dataset_hash = dataset_metadata.get("dataset/sha256")
    dvc_lock_hash = dvc_metadata.get("dvc_lock_sha256")
    model_version = config.get("experiment", {}).get("model_version", "unversioned")
    config_path = Path(config_path)

    return {
        "git/commit_hash": git_metadata.get("commit"),
        "git/branch": git_metadata.get("branch"),
        "git/status_short": git_metadata.get("status_short"),
        "dataset/path": dataset_metadata.get("dataset/path"),
        "dataset/version": _short_hash(dvc_lock_hash or dataset_hash),
        "dataset/hash_sha256": dataset_hash,
        "dataset/checksum": dataset_hash,
        "dataset/size_bytes": dataset_metadata.get("dataset/size_bytes"),
        "dataset/dvc_lock_sha256": dvc_lock_hash,
        "dataset/dvc_yaml_sha256": dvc_metadata.get("dvc_yaml_sha256"),
        "model/version": model_version,
        "model/parameter_count": parameter_breakdown["total"],
        "model/input_to_gru_weights": parameter_breakdown["input_to_gru_weights"],
        "model/hidden_to_hidden_weights": parameter_breakdown[
            "hidden_to_hidden_weights"
        ],
        "model/biases": parameter_breakdown["biases"],
        "config/path": str(config_path),
        "config/sha256": file_sha256(config_path) if config_path.exists() else None,
        "training/random_seed": int(config["training"]["seed"]),
        "training/device": str(device),
        "training/distributed": 1 if distributed_context["distributed"] else 0,
        "training/rank": distributed_context["rank"],
        "training/world_size": distributed_context["world_size"],
        "training/gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "training/amp_enabled": 1 if config["training"].get("amp_enabled") else 0,
        "training/precision": config["training"].get("precision", "float32"),
        "training/sequence_count": sequence_count,
        "training/train_split_size": split_sizes["train"],
        "training/validation_split_size": split_sizes["validation"],
        "training/test_split_size": split_sizes["test"],
        "task/type": config.get("task", {}).get("type", "regression"),
        "tracking/training_curves_logged": 1,
        "tracking/validation_curves_logged": 1,
    }


def _short_hash(value):
    if not value:
        return None
    return str(value)[:12]


def log_initial_tracking_metadata(monitor, tracking_metadata):
    if monitor is not None:
        monitor.update_summary(tracking_metadata)


def log_carbon_summary(config, monitor, carbon_summary):
    if monitor is None or not carbon_summary:
        return

    metric_payload = {
        key: value
        for key, value in carbon_summary.items()
        if isinstance(value, int | float) and value is not None
    }
    monitor.log_metrics(
        metric_payload,
        step=int(config["training"]["epochs"]),
    )
    log_files = carbontracker_log_files(
        config.get("carbon_tracking", {}).get("log_dir", "reports/carbontracker")
    )
    monitor.log_files_artifact(
        files=log_files,
        artifact_name="carbontracker-logs",
        artifact_type="carbontracker",
        metadata=carbon_summary,
        aliases=["latest"],
    )


def write_model_card(
    config,
    tracking_metadata,
    final_tracking_summary,
    test_metrics,
    carbon_summary,
    output_path="reports/model_card.md",
):
    card_path = Path(output_path)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(
        render_model_card(
            config=config,
            tracking_metadata=tracking_metadata,
            final_tracking_summary=final_tracking_summary,
            test_metrics=test_metrics,
            carbon_summary=carbon_summary,
        ),
        encoding="utf-8",
    )
    return card_path


def render_model_card(
    config,
    tracking_metadata,
    final_tracking_summary,
    test_metrics,
    carbon_summary,
):
    experiment_config = config.get("experiment", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    task_config = config.get("task", {})
    model_version = experiment_config.get("model_version", "unversioned")

    sections = [
        f"# Model Card: {model_version}",
        "",
        "## Overview",
        markdown_table(
            [
                ("Model version", model_version),
                ("Task type", task_config.get("type", "regression")),
                ("Model class", "GRU"),
                ("Checkpoint", final_tracking_summary.get("model/best_checkpoint")),
                (
                    "Checkpoint SHA256",
                    final_tracking_summary.get("model/best_checkpoint_sha256"),
                ),
                ("Generated by", "Jenkins training pipeline"),
            ]
        ),
        "",
        "## Intended Use",
        (
            "This model is trained for the project energy consumption forecasting "
            "workflow. It should be treated as an experiment artifact unless a "
            "separate validation/release process promotes it for production use."
        ),
        "",
        "## Dataset",
        markdown_table(
            [
                ("Processed dataset", data_config.get("processed_path")),
                ("Target column", data_config.get("target_column")),
                ("Feature columns", ", ".join(data_config.get("feature_columns", []))),
                ("Dataset version", tracking_metadata.get("dataset/version")),
                ("Dataset SHA256", tracking_metadata.get("dataset/hash_sha256")),
                ("DVC lock SHA256", tracking_metadata.get("dataset/dvc_lock_sha256")),
                (
                    "Train rows/sequences",
                    tracking_metadata.get("training/train_split_size"),
                ),
                (
                    "Validation rows/sequences",
                    tracking_metadata.get("training/validation_split_size"),
                ),
                (
                    "Test rows/sequences",
                    tracking_metadata.get("training/test_split_size"),
                ),
            ]
        ),
        "",
        "## Model Parameters",
        markdown_table(
            [
                ("Input size", model_config.get("input_size")),
                ("Hidden size", model_config.get("hidden_size")),
                ("GRU layers", model_config.get("num_layers")),
                ("Output size", model_config.get("output_size")),
                ("Parameter count", tracking_metadata.get("model/parameter_count")),
            ]
        ),
        "",
        "## Training Configuration",
        markdown_table(
            [
                ("Epochs", training_config.get("epochs")),
                ("Batch size", training_config.get("batch_size")),
                ("Sequence length", training_config.get("sequence_length")),
                ("Learning rate", training_config.get("learning_rate")),
                ("Weight decay", training_config.get("weight_decay")),
                ("Optimizer", training_config.get("optimizer")),
                ("Loss", training_config.get("loss")),
                ("Random seed", training_config.get("seed")),
                ("Device", tracking_metadata.get("training/device")),
                ("Distributed", bool(tracking_metadata.get("training/distributed"))),
                ("World size", tracking_metadata.get("training/world_size")),
                ("GPU count", tracking_metadata.get("training/gpu_count")),
                ("Automatic mixed precision", training_config.get("amp_enabled")),
                ("Precision", tracking_metadata.get("training/precision")),
            ]
        ),
        "",
        "## Evaluation",
        markdown_table(model_card_metric_rows(final_tracking_summary, "model/")),
        "",
        markdown_table(model_card_metric_rows(test_metrics, "")),
        "",
        "## Carbon And Hardware Tracking",
        markdown_table(model_card_metric_rows(carbon_summary, "carbontracker/")),
        "",
        "## Reproducibility",
        markdown_table(
            [
                ("Git commit", tracking_metadata.get("git/commit_hash")),
                ("Git branch", tracking_metadata.get("git/branch")),
                ("Git status", tracking_metadata.get("git/status_short")),
                ("Training config", tracking_metadata.get("config/path")),
                ("Training config SHA256", tracking_metadata.get("config/sha256")),
                ("DVC YAML SHA256", tracking_metadata.get("dataset/dvc_yaml_sha256")),
            ]
        ),
        "",
        "## Limitations",
        (
            "This card is generated automatically from one pipeline run. Review "
            "dataset quality, leakage risk, deployment constraints, and monitoring "
            "requirements before using the model outside the experiment workflow."
        ),
        "",
    ]
    return "\n".join(sections)


def model_card_metric_rows(metrics, prefix):
    if not metrics:
        return [("Available", "No")]

    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if key.startswith(prefix) and is_model_card_scalar(value):
            rows.append((key.removeprefix(prefix), value))

    return rows or [("Available", "No scalar metrics")]


def is_model_card_scalar(value):
    return value is None or isinstance(value, str | int | float | bool)


def markdown_table(rows):
    clean_rows = [(str(name), format_model_card_value(value)) for name, value in rows]
    table = ["| Field | Value |", "| --- | --- |"]
    table.extend(
        f"| {escape_markdown_cell(name)} | {escape_markdown_cell(value)} |"
        for name, value in clean_rows
    )
    return "\n".join(table)


def format_model_card_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def escape_markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def load_checkpoint_into_model(path, model, device):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def log_final_tracking_outputs(
    config,
    monitor,
    model,
    test_loader,
    criterion,
    device,
    checkpoint_path,
    best_epoch,
    best_metric,
    precision="float32",
    amp_enabled=False,
):
    task_config = config.get("task", {})
    task_type = task_config.get("type", "regression").lower()
    evaluation_model = unwrap_model(model)
    checkpoint = load_checkpoint_into_model(checkpoint_path, evaluation_model, device)
    if checkpoint is not None:
        best_epoch = checkpoint.get("epoch", best_epoch)
        best_metric = checkpoint.get("best_metric", best_metric)

    test_metrics = evaluate(
        model=evaluation_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        task_config=task_config,
        precision=precision,
        amp_enabled=amp_enabled,
    )
    final_step = int(config["training"]["epochs"]) + 1
    checkpoint_hash = (
        file_sha256(Path(checkpoint_path)) if Path(checkpoint_path).exists() else None
    )
    model_version = config.get("experiment", {}).get("model_version", "unversioned")

    summary = {
        "model/best_epoch": best_epoch,
        "model/best_metric": best_metric,
        "model/best_checkpoint": str(checkpoint_path),
        "model/best_checkpoint_sha256": checkpoint_hash,
        "model/version": model_version,
        "test/available": 1 if test_metrics else 0,
    }
    if task_type == "classification":
        summary["classification/metrics_applicable"] = 1
    else:
        summary["classification/metrics_applicable"] = 0
        summary["classification/not_applicable_reason"] = "task.type is regression"

    if monitor is not None:
        if test_metrics:
            monitor.log_metrics(prefix_metrics(test_metrics, "test"), step=final_step)
            if task_type == "classification" and "confusion_matrix" in test_metrics:
                monitor.log_confusion_matrix(
                    confusion_matrix=test_metrics["confusion_matrix"],
                    class_names=test_metrics["class_names"],
                    title="test/confusion_matrix",
                    step=final_step,
                )

        monitor.update_summary(summary)
        monitor.log_model_artifact(
            checkpoint_path=checkpoint_path,
            artifact_name=f"{safe_artifact_name(model_version)}-best-checkpoint",
            artifact_type="model",
            metadata=summary,
            aliases=["latest", "best", model_version],
        )

    return test_metrics, summary


def safe_artifact_name(value):
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in str(value)
    )


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    gradient_clip_norm,
    precision="float32",
    amp_enabled=False,
    scaler=None,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_autocast = autocast_enabled(device, precision, amp_enabled)

    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_autocast):
            predictions = model(features)
            loss = criterion(predictions, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate(
    model,
    data_loader,
    criterion,
    device,
    task_config=None,
    precision="float32",
    amp_enabled=False,
):
    if data_loader is None:
        return {}

    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions_batches = []
    targets_batches = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=autocast_enabled(device, precision, amp_enabled),
            ):
                predictions = model(features)
                loss = criterion(predictions, targets)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions_batches.append(predictions.detach().cpu())
            targets_batches.append(targets.detach().cpu())

    predictions = torch.cat(predictions_batches).numpy()
    targets = torch.cat(targets_batches).numpy()
    metrics = {"loss": total_loss / max(total_samples, 1)}
    task_type = (task_config or {}).get("type", "regression").lower()

    if task_type == "classification":
        metrics.update(
            compute_classification_metrics(
                predictions=predictions,
                targets=targets,
                task_config=task_config or {},
            )
        )
    else:
        metrics.update(
            compute_regression_metrics(
                predictions=predictions,
                targets=targets,
            )
        )

    return metrics


def compute_regression_metrics(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors**2
    target_variance = np.sum((targets - np.mean(targets)) ** 2)

    metrics = {
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
    }
    metrics["r2"] = (
        float(1 - np.sum(squared_errors) / target_variance)
        if target_variance > 0
        else None
    )

    return metrics


def compute_classification_metrics(predictions, targets, task_config):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets).reshape(-1).astype(int)
    threshold = float(task_config.get("positive_class_threshold", 0.5))

    if predictions.ndim == 1 or predictions.shape[1] == 1:
        scores = 1 / (1 + np.exp(-predictions.reshape(-1)))
        predicted_classes = (scores >= threshold).astype(int)
        class_count = 2
        auc_scores = scores
    else:
        shifted = predictions - np.max(predictions, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        predicted_classes = np.argmax(probabilities, axis=1)
        class_count = predictions.shape[1]
        auc_scores = probabilities

    class_names = resolve_class_names(task_config.get("class_names"), class_count)
    confusion = build_confusion_matrix(targets, predicted_classes, class_count)
    metrics = {
        "accuracy": float(np.mean(predicted_classes == targets)),
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
    }

    per_class = compute_per_class_metrics(confusion, class_names)
    metrics.update(flatten_per_class_metrics(per_class))
    metrics.update(aggregate_classification_metrics(per_class))
    metrics["auc"] = compute_auc(targets, auc_scores)

    return metrics


def build_confusion_matrix(targets, predicted_classes, class_count):
    confusion = np.zeros((class_count, class_count), dtype=int)
    for target, prediction in zip(targets, predicted_classes, strict=True):
        if 0 <= target < class_count and 0 <= prediction < class_count:
            confusion[target, prediction] += 1
    return confusion


def resolve_class_names(class_names, class_count):
    if not class_names:
        return [str(class_index) for class_index in range(class_count)]

    resolved = [str(class_name) for class_name in class_names[:class_count]]
    while len(resolved) < class_count:
        resolved.append(str(len(resolved)))
    return resolved


def compute_per_class_metrics(confusion, class_names):
    per_class = {}
    for class_index, class_name in enumerate(class_names):
        true_positive = confusion[class_index, class_index]
        false_positive = confusion[:, class_index].sum() - true_positive
        false_negative = confusion[class_index, :].sum() - true_positive
        support = confusion[class_index, :].sum()

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[str(class_name)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    return per_class


def flatten_per_class_metrics(per_class):
    return {
        f"per_class/{class_name}/{metric_name}": metric_value
        for class_name, class_metrics in per_class.items()
        for metric_name, metric_value in class_metrics.items()
    }


def aggregate_classification_metrics(per_class):
    precision_values = [metrics["precision"] for metrics in per_class.values()]
    recall_values = [metrics["recall"] for metrics in per_class.values()]
    f1_values = [metrics["f1"] for metrics in per_class.values()]
    return {
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "f1": float(np.mean(f1_values)),
    }


def compute_auc(targets, scores):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    try:
        if np.asarray(scores).ndim == 2 and np.asarray(scores).shape[1] > 2:
            return float(roc_auc_score(targets, scores, multi_class="ovr"))
        return float(roc_auc_score(targets, scores))
    except ValueError:
        return None


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def prefix_metrics(metrics, prefix):
    return {
        f"{prefix}/{key}": value
        for key, value in metrics.items()
        if isinstance(value, int | float) and value is not None
    }


def normalize_precision(value):
    precision = str(value or "float32").lower()
    aliases = {
        "16": "float16",
        "fp16": "float16",
        "float16": "float16",
        "32": "float32",
        "fp32": "float32",
        "float32": "float32",
    }
    if precision not in aliases:
        raise ValueError("training.precision must be one of float16 or float32.")
    return aliases[precision]


def autocast_enabled(device, precision, amp_enabled=False):
    return (
        bool(amp_enabled)
        and device.type == "cuda"
        and normalize_precision(precision) == "float16"
    )


def save_checkpoint(path, model, optimizer, epoch, best_metric, config):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped_model = unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "parameter_breakdown": unwrapped_model.parameter_breakdown(),
        },
        checkpoint_path,
    )


def export_torchscript_model(model, output_path, sequence_length, device):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_model = unwrap_model(model).to(device)
    evaluation_model.eval()
    _ = sequence_length
    with torch.no_grad():
        scripted_model = torch.jit.script(evaluation_model)
    scripted_model.save(str(output_path))
    return output_path


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def main(config_path="configs/train_config.yaml"):
    config = load_train_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]
    distributed_context = setup_distributed_if_requested(training_config)
    is_main_process = distributed_context["is_main"]

    if int(model_config["input_size"]) != len(config["data"]["feature_columns"]):
        raise ValueError(
            "model.input_size must match the number of configured feature columns."
        )

    seed_everything(int(training_config["seed"]))
    device = resolve_device(training_config["device"], distributed_context)
    precision = normalize_precision(training_config.get("precision", "float32"))
    amp_enabled = bool(training_config.get("amp_enabled", False))
    run_train = bool(training_config.get("run_train", True))
    run_validation = bool(training_config.get("run_validation", True))
    run_test = bool(training_config.get("run_test", True))

    model = GruModel(**model_config).to(device)
    model = maybe_distribute_model(model, device, distributed_context)
    criterion = build_criterion(training_config["loss"])
    optimizer = build_optimizer(model, training_config)
    scaler = torch.cuda.amp.GradScaler(
        enabled=autocast_enabled(device, precision, amp_enabled)
    )
    train_loader, val_loader, test_loader, sequence_count, split_sizes = (
        build_dataloaders(config, distributed_context=distributed_context)
    )
    tracking_metadata = build_tracking_metadata(
        config=config,
        config_path=config_path,
        model=model,
        sequence_count=sequence_count,
        split_sizes=split_sizes,
        device=device,
        distributed_context=distributed_context,
    )
    config["tracking"] = tracking_metadata
    monitor = (
        start_monitoring_if_enabled(config, unwrap_model(model))
        if is_main_process
        else None
    )
    if is_main_process:
        log_initial_tracking_metadata(monitor, tracking_metadata)
        log_dataset_version_if_enabled(config, monitor)
    carbon_tracker = (
        start_carbon_tracker_if_enabled(config)
        if is_main_process and run_train
        else None
    )

    best_metric = float("inf")
    best_epoch = None
    save_best_only = bool(checkpoint_config["save_best_only"])
    checkpoint_path = checkpoint_config["output_path"]

    if is_main_process:
        print(f"Loaded config: {config_path}")
        print(f"Processed data: {config['data']['processed_path']}")
        print(f"Device: {device}")
        print(f"Distributed: {distributed_context['distributed']}")
        print(f"World size: {distributed_context['world_size']}")
        print(f"Automatic mixed precision: {amp_enabled}")
        print(f"Precision: {precision}")
        print(f"Sequences: {sequence_count}")
        print(f"Split sizes: {split_sizes}")
        print(f"Epochs: {training_config['epochs']}")
        print(f"Batch size: {training_config['batch_size']}")
        print(f"Sequence length: {training_config['sequence_length']}")
        print(f"Learning rate: {training_config['learning_rate']}")
        print(f"Parameter breakdown: {unwrap_model(model).parameter_breakdown()}")
        if carbon_tracker is not None:
            print(
                "CarbonTracker: enabled "
                f"({config['carbon_tracking']['components']}, "
                f"log_dir={config['carbon_tracking']['log_dir']})"
            )

    carbon_summary = {}
    try:
        if run_train:
            for epoch in range(1, int(training_config["epochs"]) + 1):
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                if carbon_tracker is not None:
                    carbon_tracker.epoch_start()

                train_loss = train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    gradient_clip_norm=float(training_config["gradient_clip_norm"]),
                    precision=precision,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                )

                if carbon_tracker is not None:
                    carbon_tracker.epoch_end()

                val_metrics = {}
                val_loss = None
                if is_main_process and run_validation:
                    val_metrics = evaluate(
                        model=unwrap_model(model),
                        data_loader=val_loader,
                        criterion=criterion,
                        device=device,
                        task_config=config["task"],
                        precision=precision,
                        amp_enabled=amp_enabled,
                    )
                    val_loss = val_metrics.get("loss") if val_metrics else None

                checkpoint_metric = val_loss if val_loss is not None else train_loss
                should_save = is_main_process and (
                    not save_best_only or checkpoint_metric < best_metric
                )
                if should_save:
                    best_metric = checkpoint_metric
                    best_epoch = epoch
                    save_checkpoint(
                        path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_metric=best_metric,
                        config=config,
                    )

                if distributed_context["distributed"]:
                    dist.barrier()

                metrics = prefix_metrics(
                    {key: value for key, value in val_metrics.items() if key != "loss"},
                    "val",
                )
                if monitor is not None:
                    monitor.log_training_step(
                        step=epoch,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        learning_rate=get_learning_rate(optimizer),
                        metrics=metrics,
                        model=unwrap_model(model),
                    )

                if is_main_process:
                    val_text = f"{val_loss:.6f}" if val_loss is not None else "n/a"
                    print(
                        f"Epoch {epoch}/{training_config['epochs']} "
                        f"train_loss={train_loss:.6f} val_loss={val_text}"
                    )
    finally:
        if distributed_context["distributed"]:
            dist.barrier()

        if is_main_process:
            if run_test:
                test_metrics, final_tracking_summary = log_final_tracking_outputs(
                    config=config,
                    monitor=monitor,
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    precision=precision,
                    amp_enabled=amp_enabled,
                )
            else:
                test_metrics = {}
                final_tracking_summary = {
                    "model/best_epoch": best_epoch,
                    "model/best_metric": best_metric,
                }

            carbon_summary = finish_carbon_tracker(carbon_tracker, config)
            log_carbon_summary(config, monitor, carbon_summary)
            model_card_path = write_model_card(
                config=config,
                tracking_metadata=tracking_metadata,
                final_tracking_summary=final_tracking_summary,
                test_metrics=test_metrics,
                carbon_summary=carbon_summary,
            )
            torchscript_path = export_torchscript_model(
                model=model,
                output_path=checkpoint_config.get(
                    "torchscript_output_path",
                    "models/gru_model_torchscript.pt",
                ),
                sequence_length=training_config["sequence_length"],
                device=device,
            )
            if monitor is not None:
                model_version = config.get("experiment", {}).get(
                    "model_version",
                    "unversioned",
                )
                monitor.log_files_artifact(
                    files=[model_card_path],
                    artifact_name=f"{safe_artifact_name(model_version)}-model-card",
                    artifact_type="model-card",
                    metadata={
                        "model_version": model_version,
                        "checkpoint": final_tracking_summary.get(
                            "model/best_checkpoint"
                        ),
                        "dataset_version": tracking_metadata.get("dataset/version"),
                    },
                    aliases=["latest", model_version],
                )
            if monitor is not None:
                monitor.finish()

            print(f"Saved checkpoint to: {checkpoint_path}")
            print(f"TorchScript model: {torchscript_path}")
            print(f"Model card: {model_card_path}")
            print(f"Best epoch: {final_tracking_summary['model/best_epoch']}")
            if test_metrics:
                print(f"Test metrics: {test_metrics}")
            if carbon_summary:
                print(f"CarbonTracker summary: {carbon_summary}")

        cleanup_distributed(distributed_context)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
