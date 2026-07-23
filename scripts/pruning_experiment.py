import argparse
import csv
import json
import math
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
import torch
import torch.nn.utils.prune as prune

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402
    build_criterion,
    build_dataloaders,
    build_optimizer,
    evaluate,
    load_train_config,
    normalize_precision,
    scaler_enabled,
    seed_everything,
    train_one_epoch,
)
from monitoring.wandb_monitor import (  # noqa: E402
    WandbMonitor,
    WandbMonitorConfig,
)
from train.gru_model import GruModel  # noqa: E402

DEFAULT_PRUNING_LEVELS = "0,10,25,50,75,90"
PRUNABLE_PARAMETER_NAMES = (
    "input_to_gru_weights",
    "hidden_to_hidden_weights",
)
QUALITY_METRICS = ("rmse", "mae", "r2")


def parse_pruning_levels(raw_levels):
    values = [
        token for token in re.split(r"[,;:\s]+", str(raw_levels or "").strip()) if token
    ]
    if not values:
        raise ValueError("At least one pruning level is required.")

    levels = sorted({float(value) for value in values})
    if any(not math.isfinite(level) for level in levels):
        raise ValueError("Pruning levels must be finite numbers.")
    if levels[0] < 0 or levels[-1] >= 100:
        raise ValueError("Pruning levels must be at least 0 and less than 100.")
    if 0.0 not in levels:
        raise ValueError("Pruning levels must include 0 as the unpruned baseline.")
    return levels


def load_checkpoint_state(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a model_state_dict.")
    return checkpoint


def build_baseline_model(config, model_state, device):
    model = GruModel(**config["model"])
    model.load_state_dict(model_state)
    return model.to(device)


def apply_global_unstructured_pruning(model, pruning_percentage):
    amount = float(pruning_percentage) / 100.0
    parameters = [
        (model, parameter_name) for parameter_name in PRUNABLE_PARAMETER_NAMES
    ]
    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return measure_prunable_sparsity(model)


def measure_prunable_sparsity(model):
    total = 0
    nonzero = 0
    by_parameter = {}
    for parameter_name in PRUNABLE_PARAMETER_NAMES:
        parameter = getattr(model, parameter_name).detach()
        parameter_total = parameter.numel()
        parameter_nonzero = int(torch.count_nonzero(parameter).item())
        total += parameter_total
        nonzero += parameter_nonzero
        by_parameter[parameter_name] = {
            "total": parameter_total,
            "nonzero": parameter_nonzero,
            "zero": parameter_total - parameter_nonzero,
            "sparsity_percent": (
                100.0 * (parameter_total - parameter_nonzero) / parameter_total
            ),
        }

    zero = total - nonzero
    return {
        "total": total,
        "nonzero": nonzero,
        "zero": zero,
        "sparsity_percent": 100.0 * zero / total,
        "dense_equivalent_compression_ratio": (
            float(total / nonzero) if nonzero else None
        ),
        "by_parameter": by_parameter,
    }


def fine_tune_pruned_model(
    model,
    train_loader,
    validation_loader,
    criterion,
    training_config,
    device,
    epochs,
    learning_rate,
):
    fine_tune_config = deepcopy(training_config)
    fine_tune_config["learning_rate"] = float(learning_rate)
    optimizer = build_optimizer(model, fine_tune_config)
    precision = normalize_precision(fine_tune_config.get("precision", "float32"))
    amp_enabled = bool(fine_tune_config.get("amp_enabled", False))
    scaler = torch.amp.GradScaler(
        device.type, enabled=scaler_enabled(device, precision, amp_enabled)
    )
    best_validation_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []

    for epoch in range(1, int(epochs) + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=float(fine_tune_config.get("gradient_clip_norm", 0.0)),
            precision=precision,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        validation_metrics = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            task_config={"type": "regression"},
            precision=precision,
            amp_enabled=amp_enabled,
        )
        validation_loss = validation_metrics.get("loss", train_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "validation_rmse": validation_metrics.get("rmse"),
                "validation_mae": validation_metrics.get("mae"),
                "validation_r2": validation_metrics.get("r2"),
            }
        )
        print(
            f"  fine-tune epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Fine-tuning did not produce a model state.")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }


def _quality_metrics(metrics):
    return {
        metric_name: (
            None if metrics.get(metric_name) is None else float(metrics[metric_name])
        )
        for metric_name in ("loss", *QUALITY_METRICS)
    }


def _metric_delta(value, baseline):
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def add_baseline_deltas(results):
    baseline = next(result for result in results if result["pruning_percentage"] == 0.0)
    for result in results:
        for phase in ("immediate", "fine_tuned"):
            result[f"{phase}_delta_from_0_percent"] = {
                metric: _metric_delta(
                    result[phase].get(metric),
                    baseline[phase].get(metric),
                )
                for metric in QUALITY_METRICS
            }
    return results


def run_pruning_experiment(
    checkpoint_path,
    config_path,
    pruning_levels=DEFAULT_PRUNING_LEVELS,
    fine_tune_epochs=5,
    fine_tune_learning_rate=0.0001,
    device_name="auto",
):
    levels = parse_pruning_levels(pruning_levels)
    if int(fine_tune_epochs) < 1:
        raise ValueError("Fine-tune epochs must be at least 1.")
    if (
        not math.isfinite(float(fine_tune_learning_rate))
        or float(fine_tune_learning_rate) <= 0
    ):
        raise ValueError("Fine-tune learning rate must be greater than 0.")

    config = load_train_config(config_path)
    if config.get("task", {}).get("type", "regression").lower() != "regression":
        raise ValueError("The pruning experiment currently requires a regression task.")

    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    checkpoint = load_checkpoint_state(checkpoint_path)
    training_config = deepcopy(config["training"])
    training_config["distributed"] = False
    precision = normalize_precision(training_config.get("precision", "float32"))
    amp_enabled = bool(training_config.get("amp_enabled", False))
    seed = int(training_config["seed"])
    seed_everything(seed)
    train_loader, validation_loader, test_loader, _, split_sizes = build_dataloaders(
        config
    )
    if validation_loader is None:
        raise ValueError("A validation split is required for fine-tuning selection.")
    if test_loader is None:
        raise ValueError("A test split is required for pruning evaluation.")
    criterion = build_criterion(training_config["loss"])

    results = []
    for level in levels:
        print(f"Pruning level: {format_percentage(level)}%")
        seed_everything(seed)
        model = build_baseline_model(
            config,
            checkpoint["model_state_dict"],
            device,
        )
        sparsity_before = apply_global_unstructured_pruning(model, level)
        immediate_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            task_config=config["task"],
            precision=precision,
            amp_enabled=amp_enabled,
        )
        print(
            "  immediate: "
            f"RMSE={_format_metric(immediate_metrics['rmse'])}, "
            f"MAE={_format_metric(immediate_metrics['mae'])}, "
            f"R2={_format_metric(immediate_metrics['r2'])}"
        )
        fine_tuning = fine_tune_pruned_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            training_config=training_config,
            device=device,
            epochs=fine_tune_epochs,
            learning_rate=fine_tune_learning_rate,
        )
        fine_tuned_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            task_config=config["task"],
            precision=precision,
            amp_enabled=amp_enabled,
        )
        sparsity_after = measure_prunable_sparsity(model)
        print(
            "  fine-tuned: "
            f"RMSE={_format_metric(fine_tuned_metrics['rmse'])}, "
            f"MAE={_format_metric(fine_tuned_metrics['mae'])}, "
            f"R2={_format_metric(fine_tuned_metrics['r2'])}"
        )
        results.append(
            {
                "pruning_percentage": float(level),
                "sparsity_before_fine_tuning": sparsity_before,
                "sparsity_after_fine_tuning": sparsity_after,
                "immediate": _quality_metrics(immediate_metrics),
                "fine_tuned": _quality_metrics(fine_tuned_metrics),
                "fine_tuning": fine_tuning,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    add_baseline_deltas(results)
    return {
        "schema_version": 1,
        "model_version": config.get("experiment", {}).get(
            "model_version",
            "unversioned",
        ),
        "checkpoint": str(checkpoint_path),
        "task": "regression",
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "precision": precision,
        "pruning_method": "global_unstructured_l1_magnitude",
        "pruning_structure": "unstructured_individual_weights",
        "pruned_parameters": list(PRUNABLE_PARAMETER_NAMES),
        "biases_pruned": False,
        "pruning_levels_percent": levels,
        "fine_tune_epochs": int(fine_tune_epochs),
        "fine_tune_learning_rate": float(fine_tune_learning_rate),
        "random_seed": seed,
        "split_sizes": split_sizes,
        "test_dataset_reused_for_every_evaluation": True,
        "validation_selection_metric": "loss",
        "results": results,
        "limitations": [
            (
                "Unstructured sparsity does not automatically reduce dense "
                "PyTorch checkpoint size or inference latency."
            ),
            (
                "Actual speedup requires a sparse storage format and hardware "
                "kernels that exploit the pruning pattern."
            ),
        ],
    }


def format_percentage(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _format_metric(value):
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def build_markdown(report):
    lines = [
        "# GRU Pruning Experiment",
        "",
        f"- Model version: `{report['model_version']}`",
        f"- Device: `{report['device']}`",
        "- Method: global unstructured L1-magnitude pruning",
        "- Pruning unit: individual weights",
        "- Pruned tensors: input-to-GRU and hidden-to-hidden weights",
        "- Biases pruned: no",
        (
            f"- Fine-tuning: {report['fine_tune_epochs']} epoch(s) at learning "
            f"rate {report['fine_tune_learning_rate']:g}"
        ),
        (
            "- Model selection: lowest validation loss; the test split is reused "
            "unchanged for every reported evaluation"
        ),
        "",
        "| Pruning | Actual sparsity | Nonzero weights | Immediate RMSE | "
        "Fine-tuned RMSE | Immediate MAE | Fine-tuned MAE | Immediate R2 | "
        "Fine-tuned R2 | Best fine-tune epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        sparsity = result["sparsity_after_fine_tuning"]
        lines.append(
            "| {level}% | {sparsity:.2f}% | {nonzero} | {immediate_rmse} | "
            "{fine_rmse} | {immediate_mae} | {fine_mae} | {immediate_r2} | "
            "{fine_r2} | {epoch} |".format(
                level=format_percentage(result["pruning_percentage"]),
                sparsity=sparsity["sparsity_percent"],
                nonzero=sparsity["nonzero"],
                immediate_rmse=_format_metric(result["immediate"]["rmse"]),
                fine_rmse=_format_metric(result["fine_tuned"]["rmse"]),
                immediate_mae=_format_metric(result["immediate"]["mae"]),
                fine_mae=_format_metric(result["fine_tuned"]["mae"]),
                immediate_r2=_format_metric(result["immediate"]["r2"]),
                fine_r2=_format_metric(result["fine_tuned"]["r2"]),
                epoch=result["fine_tuning"]["best_epoch"],
            )
        )
    lines.extend(
        [
            "",
            "RMSE and MAE are lower-is-better. R2 is higher-is-better.",
            "",
            "The 0% model is evaluated both before and after the same fine-tuning "
            "budget, providing a control for improvement caused by additional "
            "training.",
            "",
            "## Limitation",
            "",
            " ".join(report["limitations"]),
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def write_csv(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pruning_percentage",
        "actual_sparsity_percent",
        "prunable_weights_total",
        "prunable_weights_nonzero",
        "dense_equivalent_compression_ratio",
        "immediate_rmse",
        "fine_tuned_rmse",
        "immediate_mae",
        "fine_tuned_mae",
        "immediate_r2",
        "fine_tuned_r2",
        "fine_tune_best_epoch",
        "fine_tune_best_validation_loss",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in report["results"]:
            sparsity = result["sparsity_after_fine_tuning"]
            writer.writerow(
                {
                    "pruning_percentage": result["pruning_percentage"],
                    "actual_sparsity_percent": sparsity["sparsity_percent"],
                    "prunable_weights_total": sparsity["total"],
                    "prunable_weights_nonzero": sparsity["nonzero"],
                    "dense_equivalent_compression_ratio": sparsity[
                        "dense_equivalent_compression_ratio"
                    ],
                    "immediate_rmse": result["immediate"]["rmse"],
                    "fine_tuned_rmse": result["fine_tuned"]["rmse"],
                    "immediate_mae": result["immediate"]["mae"],
                    "fine_tuned_mae": result["fine_tuned"]["mae"],
                    "immediate_r2": result["immediate"]["r2"],
                    "fine_tuned_r2": result["fine_tuned"]["r2"],
                    "fine_tune_best_epoch": result["fine_tuning"]["best_epoch"],
                    "fine_tune_best_validation_loss": result["fine_tuning"][
                        "best_validation_loss"
                    ],
                }
            )
    return output_path


def write_markdown(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(report), encoding="utf-8")
    return output_path


def write_plot(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    percentages = [result["pruning_percentage"] for result in report["results"]]
    metric_specs = (
        ("rmse", "Test RMSE (lower is better)"),
        ("mae", "Test MAE (lower is better)"),
        ("r2", "Test R2 (higher is better)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (metric, ylabel) in zip(axes, metric_specs, strict=True):
        immediate = [result["immediate"][metric] for result in report["results"]]
        fine_tuned = [result["fine_tuned"][metric] for result in report["results"]]
        axis.plot(
            percentages,
            immediate,
            color="#0072B2",
            marker="o",
            linewidth=2,
            label="Immediately after pruning",
        )
        axis.plot(
            percentages,
            fine_tuned,
            color="#D55E00",
            marker="s",
            linewidth=2,
            label="After fine-tuning",
        )
        axis.set_xlabel("Pruning percentage")
        axis.set_ylabel(ylabel)
        axis.set_xticks(percentages)
        axis.grid(True, color="#D9D9D9", linewidth=0.8)
    axes[0].legend(frameon=False)
    figure.suptitle("GRU quality across unstructured pruning levels")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def safe_name(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    return normalized.strip("-") or "unversioned"


def log_report_to_wandb(report, monitoring_config_path, artifact_paths):
    monitor_config = WandbMonitorConfig.from_yaml(monitoring_config_path)
    if monitor_config.mode == "online" and not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; skipping pruning W&B logging.")
        return

    base_run_name = os.getenv("WANDB_RUN_NAME") or (
        f"pruning-{report['model_version']}"
    )
    monitor_config.run_name = f"{safe_name(base_run_name)}-pruning"
    monitor_config.group = safe_name(report["model_version"])
    monitor_config.job_type = "pruning-experiment"
    monitor_config.tags = list(
        dict.fromkeys(
            [
                *(monitor_config.tags or []),
                "pruning",
                "unstructured",
                "l1-magnitude",
                "regression",
            ]
        )
    )
    run_config = {
        key: value
        for key, value in report.items()
        if key not in {"results", "limitations"}
    }
    monitor = WandbMonitor(monitor_config).start(training_config=run_config)
    try:
        for step, result in enumerate(report["results"]):
            metrics = {
                "pruning/percentage": result["pruning_percentage"],
                "pruning/actual_sparsity_percent": result["sparsity_after_fine_tuning"][
                    "sparsity_percent"
                ],
                "pruning/nonzero_weights": result["sparsity_after_fine_tuning"][
                    "nonzero"
                ],
            }
            for phase in ("immediate", "fine_tuned"):
                for metric_name in QUALITY_METRICS:
                    value = result[phase].get(metric_name)
                    if value is not None and math.isfinite(value):
                        metrics[f"pruning/{phase}/{metric_name}"] = value
            monitor.log_metrics(metrics, step=step)
        monitor.update_summary(
            {
                "pruning/method": report["pruning_method"],
                "pruning/structure": report["pruning_structure"],
                "pruning/max_percentage": max(report["pruning_levels_percent"]),
                "pruning/fine_tune_epochs": report["fine_tune_epochs"],
            }
        )
        monitor.log_files_artifact(
            files=artifact_paths,
            artifact_name=f"{safe_name(report['model_version'])}-pruning",
            artifact_type="pruning-experiment",
            metadata={
                "method": report["pruning_method"],
                "levels": report["pruning_levels_percent"],
            },
            aliases=["latest", "unstructured"],
        )
    finally:
        monitor.finish()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate global unstructured L1-magnitude pruning before and after "
            "mask-preserving fine-tuning."
        )
    )
    parser.add_argument("--checkpoint", default="models/gru_model.pt")
    parser.add_argument("--config", default="reports/runtime_train_config.yaml")
    parser.add_argument(
        "--monitoring-config",
        default="reports/runtime_monitoring_config.yaml",
    )
    parser.add_argument("--levels", default=DEFAULT_PRUNING_LEVELS)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=0.0001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--json-output",
        default="reports/pruning_experiment.json",
    )
    parser.add_argument(
        "--csv-output",
        default="reports/pruning_experiment.csv",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/pruning_experiment.md",
    )
    parser.add_argument(
        "--plot-output",
        default="reports/pruning_metrics.png",
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    report = run_pruning_experiment(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        pruning_levels=args.levels,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        device_name=args.device,
    )
    json_path = write_json(args.json_output, report)
    csv_path = write_csv(args.csv_output, report)
    markdown_path = write_markdown(args.markdown_output, report)
    plot_path = write_plot(args.plot_output, report)
    print(build_markdown(report))
    if args.wandb:
        log_report_to_wandb(
            report,
            args.monitoring_config,
            [json_path, csv_path, markdown_path, plot_path],
        )


if __name__ == "__main__":
    main()
