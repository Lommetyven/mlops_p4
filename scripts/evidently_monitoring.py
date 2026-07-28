import argparse
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import load_train_config, resolve_device, seed_everything  # noqa: E402
from monitoring.drift_monitor import (  # noqa: E402
    build_evidently_snapshot,
    build_shift_vector,
    create_evidently_workspace,
    extract_evidently_summary,
    finalize_monitoring_rows,
    load_drift_monitoring_config,
    save_evidently_snapshot,
    select_subset_indices,
    validate_drift_monitoring_config,
    write_monitoring_csv,
    write_monitoring_dashboard,
    write_monitoring_json,
    write_monitoring_markdown,
    write_workspace_metadata,
)
from monitoring.wandb_monitor import (  # noqa: E402
    WandbMonitor,
    WandbMonitorConfig,
)
from train.dataset import (  # noqa: E402
    build_sequence_dataset,
    load_processed_dataframe,
    split_train_val_test_dataset,
)
from train.gru_model import GruModel  # noqa: E402


def safe_name(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    return normalized.strip("-") or "unversioned"


def load_model(checkpoint_path, model_config, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a model_state_dict.")
    model = GruModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_monitoring_frame(
    model,
    sequence_dataset,
    sequence_indices,
    feature_columns,
    shift_vector,
    batch_size,
    device,
):
    loader = DataLoader(
        Subset(sequence_dataset, list(map(int, sequence_indices))),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    shift_tensor = torch.as_tensor(
        np.asarray(shift_vector, dtype=np.float32),
        device=device,
    ).view(1, 1, -1)
    feature_batches = []
    prediction_batches = []
    target_batches = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            shifted_inputs = inputs + shift_tensor
            predictions = model(shifted_inputs)
            feature_batches.append(shifted_inputs[:, -1, :].cpu().numpy())
            prediction_batches.append(predictions.reshape(-1).cpu().numpy())
            target_batches.append(targets.reshape(-1).cpu().numpy())

    features = np.concatenate(feature_batches, axis=0)
    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    frame = pd.DataFrame(features, columns=list(feature_columns))
    frame["target"] = targets
    frame["prediction"] = predictions
    return frame


def monitoring_timestamps(window_count):
    final_timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    first_timestamp = final_timestamp - timedelta(days=max(0, window_count - 1))
    return [
        first_timestamp + timedelta(days=window_index)
        for window_index in range(window_count)
    ]


def run_evidently_monitoring(
    checkpoint_path,
    train_config_path,
    drift_config_path,
    output_dir,
    device_name="auto",
):
    train_config = load_train_config(train_config_path)
    if train_config.get("task", {}).get("type", "").lower() != "regression":
        raise ValueError("Evidently monitoring currently requires a regression task.")

    drift_config = load_drift_monitoring_config(drift_config_path)
    feature_columns = list(train_config["data"]["feature_columns"])
    validate_drift_monitoring_config(drift_config, feature_columns)
    seed = int(train_config["training"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)

    dataframe = load_processed_dataframe(train_config["data"]["processed_path"])
    sequence_dataset = build_sequence_dataset(
        dataframe=dataframe,
        feature_columns=feature_columns,
        target_column=train_config["data"]["target_column"],
        sequence_length=int(train_config["training"]["sequence_length"]),
    )
    train_subset, _, test_subset = split_train_val_test_dataset(
        dataset=sequence_dataset,
        validation_split=float(train_config["training"]["validation_split"]),
        test_split=float(train_config["training"]["test_split"]),
        seed=seed,
    )
    if test_subset is None:
        raise ValueError("A test split is required for current monitoring windows.")

    reference_indices = select_subset_indices(
        train_subset,
        drift_config["reference_sample_count"],
        seed,
    )
    current_indices = select_subset_indices(
        test_subset,
        drift_config["current_sample_count"],
        seed + 1,
    )
    model = load_model(checkpoint_path, train_config["model"], device)
    zero_shift = np.zeros(len(feature_columns), dtype=np.float32)
    reference_frame = build_monitoring_frame(
        model=model,
        sequence_dataset=sequence_dataset,
        sequence_indices=reference_indices,
        feature_columns=feature_columns,
        shift_vector=zero_shift,
        batch_size=drift_config["inference_batch_size"],
        device=device,
    )
    reference_standard_deviations = (
        reference_frame[feature_columns].std(ddof=0).astype(float).to_dict()
    )

    output_path = Path(output_dir)
    reports_path = output_path / "reports"
    samples_path = output_path / "window_samples"
    workspace_path = output_path / "workspace"
    reports_path.mkdir(parents=True, exist_ok=True)
    samples_path.mkdir(parents=True, exist_ok=True)
    reference_frame.to_csv(output_path / "reference_samples.csv", index=False)

    project_name = str(
        drift_config.get("evidently_project_name")
        or f"{train_config['experiment']['model_version']}-drift-monitoring"
    )
    workspace, project, project_created = create_evidently_workspace(
        workspace_path=workspace_path,
        project_name=project_name,
    )
    timestamps = monitoring_timestamps(len(drift_config["windows"]))

    rows = []
    evidently_report_paths = []
    for window_index, (window, timestamp) in enumerate(
        zip(drift_config["windows"], timestamps, strict=True)
    ):
        window_name = str(window["name"])
        window_label = str(window.get("label") or window_name)
        shift_std = float(window.get("shift_std", 0.0))
        shift_vector = build_shift_vector(
            feature_columns=feature_columns,
            reference_standard_deviations=reference_standard_deviations,
            shifted_features=drift_config["shifted_features"],
            shift_std=shift_std,
        )
        current_frame = build_monitoring_frame(
            model=model,
            sequence_dataset=sequence_dataset,
            sequence_indices=current_indices,
            feature_columns=feature_columns,
            shift_vector=shift_vector,
            batch_size=drift_config["inference_batch_size"],
            device=device,
        )
        current_frame.to_csv(
            samples_path / f"{safe_name(window_name)}.csv",
            index=False,
        )
        snapshot = build_evidently_snapshot(
            reference_frame=reference_frame,
            current_frame=current_frame,
            feature_columns=feature_columns,
            window_name=window_name,
            timestamp=timestamp,
            model_version=train_config["experiment"]["model_version"],
            drift_share_threshold=drift_config["drift_share_threshold"],
            drift_method=drift_config["drift_method"],
            drift_threshold=drift_config["drift_threshold"],
        )
        report_prefix = reports_path / safe_name(window_name)
        evidently_report_paths.extend(save_evidently_snapshot(snapshot, report_prefix))
        workspace.add_run(project.id, snapshot, include_data=False)

        row = {
            "window_index": window_index,
            "window_name": window_name,
            "window_label": window_label,
            "timestamp": timestamp.isoformat(),
            "shift_std": shift_std,
            "sample_count": len(current_frame),
            **extract_evidently_summary(snapshot, feature_columns),
        }
        rows.append(row)
        print(
            f"{window_label}: drifted={row['drifted_feature_count']}/"
            f"{len(feature_columns)} RMSE={row['rmse']:.6f}"
        )

    finalize_monitoring_rows(
        rows=rows,
        drift_share_threshold=drift_config["drift_share_threshold"],
        performance_rmse_relative_threshold=drift_config[
            "performance_rmse_relative_threshold"
        ],
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": train_config["experiment"]["model_version"],
        "checkpoint": str(Path(checkpoint_path)),
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device).upper()
        ),
        "configuration": {
            **drift_config,
            "feature_columns": feature_columns,
            "train_config_path": str(Path(train_config_path)),
            "drift_config_path": str(Path(drift_config_path)),
            "reference_sample_count_used": len(reference_frame),
            "current_sample_count_used": len(current_indices),
            "random_seed": seed,
        },
        "reference_standard_deviations": reference_standard_deviations,
        "windows": rows,
        "mitigation_policy": {
            "data_drift_only": (
                "Investigate data quality and feature distributions. Do not "
                "retrain unless model quality also degrades or the shift persists."
            ),
            "performance_decrease": (
                "Version recent validated data with DVC, retrain through Jenkins, "
                "compare against the deployed model, and keep or restore the last "
                "validated model if the challenger fails."
            ),
        },
        "evidently_workspace": {
            "path": str(workspace_path),
            "project_id": str(project.id),
            "project_name": str(project.name),
        },
    }

    artifact_paths = [
        write_monitoring_json(output_path / "monitoring_summary.json", report),
        write_monitoring_csv(output_path / "monitoring_summary.csv", rows),
        write_monitoring_markdown(output_path / "monitoring_summary.md", report),
        write_monitoring_dashboard(
            output_path / "monitoring_dashboard.png",
            rows,
            report["configuration"],
        ),
        write_workspace_metadata(
            output_path / "workspace_info.json",
            workspace_path,
            project,
            project_created,
        ),
    ]
    artifact_paths.extend(evidently_report_paths)
    return report, artifact_paths


def log_monitoring_to_wandb(report, monitoring_config_path, artifact_paths):
    monitor_config = WandbMonitorConfig.from_yaml(monitoring_config_path)
    if monitor_config.mode == "online" and not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; skipping drift-monitoring W&B logging.")
        return

    base_run_name = os.getenv("WANDB_RUN_NAME") or (
        f"monitoring-{report['model_version']}"
    )
    monitor_config.run_name = f"{safe_name(base_run_name)}-drift"
    monitor_config.group = safe_name(report["model_version"])
    monitor_config.job_type = "drift-monitoring"
    monitor_config.tags = list(
        dict.fromkeys(
            [
                *(monitor_config.tags or []),
                "monitoring",
                "evidently",
                "data-drift",
                "regression",
            ]
        )
    )
    config = report["configuration"]
    monitor = WandbMonitor(monitor_config).start(
        training_config={
            "model_version": report["model_version"],
            "drift_method": config["drift_method"],
            "drift_threshold": config["drift_threshold"],
            "drift_share_threshold": config["drift_share_threshold"],
            "performance_rmse_relative_threshold": config[
                "performance_rmse_relative_threshold"
            ],
            "shifted_features": config["shifted_features"],
            "windows": [
                {
                    "name": window["window_name"],
                    "shift_std": window["shift_std"],
                }
                for window in report["windows"]
            ],
        }
    )
    try:
        for step, row in enumerate(report["windows"]):
            metrics = {
                "monitoring/window_index": row["window_index"],
                "monitoring/shift_std": row["shift_std"],
                "monitoring/drifted_feature_count": row["drifted_feature_count"],
                "monitoring/drifted_feature_share": row["drifted_feature_share"],
                "monitoring/dataset_drift_detected": row["dataset_drift_detected"],
                "monitoring/rmse": row["rmse"],
                "monitoring/mae": row["mae"],
                "monitoring/r2": row["r2"],
                "monitoring/rmse_change_percent_from_baseline": row[
                    "rmse_change_percent_from_baseline"
                ],
                "monitoring/performance_degraded": row["performance_degraded"],
            }
            for metric_name in ("prediction_drift_score", "target_drift_score"):
                value = row.get(metric_name)
                if value is not None and math.isfinite(float(value)):
                    metrics[f"monitoring/{metric_name}"] = value
            monitor.log_metrics(metrics, step=step)

        monitor.update_summary(
            {
                "monitoring/max_drifted_feature_share": max(
                    row["drifted_feature_share"] for row in report["windows"]
                ),
                "monitoring/max_rmse_change_percent": max(
                    row["rmse_change_percent_from_baseline"]
                    for row in report["windows"]
                ),
                "monitoring/performance_degradation_detected": max(
                    row["performance_degraded"] for row in report["windows"]
                ),
                "monitoring/evidently_project_id": report["evidently_workspace"][
                    "project_id"
                ],
            }
        )
        monitor.log_files_artifact(
            files=artifact_paths,
            artifact_name=f"{safe_name(report['model_version'])}-drift-monitoring",
            artifact_type="monitoring-evidence",
            metadata={
                "drift_method": config["drift_method"],
                "drift_share_threshold": config["drift_share_threshold"],
                "shifted_features": config["shifted_features"],
            },
            aliases=["latest", "evidently"],
        )
    finally:
        monitor.finish()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Evidently data-drift and regression-quality monitoring "
            "evidence for deterministic normal, shifted, and recovery windows."
        )
    )
    parser.add_argument("--checkpoint", default="models/gru_model.pt")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument(
        "--drift-config",
        default="configs/drift_monitoring_config.yaml",
    )
    parser.add_argument(
        "--monitoring-config",
        default="configs/monitering_config.yaml",
    )
    parser.add_argument("--output-dir", default="reports/evidently")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    report, artifact_paths = run_evidently_monitoring(
        checkpoint_path=args.checkpoint,
        train_config_path=args.config,
        drift_config_path=args.drift_config,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print((Path(args.output_dir) / "monitoring_summary.md").read_text("utf-8"))
    if args.wandb:
        log_monitoring_to_wandb(
            report=report,
            monitoring_config_path=args.monitoring_config,
            artifact_paths=artifact_paths,
        )


if __name__ == "__main__":
    main()
