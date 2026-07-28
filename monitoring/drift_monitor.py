import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUALITY_METRICS = ("rmse", "mae", "r2")
DEFAULT_REACTION_PLAN = {
    "monitor": (
        "Continue monitoring because neither persistent data drift nor a "
        "meaningful performance decrease was detected."
    ),
    "investigate_data_shift": (
        "Inspect data quality, schema changes, missing values, and the drifted "
        "features. Do not retrain until the effect on model quality is known."
    ),
    "investigate_and_retrain": (
        "Investigate the drift, create a new validated DVC dataset version, "
        "retrain through Jenkins, compare the challenger with the deployed "
        "model, and retain or restore the last validated model if the "
        "challenger does not pass."
    ),
}


def load_drift_monitoring_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    return raw_config.get("drift_monitoring", raw_config)


def validate_drift_monitoring_config(config, feature_columns):
    feature_names = list(feature_columns)
    if not feature_names:
        raise ValueError("At least one model feature is required.")

    shifted_features = list(config.get("shifted_features") or [])
    if not shifted_features:
        raise ValueError("At least one shifted feature is required.")
    unknown_features = sorted(set(shifted_features) - set(feature_names))
    if unknown_features:
        raise ValueError(
            "Drift configuration contains unknown features: "
            + ", ".join(unknown_features)
        )

    windows = list(config.get("windows") or [])
    if len(windows) < 3:
        raise ValueError(
            "Configure at least three monitoring windows to demonstrate "
            "normal operation and drift."
        )
    window_names = [str(window.get("name") or "").strip() for window in windows]
    if any(not name for name in window_names):
        raise ValueError("Every monitoring window must have a name.")
    if len(set(window_names)) != len(window_names):
        raise ValueError("Monitoring window names must be unique.")

    shift_values = []
    for window in windows:
        shift_std = float(window.get("shift_std", 0.0))
        if not math.isfinite(shift_std) or shift_std < 0:
            raise ValueError("Window shift_std values must be finite and non-negative.")
        shift_values.append(shift_std)
    if not any(value == 0 for value in shift_values):
        raise ValueError("At least one unshifted control window is required.")
    if not any(value > 0 for value in shift_values):
        raise ValueError("At least one shifted monitoring window is required.")

    drift_share = float(config.get("drift_share_threshold", 0.25))
    if not 0 < drift_share <= 1:
        raise ValueError("drift_share_threshold must be in the interval (0, 1].")

    rmse_threshold = float(config.get("performance_rmse_relative_threshold", 0.10))
    if not 0 < rmse_threshold:
        raise ValueError(
            "performance_rmse_relative_threshold must be greater than zero."
        )

    sample_keys = ("reference_sample_count", "current_sample_count")
    for key in sample_keys:
        if int(config.get(key, 0)) < 2:
            raise ValueError(f"{key} must be at least 2.")

    return config


def select_subset_indices(subset, sample_count, seed):
    indices = np.asarray(subset.indices, dtype=np.int64)
    requested_count = min(int(sample_count), len(indices))
    if requested_count < 2:
        raise ValueError("The selected dataset subset must contain at least 2 samples.")
    generator = np.random.default_rng(int(seed))
    selected_positions = generator.choice(
        len(indices),
        size=requested_count,
        replace=False,
    )
    return indices[selected_positions]


def build_shift_vector(
    feature_columns,
    reference_standard_deviations,
    shifted_features,
    shift_std,
):
    feature_columns = list(feature_columns)
    shifted_features = set(shifted_features)
    shift_vector = np.zeros(len(feature_columns), dtype=np.float32)
    for index, feature_name in enumerate(feature_columns):
        if feature_name not in shifted_features:
            continue
        standard_deviation = float(reference_standard_deviations[feature_name])
        if not math.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(
                f"Cannot shift feature '{feature_name}' because its reference "
                "standard deviation is not positive and finite."
            )
        shift_vector[index] = float(shift_std) * standard_deviation
    return shift_vector


def apply_feature_shift(inputs, shift_vector):
    values = np.asarray(inputs, dtype=np.float32)
    vector = np.asarray(shift_vector, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Expected sequence inputs with three dimensions.")
    if values.shape[-1] != len(vector):
        raise ValueError("Shift vector length must match the input feature count.")
    return values + vector.reshape(1, 1, -1)


def calculate_regression_quality(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(predictions) != len(targets) or len(predictions) < 2:
        raise ValueError(
            "Predictions and targets must have the same length of at least two."
        )

    errors = predictions - targets
    squared_errors = errors**2
    target_variance = np.sum((targets - np.mean(targets)) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "mae": float(np.mean(np.abs(errors))),
        "r2": (
            float(1 - np.sum(squared_errors) / target_variance)
            if target_variance > 0
            else None
        ),
    }


def build_evidently_snapshot(
    reference_frame,
    current_frame,
    feature_columns,
    window_name,
    timestamp,
    model_version,
    drift_share_threshold,
    drift_method,
    drift_threshold,
):
    try:
        from evidently import DataDefinition, Dataset, Regression, Report
        from evidently.metrics import ValueDrift
        from evidently.presets import DataDriftPreset, RegressionPreset
    except ImportError as error:
        raise ImportError(
            "Evidently is required for drift monitoring. "
            "Install the project requirements first."
        ) from error

    numerical_columns = [*feature_columns, "target", "prediction"]
    definition = DataDefinition(
        numerical_columns=numerical_columns,
        regression=[Regression(target="target", prediction="prediction")],
    )
    reference_dataset = Dataset.from_pandas(
        reference_frame,
        data_definition=definition,
    )
    current_dataset = Dataset.from_pandas(
        current_frame,
        data_definition=definition,
    )
    report = Report(
        [
            DataDriftPreset(
                columns=list(feature_columns),
                drift_share=float(drift_share_threshold),
                num_method=str(drift_method),
                num_threshold=float(drift_threshold),
            ),
            ValueDrift(
                column="prediction",
                method=str(drift_method),
                threshold=float(drift_threshold),
            ),
            ValueDrift(
                column="target",
                method=str(drift_method),
                threshold=float(drift_threshold),
            ),
            RegressionPreset(),
        ],
        include_tests=True,
        tags=["monitoring", "drift", str(window_name)],
        metadata={
            "window": str(window_name),
            "model_version": str(model_version),
        },
        model_id=str(model_version),
        reference_id="training-reference",
        dataset_id=str(window_name),
    )
    return report.run(
        current_dataset,
        reference_dataset,
        timestamp=timestamp,
        tags=[str(window_name)],
    )


def extract_evidently_summary(snapshot, feature_columns):
    snapshot_data = snapshot.dict()
    feature_names = set(feature_columns)
    summary = {
        "drifted_feature_count": None,
        "drifted_feature_share": None,
        "prediction_drift_score": None,
        "target_drift_score": None,
        "rmse": None,
        "mae": None,
        "r2": None,
        "failed_test_count": sum(
            test.get("status") == "FAIL" for test in snapshot_data.get("tests", [])
        ),
        "feature_drift_scores": {},
        "drifted_features": [],
    }

    for metric in snapshot_data.get("metrics", []):
        config = metric.get("config") or {}
        metric_type = str(config.get("type") or "").rsplit(":", maxsplit=1)[-1]
        value = metric.get("value")
        if metric_type == "DriftedColumnsCount":
            summary["drifted_feature_count"] = int(round(float(value["count"])))
            summary["drifted_feature_share"] = float(value["share"])
        elif metric_type == "ValueDrift":
            column = config.get("column")
            if column in feature_names:
                summary["feature_drift_scores"][column] = float(value)
            elif column == "prediction":
                summary["prediction_drift_score"] = float(value)
            elif column == "target":
                summary["target_drift_score"] = float(value)
        elif metric_type == "RMSE":
            summary["rmse"] = float(value)
        elif metric_type == "MAE":
            summary["mae"] = float(value["mean"])
        elif metric_type == "R2Score":
            summary["r2"] = float(value)

    for test in snapshot_data.get("tests", []):
        metric_params = (test.get("metric_config") or {}).get("params") or {}
        metric_type = str(metric_params.get("type") or "").rsplit(":", maxsplit=1)[-1]
        column = metric_params.get("column")
        if (
            metric_type == "ValueDrift"
            and column in feature_names
            and test.get("status") == "FAIL"
        ):
            summary["drifted_features"].append(column)

    missing_metrics = [
        metric_name
        for metric_name in (
            "drifted_feature_count",
            "drifted_feature_share",
            *QUALITY_METRICS,
        )
        if summary[metric_name] is None
    ]
    if missing_metrics:
        raise ValueError(
            "Evidently report did not contain expected metrics: "
            + ", ".join(missing_metrics)
        )
    summary["drifted_features"] = sorted(set(summary["drifted_features"]))
    return summary


def save_evidently_snapshot(snapshot, report_prefix):
    prefix = Path(report_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    html_path = prefix.with_suffix(".html")
    json_path = prefix.with_suffix(".json")

    # Evidently 0.7.21 accepts strings or file handles and silently ignores a
    # pathlib.Path, so convert explicitly and verify both evidence files.
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))
    for output_path in (html_path, json_path):
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Evidently did not write the expected report: {output_path}"
            )
    return html_path, json_path


def finalize_monitoring_rows(
    rows,
    drift_share_threshold,
    performance_rmse_relative_threshold,
):
    if not rows:
        raise ValueError("At least one monitoring result is required.")
    baseline_rmse = float(rows[0]["rmse"])
    rmse_limit = baseline_rmse * (1 + float(performance_rmse_relative_threshold))

    for row in rows:
        row["dataset_drift_detected"] = int(
            float(row["drifted_feature_share"]) >= float(drift_share_threshold)
        )
        row["rmse_change_percent_from_baseline"] = (
            float(row["rmse"]) / baseline_rmse - 1
        ) * 100
        row["performance_degraded"] = int(float(row["rmse"]) > rmse_limit)
        if row["performance_degraded"]:
            row["reaction"] = "investigate_and_retrain"
        elif row["dataset_drift_detected"]:
            row["reaction"] = "investigate_data_shift"
        else:
            row["reaction"] = "monitor"
        row["reaction_description"] = DEFAULT_REACTION_PLAN[row["reaction"]]
    return rows


def write_monitoring_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_index",
        "window_name",
        "window_label",
        "shift_std",
        "sample_count",
        "drifted_feature_count",
        "drifted_feature_share",
        "dataset_drift_detected",
        "prediction_drift_score",
        "target_drift_score",
        "rmse",
        "mae",
        "r2",
        "rmse_change_percent_from_baseline",
        "performance_degraded",
        "failed_test_count",
        "drifted_features",
        "reaction",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            flattened["drifted_features"] = ", ".join(row["drifted_features"])
            writer.writerow({key: flattened.get(key) for key in fieldnames})
    return output_path


def write_monitoring_json(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def build_monitoring_markdown(report):
    config = report["configuration"]
    lines = [
        "# Evidently drift monitoring experiment",
        "",
        f"- Model version: `{report['model_version']}`",
        f"- Device: `{report['device']}`",
        (
            "- Drift method: "
            f"`{config['drift_method']}` with threshold "
            f"`{config['drift_threshold']}`"
        ),
        (
            "- Dataset drift threshold: "
            f"{100 * config['drift_share_threshold']:.0f}% of model features"
        ),
        (
            "- Performance alert: RMSE more than "
            f"{100 * config['performance_rmse_relative_threshold']:.0f}% "
            "above the first normal window"
        ),
        "- Reference/current samples are selected deterministically.",
        (
            "- Artificial drift is an additive mean shift measured in reference "
            "standard deviations and is applied to every time step."
        ),
        "",
        "## Monitored features",
        "",
        ", ".join(f"`{name}`" for name in config["feature_columns"]),
        "",
        "## Artificially shifted features",
        "",
        ", ".join(f"`{name}`" for name in config["shifted_features"]),
        "",
        "## Results",
        "",
        (
            "| Window | Shift | Drifted features | Dataset drift | RMSE | "
            "RMSE change | MAE | R2 | Reaction |"
        ),
        "|---|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in report["windows"]:
        lines.append(
            "| {label} | {shift:.2f} std | {count}/{total} ({share:.1%}) | "
            "{drift} | {rmse:.6f} | {rmse_change:+.2f}% | {mae:.6f} | "
            "{r2:.6f} | `{reaction}` |".format(
                label=row["window_label"],
                shift=float(row["shift_std"]),
                count=int(row["drifted_feature_count"]),
                total=len(config["feature_columns"]),
                share=float(row["drifted_feature_share"]),
                drift="yes" if row["dataset_drift_detected"] else "no",
                rmse=float(row["rmse"]),
                rmse_change=float(row["rmse_change_percent_from_baseline"]),
                mae=float(row["mae"]),
                r2=float(row["r2"]),
                reaction=row["reaction"],
            )
        )

    lines.extend(
        [
            "",
            "## Mitigation policy",
            "",
            "1. Inspect data quality, missing values, schema changes, and the "
            "features identified by Evidently.",
            "2. Treat feature or prediction drift as an investigation trigger, "
            "not automatic proof that the model has degraded.",
            "3. When ground truth becomes available, compare rolling RMSE, MAE, "
            "and R2 with the normal reference.",
            "4. If RMSE remains outside the accepted tolerance, create a new "
            "validated DVC dataset version and retrain through Jenkins.",
            "5. Deploy a challenger only when it passes the existing quality "
            "checks; otherwise retain or restore the last validated model.",
            "",
            "## Evidence",
            "",
            "- `monitoring_dashboard.png`: report-ready monitoring figure.",
            "- `monitoring_summary.csv`: flattened metrics for all windows.",
            "- `monitoring_summary.json`: complete configuration and results.",
            "- `reports/*.html`: interactive Evidently reports.",
            "- `reports/*.json`: machine-readable Evidently snapshots.",
            "- `workspace/`: local Evidently workspace for the monitoring UI.",
            "",
            "Start the local UI with:",
            "",
            "```bash",
            "evidently ui --workspace reports/evidently/workspace --port 8000",
            "```",
            "",
            "Then open `http://localhost:8000` and capture the dashboard for D6.4.",
            "",
        ]
    )
    return "\n".join(lines)


def write_monitoring_markdown(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_monitoring_markdown(report), encoding="utf-8")
    return output_path


def write_monitoring_dashboard(path, rows, config):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["window_label"] for row in rows]
    positions = np.arange(len(rows))
    drift_share = np.asarray(
        [float(row["drifted_feature_share"]) * 100 for row in rows]
    )
    rmse = np.asarray([float(row["rmse"]) for row in rows])
    mae = np.asarray([float(row["mae"]) for row in rows])
    r2 = np.asarray([float(row["r2"]) for row in rows])
    colors = [
        "#DC2626"
        if row["performance_degraded"]
        else "#F59E0B"
        if row["dataset_drift_detected"]
        else "#16A34A"
        for row in rows
    ]
    baseline_rmse = rmse[0]
    rmse_limit = baseline_rmse * (
        1 + float(config["performance_rmse_relative_threshold"])
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    figure.suptitle(
        "GRU Monitoring: Evidently Drift and Regression Quality",
        fontsize=20,
        fontweight="bold",
    )

    axes[0, 0].bar(positions, drift_share, color=colors)
    axes[0, 0].axhline(
        100 * float(config["drift_share_threshold"]),
        color="#DC2626",
        linestyle="--",
        label="Dataset drift threshold",
    )
    axes[0, 0].set_ylabel("Drifted model features (%)")
    axes[0, 0].set_title("Data drift")
    axes[0, 0].legend()

    axes[0, 1].plot(positions, rmse, marker="o", color="#2563EB", linewidth=2.5)
    axes[0, 1].axhline(
        rmse_limit,
        color="#DC2626",
        linestyle="--",
        label="Performance alert threshold",
    )
    axes[0, 1].set_ylabel("Test RMSE (lower is better)")
    axes[0, 1].set_title("Rolling regression error")
    axes[0, 1].legend()

    axes[1, 0].plot(positions, mae, marker="s", color="#EA580C", linewidth=2.5)
    axes[1, 0].set_ylabel("Test MAE (lower is better)")
    axes[1, 0].set_title("Mean absolute error")

    axes[1, 1].plot(positions, r2, marker="^", color="#7C3AED", linewidth=2.5)
    axes[1, 1].axhline(0, color="#64748B", linewidth=1)
    axes[1, 1].set_ylabel("Test R2 (higher is better)")
    axes[1, 1].set_title("Explained variance")

    for axis in axes.flat:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    shifted = ", ".join(config["shifted_features"])
    figure.text(
        0.5,
        0.01,
        (
            f"Artificial mean shift applied to: {shifted}. "
            "Green=normal, amber=data drift, red=performance degradation."
        ),
        ha="center",
        fontsize=10,
        color="#334155",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_evidently_workspace(workspace_path, project_name):
    try:
        from evidently.sdk.models import PanelMetric
        from evidently.sdk.panels import DashboardPanelPlot
        from evidently.ui.workspace import Workspace
    except ImportError as error:
        raise ImportError(
            "Evidently is required to create the local monitoring workspace."
        ) from error

    workspace_path = Path(workspace_path)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace = Workspace.create(workspace_path)
    existing_projects = workspace.search_project(project_name)
    if existing_projects:
        return workspace, existing_projects[0], False

    project = workspace.create_project(project_name)
    project.description = (
        "GRU data drift, prediction drift, and regression quality over "
        "simulated production windows."
    )
    panels = [
        DashboardPanelPlot(
            title="Drifted model feature share",
            subtitle="Share of the 16 input features detected as drifted.",
            size="half",
            values=[
                PanelMetric(
                    legend="Drifted share",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                )
            ],
            plot_params={"plot_type": "line"},
        ),
        DashboardPanelPlot(
            title="Prediction drift",
            subtitle="KS p-value for the model prediction distribution.",
            size="half",
            values=[
                PanelMetric(
                    legend="Prediction drift score",
                    metric="ValueDrift",
                    metric_labels={"column": "prediction"},
                )
            ],
            plot_params={"plot_type": "line"},
        ),
        DashboardPanelPlot(
            title="RMSE",
            subtitle="Regression error for each monitoring window.",
            size="half",
            values=[
                PanelMetric(
                    legend="RMSE",
                    metric="RMSE",
                    metric_labels={"regression_name": "default"},
                )
            ],
            plot_params={"plot_type": "line"},
        ),
        DashboardPanelPlot(
            title="MAE and R2",
            subtitle="Additional regression quality metrics.",
            size="half",
            values=[
                PanelMetric(
                    legend="MAE",
                    metric="MAE",
                    metric_labels={"regression_name": "default"},
                ),
                PanelMetric(
                    legend="R2",
                    metric="R2Score",
                    metric_labels={"regression_name": "default"},
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
    ]
    for panel in panels:
        project.dashboard.add_panel(panel)
    project.save()
    return workspace, project, True


def write_workspace_metadata(path, workspace_path, project, created):
    metadata = {
        "workspace": str(Path(workspace_path)),
        "project_id": str(project.id),
        "project_name": str(project.name),
        "project_created_by_this_run": bool(created),
        "ui_command": (f"evidently ui --workspace {Path(workspace_path)} --port 8000"),
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    return write_monitoring_json(path, metadata)
