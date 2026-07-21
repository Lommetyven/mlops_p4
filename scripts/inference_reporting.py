import json
import math
import os
import re
from argparse import ArgumentParser
from pathlib import Path

try:
    from monitoring.wandb_monitor import WandbMonitor, WandbMonitorConfig
except ModuleNotFoundError:
    from pathlib import Path as _Path
    from sys import path as _python_path

    _python_path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from monitoring.wandb_monitor import WandbMonitor, WandbMonitorConfig


def count_csv_rows(path):
    with open(path, "r", encoding="utf-8") as input_file:
        line_count = sum(1 for line in input_file if line.strip())
    return max(0, line_count - 1)


def count_predictions(path):
    prediction_path = Path(path)
    if not prediction_path.is_file():
        return 0

    count = 0
    with prediction_path.open("r", encoding="utf-8") as prediction_file:
        for line in prediction_file:
            value = line.strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                continue
            count += 1
    return count


def tail_text(path, max_characters=2000):
    log_path = Path(path) if path else None
    if log_path is None or not log_path.is_file():
        return None
    return log_path.read_text(encoding="utf-8", errors="replace")[-max_characters:]


def build_inference_metrics(
    input_path,
    predictions_path,
    precision,
    batch_size,
    sequence_length,
    runtime_seconds,
    status,
    runner,
    backend,
    device,
    model_version="",
    error=None,
    extra_metrics=None,
):
    input_rows = count_csv_rows(input_path)
    expected_windows = max(0, input_rows - int(sequence_length) + 1)
    prediction_count = count_predictions(predictions_path)
    runtime_seconds = max(0.0, float(runtime_seconds))
    throughput = prediction_count / runtime_seconds if runtime_seconds > 0 else 0.0

    metrics = {
        "inference/status": status,
        "inference/completed": int(status == "success"),
        "inference/runner": runner,
        "inference/backend": backend,
        "inference/device": device,
        "inference/precision": precision.upper(),
        "inference/model_version": model_version or "unversioned",
        "inference/input_rows": input_rows,
        "inference/expected_windows": expected_windows,
        "inference/prediction_count": prediction_count,
        "inference/batch_size": int(batch_size),
        "inference/sequence_length": int(sequence_length),
        "inference/runtime_seconds": runtime_seconds,
        "inference/throughput_windows_per_second": throughput,
    }
    if error:
        metrics["inference/error"] = str(error)
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def write_metrics(path, metrics):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def safe_name(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    return normalized.strip("-") or "unversioned"


def start_wandb_inference(metrics, monitoring_config_path):
    monitor_config = WandbMonitorConfig.from_yaml(monitoring_config_path)
    if monitor_config.mode == "online" and not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; skipping inference W&B logging.")
        return None

    base_run_name = os.getenv("WANDB_RUN_NAME") or (
        f"inference-{metrics['inference/model_version']}"
    )
    monitor_config.run_name = (
        f"{safe_name(base_run_name)}-{metrics['inference/precision'].lower()}"
    )
    monitor_config.group = safe_name(metrics["inference/model_version"])
    monitor_config.job_type = "inference"
    monitor_config.tags = list(
        dict.fromkeys(
            [
                *(monitor_config.tags or []),
                "inference",
                metrics["inference/precision"].lower(),
                safe_name(metrics["inference/runner"]).lower(),
            ]
        )
    )
    return WandbMonitor(monitor_config).start(training_config=metrics)


def finish_wandb_inference(monitor, metrics, metrics_path=None):
    if monitor is None:
        return

    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    monitor.log_metrics(numeric_metrics)
    monitor.log_hardware(force=True)
    monitor.update_summary(metrics)
    if metrics_path:
        monitor.log_files_artifact(
            files=[metrics_path],
            artifact_name=(
                f"{safe_name(metrics['inference/model_version'])}-"
                f"{metrics['inference/precision'].lower()}-inference-metrics"
            ),
            artifact_type="inference-metrics",
            metadata=metrics,
            aliases=["latest", metrics["inference/precision"].lower()],
        )
    monitor.finish()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--runtime-seconds", type=float, required=True)
    parser.add_argument("--status", choices=("success", "failed"), required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", default="unknown")
    parser.add_argument("--model-version", default="")
    parser.add_argument("--error-log")
    parser.add_argument(
        "--monitoring-config",
        default="reports/runtime_monitoring_config.yaml",
    )
    parser.add_argument("--output", default="reports/inference_metrics.json")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    metrics = build_inference_metrics(
        input_path=args.input,
        predictions_path=args.predictions,
        precision=args.precision,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        runtime_seconds=args.runtime_seconds,
        status=args.status,
        runner=args.runner,
        backend=args.backend,
        device=args.device,
        model_version=args.model_version,
        error=tail_text(args.error_log) if args.status == "failed" else None,
    )
    metrics_path = write_metrics(args.output, metrics)
    if args.wandb:
        monitor = start_wandb_inference(metrics, args.monitoring_config)
        finish_wandb_inference(monitor, metrics, metrics_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
