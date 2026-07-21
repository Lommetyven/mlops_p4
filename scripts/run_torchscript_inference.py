import json
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import yaml

try:
    from scripts.inference_reporting import (
        build_inference_metrics,
        finish_wandb_inference,
        start_wandb_inference,
        write_metrics,
    )
except ModuleNotFoundError:
    from inference_reporting import (
        build_inference_metrics,
        finish_wandb_inference,
        start_wandb_inference,
        write_metrics,
    )


def load_model_version(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config.get("experiment", {}).get("model_version", "unversioned")


def resolve_precision(precision, device):
    normalized = precision.upper()
    if normalized == "FP32":
        return torch.float32
    if normalized == "FP16":
        if device.type != "cuda":
            raise RuntimeError("FP16 inference requires a CUDA GPU.")
        return torch.float16
    if normalized == "FP8":
        raise RuntimeError(
            "FP8 inference is not supported by the current TorchScript GRU backend."
        )
    raise ValueError("Inference precision must be FP32, FP16, or FP8.")


def write_predictions(path, predictions):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(f"{float(prediction)}\n")
    return output_path


def run_torchscript_inference(
    model_path,
    input_path,
    predictions_path,
    metrics_path,
    precision,
    sequence_length,
    batch_size,
    config_path,
    monitoring_config_path,
    runner="AI_LAB",
    enable_wandb=False,
):
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("Sequence length and batch size must be greater than zero.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_version = load_model_version(config_path)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    run_context = build_inference_metrics(
        input_path=input_path,
        predictions_path=predictions_path,
        precision=precision,
        batch_size=batch_size,
        sequence_length=sequence_length,
        runtime_seconds=0.0,
        status="running",
        runner=runner,
        backend="python-torchscript",
        device=device_name,
        model_version=model_version,
        extra_metrics={
            "inference/cuda_available": int(torch.cuda.is_available()),
            "inference/gpu_count": torch.cuda.device_count(),
        },
    )
    monitor = (
        start_wandb_inference(run_context, monitoring_config_path)
        if enable_wandb
        else None
    )

    started_at = None
    try:
        dtype = resolve_precision(precision, device)
        dataframe = pd.read_csv(input_path)
        features = torch.from_numpy(dataframe.to_numpy(dtype="float32", copy=True))
        if len(features) < sequence_length:
            raise ValueError(
                f"Input has {len(features)} rows but sequence length is "
                f"{sequence_length}."
            )

        windows = features.unfold(0, sequence_length, 1).permute(0, 2, 1)
        window_count = len(windows)
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        model.to(device=device, dtype=dtype)

        warmup_size = min(batch_size, window_count)
        with torch.inference_mode():
            warmup = windows[:warmup_size].contiguous().to(device, dtype=dtype)
            model(warmup)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            started_at = time.perf_counter()
            prediction_batches = []
            for batch_start in range(0, window_count, batch_size):
                batch = (
                    windows[batch_start : batch_start + batch_size]
                    .contiguous()
                    .to(device, dtype=dtype)
                )
                output = model(batch)
                prediction_batches.append(output.flatten().float().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_seconds = time.perf_counter() - started_at

        predictions = torch.cat(prediction_batches).tolist()
        write_predictions(predictions_path, predictions)
        extra_metrics = {
            "inference/cuda_available": int(torch.cuda.is_available()),
            "inference/gpu_count": torch.cuda.device_count(),
            "inference/warmup_windows": warmup_size,
        }
        if device.type == "cuda":
            extra_metrics["inference/max_cuda_memory_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
        metrics = build_inference_metrics(
            input_path=input_path,
            predictions_path=predictions_path,
            precision=precision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            runtime_seconds=runtime_seconds,
            status="success",
            runner=runner,
            backend="python-torchscript",
            device=device_name,
            model_version=model_version,
            extra_metrics=extra_metrics,
        )
    except Exception as error:
        runtime_seconds = (
            time.perf_counter() - started_at if started_at is not None else 0.0
        )
        metrics = build_inference_metrics(
            input_path=input_path,
            predictions_path=predictions_path,
            precision=precision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            runtime_seconds=runtime_seconds,
            status="failed",
            runner=runner,
            backend="python-torchscript",
            device=device_name,
            model_version=model_version,
            error=error,
            extra_metrics={
                "inference/cuda_available": int(torch.cuda.is_available()),
                "inference/gpu_count": torch.cuda.device_count(),
            },
        )
        metrics_output = write_metrics(metrics_path, metrics)
        finish_wandb_inference(monitor, metrics, metrics_output)
        raise

    metrics_output = write_metrics(metrics_path, metrics)
    finish_wandb_inference(monitor, metrics, metrics_output)
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--config", default="reports/runtime_train_config.yaml")
    parser.add_argument(
        "--monitoring-config",
        default="reports/runtime_monitoring_config.yaml",
    )
    parser.add_argument("--runner", default="AI_LAB")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    metrics = run_torchscript_inference(
        model_path=args.model,
        input_path=args.input,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        precision=args.precision,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        config_path=args.config,
        monitoring_config_path=args.monitoring_config,
        runner=args.runner,
        enable_wandb=args.wandb,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
