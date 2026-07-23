import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
import torch

try:
    from monitoring.wandb_monitor import WandbMonitor, WandbMonitorConfig
    from scripts.inference_reporting import safe_name
    from scripts.run_torchscript_inference import load_model_version, resolve_precision
except ModuleNotFoundError:
    from inference_reporting import safe_name
    from run_torchscript_inference import load_model_version, resolve_precision

    from monitoring.wandb_monitor import WandbMonitor, WandbMonitorConfig


DEFAULT_BATCH_SIZES = "1,8,32,128,512,1024,2048,4096"
CSV_FIELDS = (
    "batch_size",
    "status",
    "batch_count",
    "repeats",
    "median_runtime_seconds",
    "median_throughput_predictions_per_second",
    "speedup_vs_batch_1",
    "incremental_throughput_gain_percent",
    "median_batch_latency_ms",
    "p95_batch_latency_ms",
    "amortized_latency_ms_per_prediction",
    "peak_cuda_memory_gb",
    "gpu_utilization_mean_percent",
    "gpu_memory_activity_mean_percent",
    "gpu_sample_count",
)


def parse_batch_sizes(value):
    tokens = [
        token for token in re.split(r"[\s,;:]+", str(value or "").strip()) if token
    ]
    if not tokens:
        raise ValueError("At least one inference batch size is required.")

    try:
        batch_sizes = sorted({int(token) for token in tokens})
    except ValueError as error:
        raise ValueError(
            "Inference batch sizes must be positive integers separated by commas."
        ) from error
    if batch_sizes[0] < 1:
        raise ValueError("Inference batch sizes must be greater than zero.")
    if 1 not in batch_sizes:
        raise ValueError(
            "Inference batch sizes must include 1 as the speedup baseline."
        )
    return batch_sizes


def percentile(values, percentile_value):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile_value)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def classify_hardware_bound(gpu_utilization, memory_activity):
    if gpu_utilization is None or memory_activity is None:
        return {
            "classification": "unavailable",
            "confidence": "unavailable",
            "explanation": (
                "Sustained NVIDIA GPU activity samples were not available."
            ),
        }

    gpu_utilization = float(gpu_utilization)
    memory_activity = float(memory_activity)
    if gpu_utilization >= 80 and memory_activity < 70:
        classification = "likely_compute_bound"
        explanation = (
            "SM activity was high while device-memory activity was materially lower."
        )
    elif memory_activity >= 80 and gpu_utilization < 70:
        classification = "likely_memory_bandwidth_bound"
        explanation = (
            "Device-memory activity was high while SM activity was materially lower."
        )
    elif gpu_utilization >= 75 and memory_activity >= 75:
        classification = "mixed_compute_and_memory_pressure"
        explanation = "Both SM and device-memory activity were high."
    elif gpu_utilization >= memory_activity + 20 and gpu_utilization >= 60:
        classification = "likely_compute_bound"
        explanation = "SM activity substantially exceeded device-memory activity."
    elif memory_activity >= gpu_utilization + 20 and memory_activity >= 60:
        classification = "likely_memory_bandwidth_bound"
        explanation = "Device-memory activity substantially exceeded SM activity."
    else:
        classification = "underutilized_or_overhead_bound"
        explanation = (
            "Neither SM nor device-memory activity dominated; launch, transfer, "
            "or CPU overhead may be limiting throughput."
        )

    return {
        "classification": classification,
        "confidence": "heuristic",
        "explanation": explanation,
    }


def analyze_results(results, saturation_threshold=0.95):
    successful = sorted(
        (result for result in results if result["status"] == "ok"),
        key=lambda result: result["batch_size"],
    )
    if not successful:
        hardware_bound = classify_hardware_bound(None, None)
        hardware_bound.update(
            {
                "evaluated_at_batch_size": None,
                "gpu_utilization_mean_percent": None,
                "gpu_memory_activity_mean_percent": None,
                "method": (
                    "Mean nvidia-smi SM and memory-activity samples collected "
                    "during the timed inference passes."
                ),
                "limitation": (
                    "No successful batch size was available for hardware-bound "
                    "analysis."
                ),
            }
        )
        return {
            "peak_throughput_predictions_per_second": None,
            "peak_throughput_batch_size": None,
            "saturation_batch_size": None,
            "speedup_at_saturation": None,
            "baseline_batch_latency_ms": None,
            "saturation_batch_latency_ms": None,
            "batch_latency_increase_at_saturation": None,
            "hardware_bound": hardware_bound,
        }

    baseline = next(
        (result for result in successful if result["batch_size"] == 1),
        successful[0],
    )
    baseline_throughput = baseline["median_throughput_predictions_per_second"]
    peak_result = max(
        successful,
        key=lambda result: result["median_throughput_predictions_per_second"],
    )
    peak_throughput = peak_result["median_throughput_predictions_per_second"]

    previous_throughput = None
    for result in successful:
        throughput = result["median_throughput_predictions_per_second"]
        result["speedup_vs_batch_1"] = (
            throughput / baseline_throughput if baseline_throughput > 0 else 0.0
        )
        result["incremental_throughput_gain_percent"] = (
            None
            if previous_throughput is None or previous_throughput <= 0
            else (throughput / previous_throughput - 1.0) * 100.0
        )
        previous_throughput = throughput

    saturation_result = next(
        (
            result
            for result in successful
            if result["median_throughput_predictions_per_second"]
            >= peak_throughput * float(saturation_threshold)
        ),
        peak_result,
    )
    saturation_region = [
        result
        for result in successful
        if result["batch_size"] >= saturation_result["batch_size"]
    ]

    def weighted_activity(field):
        sampled = [
            result
            for result in saturation_region
            if result.get(field) is not None
            and int(result.get("gpu_sample_count") or 0) > 0
        ]
        sample_count = sum(result["gpu_sample_count"] for result in sampled)
        if sample_count == 0:
            return None
        return (
            sum(result[field] * result["gpu_sample_count"] for result in sampled)
            / sample_count
        )

    gpu_utilization = weighted_activity("gpu_utilization_mean_percent")
    memory_activity = weighted_activity("gpu_memory_activity_mean_percent")
    hardware_bound = classify_hardware_bound(
        gpu_utilization,
        memory_activity,
    )
    hardware_bound.update(
        {
            "evaluated_at_batch_size": saturation_result["batch_size"],
            "evaluated_batch_sizes": [
                result["batch_size"] for result in saturation_region
            ],
            "gpu_utilization_mean_percent": gpu_utilization,
            "gpu_memory_activity_mean_percent": memory_activity,
            "method": (
                "Sample-count-weighted nvidia-smi SM and memory-activity means "
                "across all batch sizes in the saturated throughput region."
            ),
            "limitation": (
                "nvidia-smi memory utilization measures time with active memory "
                "traffic, not achieved DRAM GB/s. Use NVIDIA Nsight Compute "
                "roofline metrics for a definitive compute-versus-bandwidth claim."
            ),
        }
    )
    baseline_latency = baseline["median_batch_latency_ms"]
    saturation_latency = saturation_result["median_batch_latency_ms"]
    return {
        "peak_throughput_predictions_per_second": peak_throughput,
        "peak_throughput_batch_size": peak_result["batch_size"],
        "saturation_batch_size": saturation_result["batch_size"],
        "speedup_at_saturation": saturation_result["speedup_vs_batch_1"],
        "baseline_batch_latency_ms": baseline_latency,
        "saturation_batch_latency_ms": saturation_latency,
        "batch_latency_increase_at_saturation": (
            saturation_latency / baseline_latency if baseline_latency > 0 else None
        ),
        "hardware_bound": hardware_bound,
    }


def theoretical_cuda_memory_bandwidth_gb_s(device):
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    memory_clock_rate = getattr(properties, "memory_clock_rate", None)
    memory_bus_width = getattr(properties, "memory_bus_width", None)
    if not memory_clock_rate or not memory_bus_width:
        return None
    return (
        2.0
        * float(memory_clock_rate)
        * 1000.0
        * (float(memory_bus_width) / 8.0)
        / 1_000_000_000.0
    )


class NvidiaSmiSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms = max(50, int(interval_ms))
        self.samples = []
        self.process = None
        self.reader_thread = None

    def start(self):
        if shutil.which("nvidia-smi") is None:
            return
        visible_device = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
            "memory.total,power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        if visible_device:
            command.extend(["--id", visible_device])

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self.process = None
            return

        self.reader_thread = threading.Thread(
            target=self._read_samples,
            daemon=True,
        )
        self.reader_thread.start()

    def _read_samples(self):
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            values = [value.strip() for value in line.split(",")]
            if len(values) != 5:
                continue
            try:
                self.samples.append(
                    {
                        "gpu_utilization_percent": float(values[0]),
                        "memory_activity_percent": float(values[1]),
                        "memory_used_mb": float(values[2]),
                        "memory_total_mb": float(values[3]),
                        "power_watts": float(values[4]),
                    }
                )
            except ValueError:
                continue

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2)

    def summary(self):
        samples = self.samples[1:] if len(self.samples) > 2 else self.samples
        if not samples:
            return {
                "gpu_sample_count": 0,
                "gpu_utilization_mean_percent": None,
                "gpu_utilization_peak_percent": None,
                "gpu_memory_activity_mean_percent": None,
                "gpu_memory_activity_peak_percent": None,
                "gpu_memory_used_peak_mb": None,
                "gpu_power_mean_watts": None,
            }

        def values(name):
            return [sample[name] for sample in samples]

        return {
            "gpu_sample_count": len(samples),
            "gpu_utilization_mean_percent": statistics.fmean(
                values("gpu_utilization_percent")
            ),
            "gpu_utilization_peak_percent": max(values("gpu_utilization_percent")),
            "gpu_memory_activity_mean_percent": statistics.fmean(
                values("memory_activity_percent")
            ),
            "gpu_memory_activity_peak_percent": max(values("memory_activity_percent")),
            "gpu_memory_used_peak_mb": max(values("memory_used_mb")),
            "gpu_power_mean_watts": statistics.fmean(values("power_watts")),
        }


def _is_cuda_oom(error):
    return isinstance(error, torch.OutOfMemoryError) or (
        "out of memory" in str(error).lower()
    )


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_candidate(
    model,
    windows,
    batch_size,
    repeats,
    warmup_iterations,
    device,
    dtype,
    sampler=None,
):
    window_count = len(windows)
    warmup_size = min(batch_size, window_count)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            warmup = windows[:warmup_size].contiguous().to(device, dtype=dtype)
            model(warmup)
        _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sampler = sampler or NvidiaSmiSampler()
    repeat_runtimes = []
    repeat_throughputs = []
    full_batch_latencies = []
    partial_batch_count = 0
    sampler.start()
    try:
        with torch.inference_mode():
            for _ in range(repeats):
                repeat_started = time.perf_counter()
                for batch_start in range(0, window_count, batch_size):
                    operation_started = time.perf_counter()
                    batch = (
                        windows[batch_start : batch_start + batch_size]
                        .contiguous()
                        .to(device, dtype=dtype)
                    )
                    output = model(batch)
                    output.flatten().float().cpu()
                    _synchronize(device)
                    operation_latency_ms = (
                        time.perf_counter() - operation_started
                    ) * 1000.0
                    if len(batch) == batch_size:
                        full_batch_latencies.append(operation_latency_ms)
                    else:
                        partial_batch_count += 1

                runtime_seconds = time.perf_counter() - repeat_started
                repeat_runtimes.append(runtime_seconds)
                repeat_throughputs.append(window_count / runtime_seconds)
    finally:
        sampler.stop()

    representative_latencies = full_batch_latencies
    if not representative_latencies:
        representative_latencies = [runtime * 1000.0 for runtime in repeat_runtimes]
    median_throughput = statistics.median(repeat_throughputs)
    peak_cuda_memory_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    return {
        "batch_size": int(batch_size),
        "status": "ok",
        "batch_count": math.ceil(window_count / batch_size),
        "repeats": int(repeats),
        "prediction_count_per_repeat": window_count,
        "repeat_runtime_seconds": repeat_runtimes,
        "repeat_throughput_predictions_per_second": repeat_throughputs,
        "median_runtime_seconds": statistics.median(repeat_runtimes),
        "median_throughput_predictions_per_second": median_throughput,
        "median_batch_latency_ms": statistics.median(representative_latencies),
        "p95_batch_latency_ms": percentile(representative_latencies, 0.95),
        "amortized_latency_ms_per_prediction": 1000.0 / median_throughput,
        "latency_samples_full_batches": len(full_batch_latencies),
        "partial_batch_count": partial_batch_count,
        "peak_cuda_memory_gb": peak_cuda_memory_gb,
        **sampler.summary(),
    }


def run_benchmark(
    model_path,
    input_path,
    precision,
    sequence_length,
    batch_sizes,
    repeats,
    warmup_iterations,
    config_path,
    saturation_threshold=0.95,
    sampler_factory=NvidiaSmiSampler,
):
    if sequence_length < 1 or repeats < 1 or warmup_iterations < 0:
        raise ValueError(
            "Sequence length and repeats must be positive; warmup may be zero."
        )
    batch_sizes = parse_batch_sizes(batch_sizes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = resolve_precision(precision, device)
    dataframe = pd.read_csv(input_path)
    features = torch.from_numpy(dataframe.to_numpy(dtype="float32", copy=True))
    if len(features) < sequence_length:
        raise ValueError(
            f"Input has {len(features)} rows but sequence length is {sequence_length}."
        )

    windows = features.unfold(0, sequence_length, 1).permute(0, 2, 1)
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    model.to(device=device, dtype=dtype)
    results = []
    for batch_size in batch_sizes:
        print(f"Benchmarking inference batch size {batch_size}...")
        try:
            result = benchmark_candidate(
                model=model,
                windows=windows,
                batch_size=batch_size,
                repeats=repeats,
                warmup_iterations=warmup_iterations,
                device=device,
                dtype=dtype,
                sampler=sampler_factory(),
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            result = {
                "batch_size": batch_size,
                "status": "out_of_memory",
                "error": str(error),
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
            results.append(result)
            print(f"Batch size {batch_size}: out of memory")
            break
        results.append(result)
        print(
            f"Batch size {batch_size}: "
            f"{result['median_throughput_predictions_per_second']:.2f} "
            "predictions/s, "
            f"{result['median_batch_latency_ms']:.3f} ms median latency"
        )

    analysis = analyze_results(results, saturation_threshold)
    return {
        "benchmark": "inference_batch_size",
        "latency_definition": (
            "End-to-end time to complete one batch inference operation, including "
            "host-to-device input transfer and device-to-host output transfer, "
            "but excluding model/data loading and request queueing."
        ),
        "throughput_definition": "Completed predictions per second.",
        "bandwidth_definition": "Maximum rate at which data can be transferred.",
        "model_version": load_model_version(config_path),
        "precision": precision.upper(),
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"),
        "sequence_length": int(sequence_length),
        "feature_count": int(features.shape[1]),
        "prediction_count_per_repeat": len(windows),
        "repeats": int(repeats),
        "warmup_iterations": int(warmup_iterations),
        "saturation_threshold_fraction": float(saturation_threshold),
        "gpu_theoretical_memory_bandwidth_gb_s": (
            theoretical_cuda_memory_bandwidth_gb_s(device)
        ),
        **analysis,
        "results": results,
    }


def write_json(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def write_csv(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in report["results"]:
            writer.writerow({field: result.get(field) for field in CSV_FIELDS})
    return output_path


def _format_number(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_markdown(report):
    hardware_bound = report["hardware_bound"]
    lines = [
        "# Inference Batch Benchmark",
        "",
        f"- Model version: `{report['model_version']}`",
        f"- Device: `{report['device']}`",
        f"- Precision: `{report['precision']}`",
        f"- Predictions per repeat: {report['prediction_count_per_repeat']}",
        f"- Repeats: {report['repeats']}",
        (
            "- Saturation batch size: "
            f"`{report['saturation_batch_size']}` "
            f"({report['saturation_threshold_fraction']:.0%} of peak throughput)"
        ),
        (
            "- Peak throughput: "
            f"{_format_number(report['peak_throughput_predictions_per_second'], 2)} "
            "predictions/s"
        ),
        (
            "- Speedup at saturation: "
            f"{_format_number(report['speedup_at_saturation'], 2)}x"
        ),
        (
            "- Median batch latency at saturation: "
            f"{_format_number(report['saturation_batch_latency_ms'])} ms "
            f"({_format_number(report['batch_latency_increase_at_saturation'], 2)}x "
            "the batch-1 latency)"
        ),
        (
            "- Theoretical maximum GPU memory bandwidth: "
            f"{_format_number(report['gpu_theoretical_memory_bandwidth_gb_s'], 2)} "
            "GB/s"
        ),
        (f"- Hardware-bound diagnosis: `{hardware_bound['classification']}`"),
        f"- Diagnosis: {hardware_bound['explanation']}",
        "",
        "Latency is the end-to-end time to complete one batch inference operation. "
        "It excludes model loading and request queueing.",
        "",
        "| Batch | Status | Runtime (s) | Throughput (pred/s) | Speedup | "
        "Gain vs previous | Median latency (ms) | P95 latency (ms) | "
        "Peak CUDA memory (GB) | SM activity (%) | Memory activity (%) |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        lines.append(
            "| {batch_size} | {status} | {runtime} | {throughput} | "
            "{speedup} | {gain} | {latency} | {p95} | {memory} | "
            "{gpu_util} | {memory_activity} |".format(
                batch_size=result["batch_size"],
                status=result["status"],
                runtime=_format_number(result.get("median_runtime_seconds")),
                throughput=_format_number(
                    result.get("median_throughput_predictions_per_second"),
                    2,
                ),
                speedup=(
                    "n/a"
                    if result.get("speedup_vs_batch_1") is None
                    else f"{result['speedup_vs_batch_1']:.2f}x"
                ),
                gain=(
                    "n/a"
                    if result.get("incremental_throughput_gain_percent") is None
                    else f"{result['incremental_throughput_gain_percent']:.1f}%"
                ),
                latency=_format_number(result.get("median_batch_latency_ms")),
                p95=_format_number(result.get("p95_batch_latency_ms")),
                memory=_format_number(result.get("peak_cuda_memory_gb")),
                gpu_util=_format_number(
                    result.get("gpu_utilization_mean_percent"),
                    1,
                ),
                memory_activity=_format_number(
                    result.get("gpu_memory_activity_mean_percent"),
                    1,
                ),
            )
        )
    lines.extend(
        [
            "",
            f"Hardware diagnosis method: {hardware_bound['method']}",
            "",
            f"Limitation: {hardware_bound['limitation']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(path, report):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(report), encoding="utf-8")
    return output_path


def log_report_to_wandb(report, monitoring_config_path, artifact_paths):
    monitor_config = WandbMonitorConfig.from_yaml(monitoring_config_path)
    if monitor_config.mode == "online" and not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; skipping batch benchmark W&B logging.")
        return

    base_run_name = os.getenv("WANDB_RUN_NAME") or (
        f"inference-{report['model_version']}"
    )
    monitor_config.run_name = f"{safe_name(base_run_name)}-batch-benchmark"
    monitor_config.group = safe_name(report["model_version"])
    monitor_config.job_type = "inference-benchmark"
    monitor_config.tags = list(
        dict.fromkeys(
            [
                *(monitor_config.tags or []),
                "inference",
                "batch-benchmark",
                report["precision"].lower(),
            ]
        )
    )
    run_config = {
        key: value
        for key, value in report.items()
        if key not in {"results", "hardware_bound"}
    }
    monitor = WandbMonitor(monitor_config).start(training_config=run_config)
    try:
        for step, result in enumerate(report["results"]):
            if result["status"] != "ok":
                continue
            metrics = {
                f"benchmark/{key}": value
                for key, value in result.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            monitor.log_metrics(metrics, step=step)
        monitor.update_summary(
            {
                "benchmark/saturation_batch_size": report["saturation_batch_size"],
                "benchmark/peak_throughput_predictions_per_second": report[
                    "peak_throughput_predictions_per_second"
                ],
                "benchmark/speedup_at_saturation": report["speedup_at_saturation"],
                "benchmark/saturation_batch_latency_ms": report[
                    "saturation_batch_latency_ms"
                ],
                "benchmark/batch_latency_increase_at_saturation": report[
                    "batch_latency_increase_at_saturation"
                ],
                "benchmark/hardware_bound": report["hardware_bound"]["classification"],
                "benchmark/hardware_bound_confidence": report["hardware_bound"][
                    "confidence"
                ],
            }
        )
        monitor.log_files_artifact(
            files=artifact_paths,
            artifact_name=(
                f"{safe_name(report['model_version'])}-"
                f"{report['precision'].lower()}-batch-benchmark"
            ),
            artifact_type="inference-benchmark",
            metadata={
                "saturation_batch_size": report["saturation_batch_size"],
                "hardware_bound": report["hardware_bound"]["classification"],
            },
            aliases=["latest", report["precision"].lower()],
        )
    finally:
        monitor.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TorchScript inference across multiple batch sizes."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--saturation-threshold", type=float, default=0.95)
    parser.add_argument("--config", default="reports/runtime_train_config.yaml")
    parser.add_argument(
        "--monitoring-config",
        default="reports/runtime_monitoring_config.yaml",
    )
    parser.add_argument(
        "--json-output",
        default="reports/inference_batch_benchmark.json",
    )
    parser.add_argument(
        "--csv-output",
        default="reports/inference_batch_benchmark.csv",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/inference_batch_benchmark.md",
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    if not 0 < args.saturation_threshold <= 1:
        raise ValueError("Saturation threshold must be greater than 0 and at most 1.")

    report = run_benchmark(
        model_path=args.model,
        input_path=args.input,
        precision=args.precision,
        sequence_length=args.sequence_length,
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        warmup_iterations=args.warmup_iterations,
        config_path=args.config,
        saturation_threshold=args.saturation_threshold,
    )
    json_path = write_json(args.json_output, report)
    csv_path = write_csv(args.csv_output, report)
    markdown_path = write_markdown(args.markdown_output, report)
    print(build_markdown(report))
    if args.wandb:
        log_report_to_wandb(
            report,
            args.monitoring_config,
            [json_path, csv_path, markdown_path],
        )


if __name__ == "__main__":
    main()
