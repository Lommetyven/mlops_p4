import pytest
import torch

from scripts.benchmark_inference_batches import (
    analyze_results,
    benchmark_candidate,
    build_markdown,
    classify_hardware_bound,
    parse_batch_sizes,
    run_benchmark,
    write_csv,
    write_json,
    write_markdown,
)


class LastStepModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, -1, :1]


class EmptySampler:
    def start(self):
        return None

    def stop(self):
        return None

    def summary(self):
        return {
            "gpu_sample_count": 0,
            "gpu_utilization_mean_percent": None,
            "gpu_utilization_peak_percent": None,
            "gpu_memory_activity_mean_percent": None,
            "gpu_memory_activity_peak_percent": None,
            "gpu_memory_used_peak_mb": None,
            "gpu_power_mean_watts": None,
        }


def test_parse_batch_sizes_sorts_deduplicates_and_requires_baseline():
    assert parse_batch_sizes("32, 1 8:8") == [1, 8, 32]

    with pytest.raises(ValueError, match="include 1"):
        parse_batch_sizes("8,32")


def test_analyze_results_calculates_speedup_and_saturation_knee():
    results = [
        {
            "batch_size": 1,
            "status": "ok",
            "median_throughput_predictions_per_second": 100.0,
            "median_batch_latency_ms": 1.0,
            "gpu_utilization_mean_percent": 30.0,
            "gpu_memory_activity_mean_percent": 20.0,
            "gpu_sample_count": 2,
        },
        {
            "batch_size": 8,
            "status": "ok",
            "median_throughput_predictions_per_second": 380.0,
            "median_batch_latency_ms": 2.0,
            "gpu_utilization_mean_percent": 90.0,
            "gpu_memory_activity_mean_percent": 50.0,
            "gpu_sample_count": 3,
        },
        {
            "batch_size": 32,
            "status": "ok",
            "median_throughput_predictions_per_second": 400.0,
            "median_batch_latency_ms": 6.0,
            "gpu_utilization_mean_percent": 92.0,
            "gpu_memory_activity_mean_percent": 55.0,
            "gpu_sample_count": 1,
        },
    ]

    analysis = analyze_results(results, saturation_threshold=0.95)

    assert analysis["peak_throughput_batch_size"] == 32
    assert analysis["saturation_batch_size"] == 8
    assert analysis["speedup_at_saturation"] == pytest.approx(3.8)
    assert analysis["batch_latency_increase_at_saturation"] == pytest.approx(2.0)
    assert analysis["hardware_bound"]["evaluated_batch_sizes"] == [8, 32]
    assert results[1]["speedup_vs_batch_1"] == pytest.approx(3.8)
    assert results[2]["incremental_throughput_gain_percent"] == pytest.approx(
        400 / 380 * 100 - 100
    )
    assert analysis["hardware_bound"]["classification"] == "likely_compute_bound"


def test_hardware_bound_can_identify_memory_activity_pressure():
    diagnosis = classify_hardware_bound(
        gpu_utilization=55,
        memory_activity=90,
    )

    assert diagnosis["classification"] == "likely_memory_bandwidth_bound"
    assert diagnosis["confidence"] == "heuristic"


def test_benchmark_candidate_records_batch_latency_and_throughput():
    windows = torch.arange(18, dtype=torch.float32).reshape(9, 2, 1)

    result = benchmark_candidate(
        model=LastStepModel(),
        windows=windows,
        batch_size=4,
        repeats=2,
        warmup_iterations=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sampler=EmptySampler(),
    )

    assert result["status"] == "ok"
    assert result["batch_count"] == 3
    assert result["prediction_count_per_repeat"] == 9
    assert result["latency_samples_full_batches"] == 4
    assert result["partial_batch_count"] == 2
    assert result["median_batch_latency_ms"] > 0
    assert result["median_throughput_predictions_per_second"] > 0


def test_run_benchmark_builds_complete_report_and_artifacts(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.csv"
    config_path = tmp_path / "config.yaml"
    example = torch.zeros(1, 2, 1)
    torch.jit.trace(LastStepModel(), example).save(str(model_path))
    input_path.write_text("feature\n1\n2\n3\n4\n5\n6\n", encoding="utf-8")
    config_path.write_text(
        "experiment:\n  model_version: batch-test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = run_benchmark(
        model_path=model_path,
        input_path=input_path,
        precision="FP32",
        sequence_length=2,
        batch_sizes="1,4",
        repeats=1,
        warmup_iterations=1,
        config_path=config_path,
        sampler_factory=EmptySampler,
    )

    assert report["prediction_count_per_repeat"] == 5
    assert report["saturation_batch_size"] in {1, 4}
    assert len(report["results"]) == 2
    assert "Latency is the end-to-end time" in build_markdown(report)
    assert write_json(tmp_path / "report.json", report).is_file()
    assert write_csv(tmp_path / "report.csv", report).is_file()
    assert write_markdown(tmp_path / "report.md", report).is_file()
