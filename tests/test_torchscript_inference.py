import pandas as pd
import pytest
import torch

from scripts.run_torchscript_inference import (
    resolve_precision,
    run_torchscript_inference,
)


class LastStepModel(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, -1, :1]


def test_torchscript_inference_processes_every_sliding_window(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.csv"
    predictions_path = tmp_path / "predictions.txt"
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "train_config.yaml"
    monitoring_config_path = tmp_path / "monitoring_config.yaml"

    example = torch.zeros(1, 2, 16)
    torch.jit.trace(LastStepModel(), example).save(str(model_path))
    dataframe = pd.DataFrame(
        [[float(row * 16 + column) for column in range(16)] for row in range(4)]
    )
    dataframe.to_csv(input_path, index=False)
    config_path.write_text(
        "experiment:\n  model_version: inference-test\n",
        encoding="utf-8",
    )
    monitoring_config_path.write_text(
        "monitoring:\n  mode: disabled\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    metrics = run_torchscript_inference(
        model_path=model_path,
        input_path=input_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        precision="FP32",
        sequence_length=2,
        batch_size=2,
        config_path=config_path,
        monitoring_config_path=monitoring_config_path,
        enable_wandb=False,
    )

    predictions = [
        float(value)
        for value in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert predictions == [16.0, 32.0, 48.0]
    assert metrics["inference/prediction_count"] == 3
    assert metrics["inference/expected_windows"] == 3
    assert metrics["inference/status"] == "success"
    assert metrics["inference/device"] == "CPU"


def test_fp16_inference_requires_cuda():
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        resolve_precision("FP16", torch.device("cpu"))
