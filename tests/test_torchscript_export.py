import pytest
import torch

from main import export_torchscript_model
from train.gru_model import GruModel

pytest.importorskip("torch")


def test_export_torchscript_model_creates_loadable_model(tmp_path):
    model = GruModel()
    output_path = tmp_path / "gru_model_torchscript.pt"

    exported_path = export_torchscript_model(
        model=model,
        output_path=output_path,
        sequence_length=2,
        device=torch.device("cpu"),
    )

    loaded = torch.jit.load(str(exported_path), map_location="cpu")
    with torch.no_grad():
        prediction = loaded(torch.zeros(1, 2, model.input_size))

    assert exported_path == output_path
    assert prediction.shape == (1, 1)


def test_export_torchscript_model_ignores_runtime_hooks(tmp_path):
    model = GruModel()
    model.register_forward_hook(lambda *_args, **_kwargs: None)
    output_path = tmp_path / "gru_model_torchscript.pt"

    exported_path = export_torchscript_model(
        model=model,
        output_path=output_path,
        sequence_length=2,
        device=torch.device("cpu"),
    )

    loaded = torch.jit.load(str(exported_path), map_location="cpu")
    with torch.no_grad():
        prediction = loaded(torch.zeros(1, 2, model.input_size))

    assert exported_path == output_path
    assert prediction.shape == (1, 1)
