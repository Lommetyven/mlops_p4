import pytest
import torch

from train.gru_model import GruModel

pytest.importorskip("torch")


def test_gru_parameter_breakdown_matches_required_budget():
    model = GruModel()

    assert model.parameter_breakdown() == {
        "input_to_gru_weights": 38_400,
        "hidden_to_hidden_weights": 1_920_000,
        "biases": 2_400,
        "total": 1_960_800,
    }


def test_gru_forward_returns_one_prediction_per_sequence():
    model = GruModel()
    batch = torch.zeros(2, 3, model.input_size)

    with torch.no_grad():
        output = model(batch)

    assert output.shape == (2, 1)


def test_gru_rejects_wrong_architecture():
    with pytest.raises(ValueError, match="fixed to input_size=16"):
        GruModel(input_size=8)


def test_gru_rejects_wrong_input_feature_count():
    model = GruModel()
    batch = torch.zeros(2, 3, model.input_size - 1)

    with pytest.raises(ValueError, match="Expected input_size=16"):
        model(batch)
