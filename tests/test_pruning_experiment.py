import pytest
import torch

from scripts.pruning_experiment import (
    add_baseline_deltas,
    apply_global_unstructured_pruning,
    build_markdown,
    fine_tune_pruned_model,
    measure_prunable_sparsity,
    parse_pruning_levels,
    write_csv,
    write_json,
    write_markdown,
    write_plot,
)


class TinyPrunableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_gru_weights = torch.nn.Parameter(
            torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)
        )
        self.hidden_to_hidden_weights = torch.nn.Parameter(
            torch.arange(5, 11, dtype=torch.float32).reshape(2, 3)
        )

    def forward(self, inputs):
        input_term = inputs @ self.input_to_gru_weights.transpose(0, 1)
        hidden_term = self.hidden_to_hidden_weights.sum(dim=1)
        return (input_term + hidden_term).mean(dim=1, keepdim=True)


def build_report():
    results = []
    for level, immediate_rmse, fine_tuned_rmse in (
        (0.0, 1.0, 0.9),
        (50.0, 1.5, 1.1),
    ):
        sparsity = {
            "total": 10,
            "nonzero": int(10 * (1 - level / 100)),
            "zero": int(10 * level / 100),
            "sparsity_percent": level,
            "dense_equivalent_compression_ratio": (
                1 / (1 - level / 100) if level < 100 else None
            ),
            "by_parameter": {},
        }
        results.append(
            {
                "pruning_percentage": level,
                "sparsity_before_fine_tuning": sparsity,
                "sparsity_after_fine_tuning": sparsity,
                "immediate": {
                    "loss": immediate_rmse**2,
                    "rmse": immediate_rmse,
                    "mae": immediate_rmse / 2,
                    "r2": 0.8 - level / 500,
                },
                "fine_tuned": {
                    "loss": fine_tuned_rmse**2,
                    "rmse": fine_tuned_rmse,
                    "mae": fine_tuned_rmse / 2,
                    "r2": 0.85 - level / 500,
                },
                "fine_tuning": {
                    "best_epoch": 2,
                    "best_validation_loss": 0.5,
                    "history": [],
                },
            }
        )
    add_baseline_deltas(results)
    return {
        "schema_version": 1,
        "model_version": "pruning-test",
        "checkpoint": "models/gru_model.pt",
        "task": "regression",
        "device": "CPU",
        "precision": "float32",
        "pruning_method": "global_unstructured_l1_magnitude",
        "pruning_structure": "unstructured_individual_weights",
        "pruned_parameters": [
            "input_to_gru_weights",
            "hidden_to_hidden_weights",
        ],
        "biases_pruned": False,
        "pruning_levels_percent": [0.0, 50.0],
        "fine_tune_epochs": 2,
        "fine_tune_learning_rate": 0.0001,
        "random_seed": 42,
        "split_sizes": {"train": 7, "validation": 2, "test": 1},
        "test_dataset_reused_for_every_evaluation": True,
        "validation_selection_metric": "loss",
        "results": results,
        "limitations": [
            "Unstructured sparsity does not make dense execution sparse.",
        ],
    }


def test_parse_pruning_levels_sorts_deduplicates_and_requires_baseline():
    assert parse_pruning_levels("50, 0 25:25") == [0.0, 25.0, 50.0]

    with pytest.raises(ValueError, match="include 0"):
        parse_pruning_levels("10,25")
    with pytest.raises(ValueError, match="less than 100"):
        parse_pruning_levels("0,100")
    with pytest.raises(ValueError, match="finite"):
        parse_pruning_levels("0,nan")


def test_global_pruning_uses_individual_weights_and_preserves_masks():
    model = TinyPrunableModel()
    sparsity = apply_global_unstructured_pruning(model, 50)

    assert sparsity["total"] == 10
    assert sparsity["zero"] == 5
    assert sparsity["sparsity_percent"] == pytest.approx(50)
    assert hasattr(model, "input_to_gru_weights_mask")
    assert hasattr(model, "hidden_to_hidden_weights_mask")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    model(torch.ones(1, 2))

    assert measure_prunable_sparsity(model)["zero"] == 5


def test_fine_tuning_selects_validation_state_without_regrowing_weights():
    model = TinyPrunableModel()
    apply_global_unstructured_pruning(model, 50)
    features = torch.tensor([[1.0, 2.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([[1.0], [1.0], [0.5], [0.5]])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, targets),
        batch_size=2,
    )

    result = fine_tune_pruned_model(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        criterion=torch.nn.MSELoss(),
        training_config={
            "optimizer": "adam",
            "weight_decay": 0.0,
            "precision": "float32",
            "amp_enabled": False,
            "gradient_clip_norm": 1.0,
        },
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=0.001,
    )

    assert result["best_epoch"] == 1
    assert measure_prunable_sparsity(model)["zero"] == 5


def test_pruning_report_writes_table_data_and_plot(tmp_path):
    report = build_report()
    markdown = build_markdown(report)

    assert "global unstructured L1-magnitude pruning" in markdown
    assert "Immediate RMSE" in markdown
    assert "50%" in markdown
    assert write_json(tmp_path / "pruning.json", report).is_file()
    assert write_csv(tmp_path / "pruning.csv", report).is_file()
    assert write_markdown(tmp_path / "pruning.md", report).is_file()
    plot_path = write_plot(tmp_path / "pruning.png", report)
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
