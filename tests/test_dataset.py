import pandas as pd
import pytest

from main import DEFAULT_FEATURE_COLUMNS, build_dataloaders, load_train_config
from train.dataset import (
    build_sequence_dataset,
    load_processed_dataframe,
    split_dataset,
    split_train_val_test_dataset,
    validate_processed_dataframe,
)

pytest.importorskip("torch")


def _processed_dataframe(row_count=8):
    data = {
        column: [float(row_index) for row_index in range(row_count)]
        for column in DEFAULT_FEATURE_COLUMNS
    }
    data["target_next_hour"] = [float(row_index + 1) for row_index in range(row_count)]
    return pd.DataFrame(data)


def test_load_processed_dataframe_reads_csv(tmp_path):
    processed_path = tmp_path / "processed.csv"
    dataframe = _processed_dataframe()
    dataframe.to_csv(processed_path, index=False)

    loaded = load_processed_dataframe(processed_path)

    assert loaded.shape == dataframe.shape


def test_build_sequence_dataset_returns_gru_windows():
    dataframe = _processed_dataframe(row_count=8)

    dataset = build_sequence_dataset(
        dataframe=dataframe,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_column="target_next_hour",
        sequence_length=3,
    )
    features, target = dataset[0]

    assert len(dataset) == 6
    assert features.shape == (3, 16)
    assert target.shape == (1,)
    assert target.item() == 3.0


def test_validate_processed_dataframe_rejects_missing_columns():
    dataframe = _processed_dataframe().drop(columns=[DEFAULT_FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_processed_dataframe(
            dataframe,
            DEFAULT_FEATURE_COLUMNS,
            "target_next_hour",
        )


def test_split_dataset_can_create_train_and_validation_sets():
    dataset = build_sequence_dataset(
        dataframe=_processed_dataframe(row_count=20),
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_column="target_next_hour",
        sequence_length=4,
    )

    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        validation_split=0.25,
        seed=42,
    )

    assert len(train_dataset) == 13
    assert len(val_dataset) == 4


def test_split_train_val_test_dataset_can_create_three_sets():
    dataset = build_sequence_dataset(
        dataframe=_processed_dataframe(row_count=30),
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        target_column="target_next_hour",
        sequence_length=4,
    )

    train_dataset, val_dataset, test_dataset = split_train_val_test_dataset(
        dataset=dataset,
        validation_split=0.2,
        test_split=0.1,
        seed=42,
    )

    assert len(train_dataset) == 20
    assert len(val_dataset) == 5
    assert len(test_dataset) == 2


def test_build_dataloaders_from_processed_csv(tmp_path):
    processed_path = tmp_path / "processed.csv"
    _processed_dataframe(row_count=12).to_csv(processed_path, index=False)
    config = load_train_config()
    config["data"]["processed_path"] = str(processed_path)
    config["training"]["sequence_length"] = 3
    config["training"]["batch_size"] = 2
    config["training"]["validation_split"] = 0.2
    config["training"]["test_split"] = 0.1
    config["training"]["num_workers"] = 0

    train_loader, val_loader, test_loader, sequence_count, split_sizes = (
        build_dataloaders(config)
    )
    features, targets = next(iter(train_loader))

    assert sequence_count == 10
    assert features.shape == (2, 3, 16)
    assert targets.shape == (2, 1)
    assert val_loader is not None
    assert test_loader is not None
    assert split_sizes == {"train": 7, "validation": 2, "test": 1}
