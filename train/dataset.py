from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


class SequenceDataset(Dataset):
    def __init__(self, features, targets, sequence_length):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        self.features = torch.as_tensor(
            np.array(features, dtype=np.float32, copy=True),
            dtype=torch.float32,
        )
        self.targets = torch.as_tensor(
            np.array(targets, dtype=np.float32, copy=True),
            dtype=torch.float32,
        ).view(-1, 1)
        self.sequence_length = int(sequence_length)

        if len(self.features) != len(self.targets):
            raise ValueError("features and targets must have the same number of rows.")

        if len(self.features) < self.sequence_length:
            raise ValueError(
                "Not enough rows to build one sequence. "
                f"Got {len(self.features)} rows and "
                f"sequence_length={self.sequence_length}."
            )

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, index):
        end_index = index + self.sequence_length
        return (
            self.features[index:end_index],
            self.targets[end_index - 1],
        )


def load_processed_dataframe(processed_path):
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed training data was not found at {path}. "
            "Run data_insight/data_handling.ipynb first to create it."
        )

    return pd.read_csv(path)


def validate_processed_dataframe(dataframe, feature_columns, target_column):
    required_columns = list(feature_columns) + [target_column]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Processed data is missing required columns: {', '.join(missing_columns)}"
        )

    if dataframe[required_columns].isna().any().any():
        raise ValueError(
            "Processed data contains NaN values in feature/target columns."
        )


def build_sequence_dataset(
    dataframe,
    feature_columns,
    target_column,
    sequence_length,
):
    validate_processed_dataframe(dataframe, feature_columns, target_column)

    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    targets = dataframe[target_column].to_numpy(dtype=np.float32)
    return SequenceDataset(features, targets, sequence_length)


def split_dataset(dataset, validation_split, seed):
    train_dataset, val_dataset, _ = split_train_val_test_dataset(
        dataset=dataset,
        validation_split=validation_split,
        test_split=0.0,
        seed=seed,
    )
    return train_dataset, val_dataset


def split_train_val_test_dataset(dataset, validation_split, test_split, seed):
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be between 0 and 1.")
    if validation_split + test_split >= 1:
        raise ValueError("validation_split + test_split must be less than 1.")

    validation_size = int(len(dataset) * validation_split)
    test_size = int(len(dataset) * test_split)
    training_size = len(dataset) - validation_size - test_size

    if training_size < 1:
        raise ValueError("Training split is empty. Lower validation/test splits.")

    lengths = [training_size, validation_size, test_size]
    split_names = ["train", "validation", "test"]

    generator = torch.Generator().manual_seed(int(seed))
    splits = random_split(
        dataset,
        lengths,
        generator=generator,
    )
    return tuple(
        split if len(split) > 0 else None
        for split_name, split in zip(split_names, splits, strict=True)
    )
