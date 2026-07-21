from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def extract_inference_window(config_path, output_path, rows=None):
    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]

    processed_path = Path(data_config["processed_path"])
    feature_columns = list(data_config["feature_columns"])
    sequence_length = int(training_config["sequence_length"])

    dataframe = pd.read_csv(processed_path)
    missing_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed data is missing required feature columns: "
            f"{', '.join(missing_columns)}"
        )

    row_count = len(dataframe) if rows is None else int(rows)
    if row_count < sequence_length:
        raise ValueError(
            "Processed dataset has only "
            f"{row_count} selected rows but needs at least {sequence_length}."
        )
    if row_count > len(dataframe):
        raise ValueError(
            f"Requested {row_count} rows but the processed dataset has "
            f"only {len(dataframe)}."
        )

    window = dataframe.iloc[:row_count, :].loc[:, feature_columns]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(output_path, index=False)

    return output_path


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="reports/runtime_train_config.yaml")
    parser.add_argument("--output", default="reports/inference_window.csv")
    parser.add_argument("--rows", type=int, default=None)
    args = parser.parse_args()

    output_path = extract_inference_window(
        config_path=args.config,
        output_path=args.output,
        rows=args.rows,
    )
    print(f"Extracted inference dataset to {output_path}")


if __name__ == "__main__":
    main()
