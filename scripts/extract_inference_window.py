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
    window_size = int(rows or training_config["sequence_length"])

    dataframe = pd.read_csv(processed_path)
    missing_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Processed data is missing required feature columns: "
            f"{', '.join(missing_columns)}"
        )

    if len(dataframe) < window_size:
        raise ValueError(
            "Processed dataset has only "
            f"{len(dataframe)} rows but needs at least {window_size}."
        )

    window = dataframe.iloc[:window_size, :].loc[:, feature_columns]

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
    print(f"Extracted inference window to {output_path}")


if __name__ == "__main__":
    main()
