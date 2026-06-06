from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FEATURE_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
TARGET_COLUMN = "target_next_hour"


def load_data_config(config_path):
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    project_root = path.parent.parent if path.parent.name == "configs" else Path(".")
    return config, project_root


def resolve_project_path(project_root, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def apply_missing_value_strategy(dataframe, strategy):
    if strategy == "drop":
        return dataframe.dropna()
    if strategy == "ffill":
        return dataframe.ffill()
    if strategy == "bfill":
        return dataframe.bfill()
    if strategy in (None, "none", ""):
        return dataframe

    raise ValueError("Invalid fill_missing option in config")


def process_household_power_data(dataframe, fill_missing, time_interval):
    dataframe = apply_missing_value_strategy(
        dataframe,
        None if fill_missing is None else str(fill_missing).lower(),
    )
    dataframe = dataframe.dropna()

    dataframe = dataframe.copy()
    dataframe["datetime"] = pd.to_datetime(
        dataframe["Date"] + " " + dataframe["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["datetime"]).set_index("datetime")

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    aggregation = {
        "Global_active_power": "mean",
        "Global_reactive_power": "mean",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
    }
    processed = dataframe.resample(time_interval).agg(aggregation)

    processed["hour"] = processed.index.hour
    processed["day_of_week"] = processed.index.dayofweek
    processed["month"] = processed.index.month
    processed["hour_sin"] = np.sin(2 * np.pi * processed["hour"] / 24)
    processed["hour_cos"] = np.cos(2 * np.pi * processed["hour"] / 24)
    processed["dow_sin"] = np.sin(2 * np.pi * processed["day_of_week"] / 7)
    processed["dow_cos"] = np.cos(2 * np.pi * processed["day_of_week"] / 7)
    processed["month_sin"] = np.sin(2 * np.pi * processed["month"] / 12)
    processed["month_cos"] = np.cos(2 * np.pi * processed["month"] / 12)

    processed[TARGET_COLUMN] = processed["Global_active_power"].shift(-1)
    processed = processed.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return processed[FEATURE_COLUMNS + [TARGET_COLUMN]]


def process_from_config(config_path="configs/data_config.yaml"):
    config, project_root = load_data_config(config_path)
    data_config = config["data"]
    input_path = resolve_project_path(project_root, data_config["input_path"])
    output_path = resolve_project_path(
        project_root,
        data_config.get(
            "processed_output_path", "data/processed/household_power_gru.csv"
        ),
    )

    dataframe = pd.read_csv(
        input_path,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    processed = process_household_power_data(
        dataframe=dataframe,
        fill_missing=data_config.get("fill_missing"),
        time_interval=data_config["time_interval"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=True, index_label="datetime")
    return output_path


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/data_config.yaml")
    args = parser.parse_args()

    output_path = process_from_config(args.config)
    print(f"Saved processed training data to: {output_path}")


if __name__ == "__main__":
    main()
