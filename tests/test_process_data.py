import pandas as pd

from data_insight.process_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    process_household_power_data,
)


def test_process_household_power_data_creates_features_and_target():
    dataframe = pd.DataFrame(
        {
            "Date": ["16/12/2006", "16/12/2006", "16/12/2006"],
            "Time": ["17:24:00", "17:25:00", "18:24:00"],
            "Global_active_power": ["4.216", "5.360", "2.000"],
            "Global_reactive_power": ["0.418", "0.436", "0.100"],
            "Voltage": ["234.840", "233.630", "235.000"],
            "Global_intensity": ["18.400", "23.000", "10.000"],
            "Sub_metering_1": ["0.000", "0.000", "1.000"],
            "Sub_metering_2": ["1.000", "1.000", "0.000"],
            "Sub_metering_3": ["17.000", "16.000", "2.000"],
        }
    )

    processed = process_household_power_data(
        dataframe=dataframe,
        fill_missing="drop",
        time_interval="1h",
    )

    assert list(processed.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(processed) == 1
    assert processed[TARGET_COLUMN].isna().sum() == 0


def test_process_household_power_data_drops_nan_features_and_targets():
    dataframe = pd.DataFrame(
        {
            "Date": ["16/12/2006", "16/12/2006", "16/12/2006", "16/12/2006"],
            "Time": ["17:24:00", "18:24:00", "19:24:00", "20:24:00"],
            "Global_active_power": ["4.216", "2.000", "?", "3.000"],
            "Global_reactive_power": ["0.418", "0.100", "?", "0.200"],
            "Voltage": ["234.840", "235.000", "?", "236.000"],
            "Global_intensity": ["18.400", "10.000", "?", "11.000"],
            "Sub_metering_1": ["0.000", "1.000", "?", "2.000"],
            "Sub_metering_2": ["1.000", "0.000", "?", "3.000"],
            "Sub_metering_3": ["17.000", "2.000", "?", "4.000"],
        }
    )

    processed = process_household_power_data(
        dataframe=dataframe,
        fill_missing="none",
        time_interval="1h",
    )

    assert len(processed) == 1
    assert processed[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum() == 0
