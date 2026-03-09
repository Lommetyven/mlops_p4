"""Data loading and preprocessing utilities for household power forecasting."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

RAW_FILENAME = "household_power_consumption.txt"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip"
)
TARGET_COL = "Global_active_power"
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TIME_FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


@dataclass
class PreparedData:
    """Container for data objects used by training and evaluation."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_cols: List[str]
    target_idx: int
    split_sizes: Dict[str, int]


class SequenceDataset(Dataset):
    """Creates (lookback, horizon) windows from hourly time series data."""

    def __init__(
        self,
        scaled_features: np.ndarray,
        target_idx: int,
        lookback: int,
        horizon: int,
        start_indices: np.ndarray,
    ) -> None:
        self.scaled_features = scaled_features
        self.target_idx = target_idx
        self.lookback = lookback
        self.horizon = horizon
        self.start_indices = start_indices

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = int(self.start_indices[idx])
        x = self.scaled_features[start - self.lookback : start]
        y = self.scaled_features[start : start + self.horizon, self.target_idx]
        return x.astype(np.float32), y.astype(np.float32)[:, None]


def download_raw_data(data_dir: Path, timeout: int = 60) -> Path:
    """Download and extract the UCI household power dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_FILENAME

    if raw_path.exists():
        return raw_path

    response = requests.get(UCI_ZIP_URL, timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        members = zf.namelist()
        if RAW_FILENAME not in members:
            raise FileNotFoundError(
                f"{RAW_FILENAME} not found in downloaded archive. Members: {members}"
            )
        zf.extract(RAW_FILENAME, path=data_dir)

    return raw_path


def _resolve_raw_path(data_dir: Path, data_path: str | None, download: bool) -> Path:
    if data_path:
        raw_path = Path(data_path)
        if raw_path.exists():
            return raw_path
        raise FileNotFoundError(f"Provided data_path does not exist: {raw_path}")

    candidate = data_dir / RAW_FILENAME
    if candidate.exists():
        return candidate

    local_candidate = Path(RAW_FILENAME)
    if local_candidate.exists():
        return local_candidate

    if download:
        return download_raw_data(data_dir)

    raise FileNotFoundError(
        "Raw dataset not found. Place household_power_consumption.txt in ./data/ "
        "or pass --data_path, or run with --download."
    )


def _build_hourly_frame(raw_path: Path) -> pd.DataFrame:
    cols = ["Date", "Time", *NUMERIC_COLS]
    df = pd.read_csv(
        raw_path,
        sep=";",
        usecols=cols,
        na_values=["?"],
        low_memory=False,
        dtype={"Date": "string", "Time": "string"},
    )

    dt = pd.to_datetime(
        df["Date"].str.cat(df["Time"], sep=" "),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
    df.index = dt
    df = df[~df.index.isna()].sort_index()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mean_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
    ]
    sum_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

    hourly = df.resample("1h").agg(
        {**{c: "mean" for c in mean_cols}, **{c: "sum" for c in sum_cols}}
    )

    hourly = hourly.interpolate(method="time").ffill().bfill()

    idx = hourly.index
    hourly["hour_sin"] = np.sin(2.0 * np.pi * idx.hour / 24.0)
    hourly["hour_cos"] = np.cos(2.0 * np.pi * idx.hour / 24.0)
    hourly["dow_sin"] = np.sin(2.0 * np.pi * idx.dayofweek / 7.0)
    hourly["dow_cos"] = np.cos(2.0 * np.pi * idx.dayofweek / 7.0)
    hourly["month_sin"] = np.sin(2.0 * np.pi * (idx.month - 1) / 12.0)
    hourly["month_cos"] = np.cos(2.0 * np.pi * (idx.month - 1) / 12.0)

    return hourly


def load_hourly_data(
    data_dir: str = "data",
    data_path: str | None = None,
    download: bool = False,
) -> pd.DataFrame:
    """Load cached hourly data when available, otherwise preprocess from raw file."""
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    raw_path = _resolve_raw_path(data_dir_path, data_path, download)

    parquet_cache = data_dir_path / "hourly_cleaned.parquet"
    csv_cache = data_dir_path / "hourly_cleaned.csv"

    if parquet_cache.exists():
        return pd.read_parquet(parquet_cache)
    if csv_cache.exists():
        return pd.read_csv(csv_cache, index_col=0, parse_dates=True)

    hourly = _build_hourly_frame(raw_path)

    try:
        hourly.to_parquet(parquet_cache)
    except Exception:
        hourly.to_csv(csv_cache)

    return hourly


def _split_start_indices(
    n_rows: int,
    lookback: int,
    horizon: int,
    train_end: int,
    val_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, val, test = [], [], []

    for start in range(lookback, n_rows - horizon + 1):
        end = start + horizon
        if end <= train_end:
            train.append(start)
        elif start >= train_end and end <= val_end:
            val.append(start)
        elif start >= val_end and end <= n_rows:
            test.append(start)

    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def prepare_dataloaders(
    hourly_df: pd.DataFrame,
    lookback: int = 168,
    horizon: int = 24,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
) -> PreparedData:
    """Create scaled datasets and dataloaders for chronological train/val/test splits."""
    feature_cols = [*NUMERIC_COLS, *TIME_FEATURE_COLS]
    n_rows = len(hourly_df)
    if n_rows <= lookback + horizon:
        raise ValueError(
            f"Not enough rows ({n_rows}) for lookback={lookback} and horizon={horizon}."
        )

    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    scaler = StandardScaler()
    scaler.fit(hourly_df.iloc[:train_end][feature_cols].values)

    scaled_features = scaler.transform(hourly_df[feature_cols].values).astype(np.float32)
    target_idx = feature_cols.index(TARGET_COL)

    train_idx, val_idx, test_idx = _split_start_indices(
        n_rows=n_rows,
        lookback=lookback,
        horizon=horizon,
        train_end=train_end,
        val_end=val_end,
    )

    train_ds = SequenceDataset(scaled_features, target_idx, lookback, horizon, train_idx)
    val_ds = SequenceDataset(scaled_features, target_idx, lookback, horizon, val_idx)
    test_ds = SequenceDataset(scaled_features, target_idx, lookback, horizon, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    split_sizes = {
        "train_rows": train_end,
        "val_rows": val_end - train_end,
        "test_rows": n_rows - val_end,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        feature_cols=feature_cols,
        target_idx=target_idx,
        split_sizes=split_sizes,
    )


def save_scaler(path: str | Path, scaler: StandardScaler, feature_cols: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scaler": scaler, "feature_cols": feature_cols, "target_col": TARGET_COL}
    joblib.dump(payload, path)


def inverse_scale_target(
    values: np.ndarray,
    scaler: StandardScaler,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
) -> np.ndarray:
    """Inverse-transform scaled target values using scaler statistics."""
    target_idx = feature_cols.index(target_col)
    mean = scaler.mean_[target_idx]
    scale = scaler.scale_[target_idx]
    return values * scale + mean
