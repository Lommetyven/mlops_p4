import io
import json
from pathlib import Path

import pytest

from scripts.dataset_store import (
    normalize_dataset_path,
    pull_dataset,
    readable_dataset_key,
)


class FakeDatasetFilesystem:
    def __init__(self, files):
        self.files = files

    def exists(self, path):
        return path in self.files

    def open(self, path, mode="r"):
        if mode != "r":
            raise ValueError("FakeDatasetFilesystem only supports text reads.")
        return io.StringIO(self.files[path].decode("utf-8"))

    def get_file(self, remote_path, local_path):
        Path(local_path).write_bytes(self.files[remote_path])


def test_readable_dataset_key_maps_only_processed_paths():
    assert readable_dataset_key("data/processed/nested/data.csv") == (
        "readable_artifacts/processed/files/nested/data.csv"
    )

    for invalid_path in (
        "data/raw/data.csv",
        "data/processed/../raw/data.csv",
        "/data/processed/data.csv",
    ):
        with pytest.raises(ValueError):
            normalize_dataset_path(invalid_path)


def test_pull_dataset_replaces_dvc_copy_and_verifies_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    destination = Path("data/processed/data.csv")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-dvc-copy")
    content = b"value\n1\n"
    key = "readable_artifacts/processed/files/data.csv"
    manifest = {
        "uploads": [
            {
                "s3_key": key,
                "size_bytes": len(content),
                "sha256": (
                    "1a80986111952a11d02e84dbed98ae00f279469aad0615d17fa81911f8a6b428"
                ),
            }
        ]
    }
    filesystem = FakeDatasetFilesystem(
        {
            f"energyconsumption/{key}": content,
            "energyconsumption/readable_artifacts/manifest.json": json.dumps(
                manifest
            ).encode("utf-8"),
        }
    )

    restored = pull_dataset(
        "data/processed/data.csv",
        filesystem=filesystem,
    )

    assert destination.read_bytes() == content
    assert restored["source_key"] == key
    assert restored["manifest_verified"] is True
    assert Path("reports/restored_dataset_manifest.json").is_file()


def test_pull_dataset_keeps_existing_file_when_checksum_is_wrong(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    destination = Path("data/processed/data.csv")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-dvc-copy")
    key = "readable_artifacts/processed/files/data.csv"
    manifest = {
        "uploads": [
            {
                "s3_key": key,
                "size_bytes": 4,
                "sha256": "0" * 64,
            }
        ]
    }
    filesystem = FakeDatasetFilesystem(
        {
            f"energyconsumption/{key}": b"new-copy",
            "energyconsumption/readable_artifacts/manifest.json": json.dumps(
                manifest
            ).encode("utf-8"),
        }
    )

    with pytest.raises(ValueError, match="size"):
        pull_dataset("data/processed/data.csv", filesystem=filesystem)

    assert destination.read_bytes() == b"old-dvc-copy"
