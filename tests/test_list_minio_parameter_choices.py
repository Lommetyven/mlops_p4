import io
import json

from scripts.list_minio_parameter_choices import build_parameter_choices


class FakeReadableFilesystem:
    def __init__(self, paths=None, files=None):
        self.paths = paths or []
        self.files = files or {}

    def exists(self, path):
        return path in self.files

    def open(self, path, mode="r"):
        if "r" not in mode:
            raise ValueError("FakeReadableFilesystem only supports reads.")
        return io.StringIO(self.files[path])

    def find(self, root):
        return [path for path in self.paths if path.startswith(root)]


def test_build_parameter_choices_reads_manifest_uploads():
    manifest = {
        "uploads": [
            {"s3_key": "readable_artifacts/processed/files/household_power_gru.csv"},
            {"s3_key": "readable_artifacts/models/files/gru_model.pt"},
            {"s3_key": "readable_artifacts/models/files/gru_model_torchscript.pt"},
            {"s3_key": "readable_artifacts/models/model_card.md"},
        ]
    }
    filesystem = FakeReadableFilesystem(
        files={
            "energyconsumption/readable_artifacts/manifest.json": json.dumps(manifest)
        }
    )

    choices = build_parameter_choices(filesystem)

    assert choices == {
        "datasets": ["data/processed/household_power_gru.csv"],
        "model_versions": ["gru_model", "gru_model_torchscript"],
    }


def test_build_parameter_choices_falls_back_to_listing_prefixes():
    filesystem = FakeReadableFilesystem(
        paths=[
            "energyconsumption/readable_artifacts/processed/files/a.csv",
            "energyconsumption/readable_artifacts/processed/files/nested/b.csv",
            "energyconsumption/readable_artifacts/models/files/run-a.pt",
            "energyconsumption/readable_artifacts/models/model_card.md",
        ]
    )

    choices = build_parameter_choices(filesystem)

    assert choices == {
        "datasets": ["data/processed/a.csv", "data/processed/nested/b.csv"],
        "model_versions": ["run-a"],
    }
