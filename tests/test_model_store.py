import io
import json
from pathlib import Path

from scripts.model_store import normalize_version, pull_model, save_model


class FakeModelFilesystem:
    def __init__(self):
        self.files = {}

    def exists(self, path):
        return path in self.files

    def put_file(self, local_path, remote_path):
        self.files[remote_path] = Path(local_path).read_bytes()

    def get_file(self, remote_path, local_path):
        Path(local_path).write_bytes(self.files[remote_path])

    def open(self, path, mode="r"):
        if mode != "r":
            raise ValueError("FakeModelFilesystem only supports text reads.")
        return io.StringIO(self.files[path].decode("utf-8"))


def test_save_and_pull_model_version_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filesystem = FakeModelFilesystem()
    models_dir = Path("models")
    models_dir.mkdir()
    Path("reports").mkdir()
    (models_dir / "gru_model.pt").write_bytes(b"checkpoint-weights")
    (models_dir / "gru_model_torchscript.pt").write_bytes(b"torchscript-weights")
    Path("reports/runtime_train_config.yaml").write_text(
        "experiment:\n"
        "  model_version: report-run-1\n"
        "training:\n"
        "  sequence_length: 24\n",
        encoding="utf-8",
    )
    Path("reports/model_card.md").write_text("# Model card\n", encoding="utf-8")

    saved = save_model(filesystem=filesystem)

    assert saved["version"] == "report-run-1"
    model_prefix = "energyconsumption/readable_artifacts/models/versions/report-run-1"
    assert filesystem.files[f"{model_prefix}/gru_model.pt"] == b"checkpoint-weights"
    assert filesystem.files[f"{model_prefix}/gru_model_torchscript.pt"] == (
        b"torchscript-weights"
    )
    latest = json.loads(
        filesystem.files[
            "energyconsumption/readable_artifacts/models/latest.json"
        ].decode("utf-8")
    )
    assert latest["version"] == "report-run-1"

    for model_path in models_dir.iterdir():
        model_path.unlink()
    restored = pull_model(filesystem=filesystem)

    assert restored["version"] == "report-run-1"
    assert (models_dir / "gru_model.pt").read_bytes() == b"checkpoint-weights"
    assert (models_dir / "gru_model_torchscript.pt").read_bytes() == (
        b"torchscript-weights"
    )
    assert (
        Path("reports/restored_model_config.yaml")
        .read_text(encoding="utf-8")
        .endswith("sequence_length: 24\n")
    )


def test_pull_model_supports_legacy_latest_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filesystem = FakeModelFilesystem()
    filesystem.files[
        "energyconsumption/readable_artifacts/models/files/gru_model_torchscript.pt"
    ] = b"legacy-torchscript"

    restored = pull_model(filesystem=filesystem)

    assert restored["version"] == "legacy-latest"
    assert Path("models/gru_model_torchscript.pt").read_bytes() == (
        b"legacy-torchscript"
    )


def test_normalize_version_rejects_path_like_values():
    try:
        normalize_version("../../model")
    except ValueError as error:
        assert "Model version" in str(error)
    else:
        raise AssertionError("Expected path-like model version to be rejected.")
