from pathlib import Path

from scripts.upload_readable_artifacts import (
    collect_file_metadata,
    load_minio_credentials,
    remove_stale_archive_uploads,
    upload_readable_artifacts,
)


class FakeFilesystem:
    def __init__(self, existing_paths):
        self.existing_paths = set(existing_paths)
        self.removed_paths = []

    def exists(self, path):
        return path in self.existing_paths

    def rm(self, path):
        self.existing_paths.remove(path)
        self.removed_paths.append(path)


class FakeUploadFilesystem(FakeFilesystem):
    def __init__(self):
        super().__init__(set())
        self.uploaded_paths = []

    def put_file(self, local_path, remote_path):
        self.uploaded_paths.append(remote_path)


def test_load_minio_credentials_from_dvc_config_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

    config_path = Path(".dvc/config.local")
    config_path.parent.mkdir()
    config_path.write_text(
        """
[remote "minio"]
    access_key_id = test-key
    secret_access_key = test-secret
    endpointurl = http://localhost:9000
""",
        encoding="utf-8",
    )

    assert load_minio_credentials() == (
        "test-key",
        "test-secret",
        "http://localhost:9000",
    )


def test_collect_file_metadata_contains_human_readable_s3_key(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    metadata = collect_file_metadata(
        file_path,
        "readable_artifacts/processed/files/sample.csv",
    )

    assert metadata["s3_key"] == "readable_artifacts/processed/files/sample.csv"
    assert metadata["size_bytes"] > 0
    assert len(metadata["sha256"]) == 64


def test_remove_stale_archive_uploads_removes_only_readable_archives():
    filesystem = FakeFilesystem(
        {
            "energyconsumption/readable_artifacts/raw/raw.tar.gz",
            "energyconsumption/readable_artifacts/processed/processed.tar.gz",
            "energyconsumption/readable_artifacts/models/models.tar.gz",
            "energyconsumption/readable_artifacts/models/files/model.pt",
            "energyconsumption/dvc/files/md5/ab/archive",
        }
    )

    remove_stale_archive_uploads(
        filesystem,
        bucket="energyconsumption",
        prefix="readable_artifacts",
    )

    assert filesystem.removed_paths == [
        "energyconsumption/readable_artifacts/raw/raw.tar.gz",
        "energyconsumption/readable_artifacts/processed/processed.tar.gz",
        "energyconsumption/readable_artifacts/models/models.tar.gz",
    ]
    assert "energyconsumption/readable_artifacts/models/files/model.pt" in (
        filesystem.existing_paths
    )
    assert "energyconsumption/dvc/files/md5/ab/archive" in filesystem.existing_paths


def test_upload_readable_artifacts_can_exclude_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "data/raw/raw.txt").write_text("raw", encoding="utf-8")
    (tmp_path / "data/processed/data.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "models/gru_model.pt").write_bytes(b"weights")
    filesystem = FakeUploadFilesystem()
    monkeypatch.setattr(
        "scripts.upload_readable_artifacts.load_minio_credentials",
        lambda _remote_name: ("key", "secret", "http://minio"),
    )
    monkeypatch.setattr(
        "scripts.upload_readable_artifacts.build_s3_filesystem",
        lambda *_args: filesystem,
    )

    manifest = upload_readable_artifacts(include_models=False)

    assert all("/models/" not in path for path in filesystem.uploaded_paths)
    assert all("/models/" not in upload["s3_key"] for upload in manifest["uploads"])
