from pathlib import Path

from scripts.upload_readable_artifacts import (
    collect_file_metadata,
    load_minio_credentials,
)


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
