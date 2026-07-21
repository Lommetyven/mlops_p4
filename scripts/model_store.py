import json
import re
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from scripts.upload_readable_artifacts import (
        DEFAULT_BUCKET,
        DEFAULT_PREFIX,
        build_s3_filesystem,
        collect_file_metadata,
        load_minio_credentials,
    )
except ModuleNotFoundError:
    from upload_readable_artifacts import (
        DEFAULT_BUCKET,
        DEFAULT_PREFIX,
        build_s3_filesystem,
        collect_file_metadata,
        load_minio_credentials,
    )

VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MODEL_FILENAMES = (
    "gru_model.pt",
    "gru_model_torchscript.pt",
)


def normalize_version(version):
    normalized = str(version or "").strip()
    if not normalized or not VERSION_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Model version must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )
    return normalized


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def config_model_version(path):
    config = load_config(path)
    return normalize_version(config.get("experiment", {}).get("model_version"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_remote_json(filesystem, path):
    with filesystem.open(path, "r") as remote_file:
        return json.load(remote_file)


def filesystem_for_remote(remote_name):
    access_key, secret_key, endpoint_url = load_minio_credentials(remote_name)
    return build_s3_filesystem(access_key, secret_key, endpoint_url)


def upload_required_file(filesystem, local_path, bucket, key):
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Required model file not found: {local_path}")
    filesystem.put_file(str(local_path), f"{bucket}/{key}")
    print(f"Saved {local_path} -> s3://{bucket}/{key}")
    return collect_file_metadata(local_path, key)


def upload_optional_file(filesystem, local_path, bucket, key):
    local_path = Path(local_path)
    if not local_path.is_file():
        return None
    filesystem.put_file(str(local_path), f"{bucket}/{key}")
    print(f"Saved {local_path} -> s3://{bucket}/{key}")
    return collect_file_metadata(local_path, key)


def save_model(
    version=None,
    config_path="reports/runtime_train_config.yaml",
    models_dir="models",
    model_card_path="reports/model_card.md",
    bucket=DEFAULT_BUCKET,
    prefix=DEFAULT_PREFIX,
    remote_name="minio",
    filesystem=None,
    manifest_output="reports/saved_model_manifest.json",
):
    version = (
        normalize_version(version) if version else config_model_version(config_path)
    )
    prefix = prefix.strip("/")
    filesystem = filesystem or filesystem_for_remote(remote_name)
    version_prefix = f"{prefix}/models/versions/{version}"
    saved_at = datetime.now(timezone.utc).isoformat()

    uploads = []
    for filename in MODEL_FILENAMES:
        uploads.append(
            upload_required_file(
                filesystem,
                Path(models_dir) / filename,
                bucket,
                f"{version_prefix}/{filename}",
            )
        )

    for local_path, filename in (
        (config_path, "runtime_train_config.yaml"),
        (model_card_path, "model_card.md"),
    ):
        upload = upload_optional_file(
            filesystem,
            local_path,
            bucket,
            f"{version_prefix}/{filename}",
        )
        if upload is not None:
            uploads.append(upload)

    manifest = {
        "version": version,
        "saved_at": saved_at,
        "bucket": bucket,
        "prefix": prefix,
        "uploads": uploads,
    }
    local_manifest = write_json(manifest_output, manifest)
    manifest_key = f"{version_prefix}/manifest.json"
    filesystem.put_file(str(local_manifest), f"{bucket}/{manifest_key}")

    latest = {
        "version": version,
        "saved_at": saved_at,
        "manifest_key": manifest_key,
    }
    latest_path = write_json(
        Path(manifest_output).with_name("latest_model.json"),
        latest,
    )
    latest_key = f"{prefix}/models/latest.json"
    filesystem.put_file(str(latest_path), f"{bucket}/{latest_key}")
    print(f"Updated latest model -> {version}")
    return manifest


def download_file(filesystem, bucket, key, local_path, required=False):
    remote_path = f"{bucket}/{key}"
    if not filesystem.exists(remote_path):
        if required:
            raise FileNotFoundError(f"Saved model object not found: s3://{remote_path}")
        return None

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    filesystem.get_file(remote_path, str(local_path))
    print(f"Restored s3://{remote_path} -> {local_path}")
    return collect_file_metadata(local_path, key)


def resolve_model_source(filesystem, bucket, prefix, version):
    if version:
        normalized_version = normalize_version(version)
        return normalized_version, f"{prefix}/models/versions/{normalized_version}"

    latest_path = f"{bucket}/{prefix}/models/latest.json"
    if filesystem.exists(latest_path):
        latest = read_remote_json(filesystem, latest_path)
        normalized_version = normalize_version(latest.get("version"))
        return normalized_version, f"{prefix}/models/versions/{normalized_version}"

    return "legacy-latest", f"{prefix}/models/files"


def pull_model(
    version=None,
    models_dir="models",
    restored_config_path="reports/restored_model_config.yaml",
    restored_model_card_path="reports/restored_model_card.md",
    bucket=DEFAULT_BUCKET,
    prefix=DEFAULT_PREFIX,
    remote_name="minio",
    filesystem=None,
    manifest_output="reports/restored_model_manifest.json",
):
    prefix = prefix.strip("/")
    filesystem = filesystem or filesystem_for_remote(remote_name)
    resolved_version, source_prefix = resolve_model_source(
        filesystem,
        bucket,
        prefix,
        version,
    )

    downloads = []
    for filename in MODEL_FILENAMES:
        download = download_file(
            filesystem,
            bucket,
            f"{source_prefix}/{filename}",
            Path(models_dir) / filename,
            required=filename == "gru_model_torchscript.pt",
        )
        if download is not None:
            downloads.append(download)

    if resolved_version != "legacy-latest":
        for filename, local_path in (
            ("runtime_train_config.yaml", restored_config_path),
            ("model_card.md", restored_model_card_path),
        ):
            download = download_file(
                filesystem,
                bucket,
                f"{source_prefix}/{filename}",
                local_path,
            )
            if download is not None:
                downloads.append(download)

    manifest = {
        "version": resolved_version,
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "source_prefix": source_prefix,
        "downloads": downloads,
    }
    write_json(manifest_output, manifest)
    print(f"Restored model version: {resolved_version}")
    return manifest


def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=("save", "pull"))
    parser.add_argument("--version", default="")
    parser.add_argument("--config", default="reports/runtime_train_config.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--remote-name", default="minio")
    args = parser.parse_args()

    if args.action == "save":
        save_model(
            version=args.version or None,
            config_path=args.config,
            models_dir=args.models_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            remote_name=args.remote_name,
        )
    else:
        pull_model(
            version=args.version or None,
            models_dir=args.models_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            remote_name=args.remote_name,
        )


if __name__ == "__main__":
    main()
