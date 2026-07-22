import json
import os
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

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

LOCAL_DATASET_ROOT = PurePosixPath("data/processed")


def normalize_dataset_path(dataset_path):
    value = str(dataset_path or "").strip().replace("\\", "/")
    if not value:
        raise ValueError("Dataset path must not be blank.")

    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        raise ValueError("Dataset path must be relative to the workspace.")

    try:
        relative_path = candidate.relative_to(LOCAL_DATASET_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Dataset path must be inside {LOCAL_DATASET_ROOT.as_posix()}/."
        ) from error

    if not relative_path.parts or ".." in relative_path.parts:
        raise ValueError("Dataset path must identify a file inside data/processed/.")

    return LOCAL_DATASET_ROOT / relative_path


def readable_dataset_key(dataset_path, prefix=DEFAULT_PREFIX):
    normalized_path = normalize_dataset_path(dataset_path)
    relative_path = normalized_path.relative_to(LOCAL_DATASET_ROOT)
    return f"{prefix.strip('/')}/processed/files/{relative_path.as_posix()}"


def read_expected_metadata(filesystem, bucket, prefix, key):
    manifest_path = f"{bucket}/{prefix.strip('/')}/manifest.json"
    if not filesystem.exists(manifest_path):
        return None

    with filesystem.open(manifest_path, "r") as manifest_file:
        manifest = json.load(manifest_file)

    return next(
        (
            upload
            for upload in manifest.get("uploads", [])
            if isinstance(upload, dict) and upload.get("s3_key") == key
        ),
        None,
    )


def validate_download(metadata, expected_metadata):
    if not expected_metadata:
        return

    expected_size = expected_metadata.get("size_bytes")
    if expected_size is not None and metadata["size_bytes"] != expected_size:
        raise ValueError(
            "Downloaded dataset size does not match readable_artifacts/manifest.json."
        )

    expected_sha256 = expected_metadata.get("sha256")
    if expected_sha256 and metadata["sha256"] != expected_sha256:
        raise ValueError(
            "Downloaded dataset checksum does not match "
            "readable_artifacts/manifest.json."
        )


def filesystem_for_remote(remote_name):
    access_key, secret_key, endpoint_url = load_minio_credentials(remote_name)
    return build_s3_filesystem(access_key, secret_key, endpoint_url)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def pull_dataset(
    dataset_path,
    bucket=DEFAULT_BUCKET,
    prefix=DEFAULT_PREFIX,
    remote_name="minio",
    filesystem=None,
    manifest_output="reports/restored_dataset_manifest.json",
):
    normalized_path = normalize_dataset_path(dataset_path)
    key = readable_dataset_key(normalized_path.as_posix(), prefix)
    remote_path = f"{bucket}/{key}"
    filesystem = filesystem or filesystem_for_remote(remote_name)

    if not filesystem.exists(remote_path):
        raise FileNotFoundError(
            f"Selected readable dataset not found: s3://{remote_path}"
        )

    workspace_root = Path.cwd().resolve()
    allowed_root = (workspace_root / LOCAL_DATASET_ROOT.as_posix()).resolve()
    destination = (workspace_root / normalized_path.as_posix()).resolve()
    if not destination.is_relative_to(allowed_root) or destination == allowed_root:
        raise ValueError("Dataset destination escaped data/processed/.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    expected_metadata = read_expected_metadata(filesystem, bucket, prefix, key)

    try:
        filesystem.get_file(remote_path, str(temporary_path))
        metadata = collect_file_metadata(temporary_path, key)
        validate_download(metadata, expected_metadata)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    metadata = collect_file_metadata(destination, key)
    metadata["local_path"] = normalized_path.as_posix()
    result = {
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "source_key": key,
        "dataset": metadata,
        "manifest_verified": expected_metadata is not None,
    }
    write_json(manifest_output, result)
    print(f"Restored s3://{remote_path} -> {normalized_path.as_posix()}")
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--remote-name", default="minio")
    parser.add_argument(
        "--manifest-output",
        default="reports/restored_dataset_manifest.json",
    )
    args = parser.parse_args()

    pull_dataset(
        dataset_path=args.dataset_path,
        bucket=args.bucket,
        prefix=args.prefix,
        remote_name=args.remote_name,
        manifest_output=args.manifest_output,
    )


if __name__ == "__main__":
    main()
