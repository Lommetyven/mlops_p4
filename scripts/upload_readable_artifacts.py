import hashlib
import json
import os
from argparse import ArgumentParser
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path

import s3fs

DEFAULT_BUCKET = "energyconsumption"
DEFAULT_PREFIX = "readable_artifacts"
DEFAULT_ENDPOINT_URL = "http://172.24.198.42:9001"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_metadata(path, s3_key):
    path = Path(path)
    return {
        "local_path": str(path),
        "s3_key": s3_key,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def load_minio_credentials(remote_name="minio"):
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL") or DEFAULT_ENDPOINT_URL

    config_path = Path(".dvc/config.local")
    if config_path.exists():
        parser = ConfigParser()
        parser.read(config_path)
        section_candidates = [
            f'remote "{remote_name}"',
            f"'remote \"{remote_name}\"'",
        ]
        section = next(
            (
                candidate
                for candidate in section_candidates
                if parser.has_section(candidate)
            ),
            None,
        )
        if section is not None:
            access_key = access_key or parser.get(
                section, "access_key_id", fallback=None
            )
            secret_key = secret_key or parser.get(
                section, "secret_access_key", fallback=None
            )
            endpoint_url = parser.get(section, "endpointurl", fallback=endpoint_url)

    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing MinIO credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            "or configure DVC local credentials."
        )

    return access_key, secret_key, endpoint_url


def build_s3_filesystem(access_key, secret_key, endpoint_url):
    return s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": endpoint_url},
    )


def upload_file(filesystem, local_path, bucket, key):
    local_path = Path(local_path)
    if not local_path.exists():
        return None

    destination = f"{bucket}/{key}"
    filesystem.put_file(str(local_path), destination)
    print(f"Uploaded {local_path} -> s3://{destination}")
    return collect_file_metadata(local_path, key)


def upload_directory(filesystem, local_directory, bucket, prefix):
    local_directory = Path(local_directory)
    if not local_directory.exists():
        return []

    uploads = []
    for path in sorted(local_directory.rglob("*")):
        if path.is_file():
            relative_path = path.relative_to(local_directory).as_posix()
            key = f"{prefix}/{relative_path}"
            metadata = upload_file(filesystem, path, bucket, key)
            if metadata is not None:
                uploads.append(metadata)

    return uploads


def remove_stale_archive_uploads(filesystem, bucket, prefix):
    stale_archive_keys = [
        f"{prefix}/raw/raw.tar.gz",
        f"{prefix}/processed/processed.tar.gz",
        f"{prefix}/models/models.tar.gz",
    ]
    for key in stale_archive_keys:
        destination = f"{bucket}/{key}"
        if filesystem.exists(destination):
            filesystem.rm(destination)
            print(f"Removed stale readable archive s3://{destination}")


def upload_manifest(filesystem, manifest, bucket, prefix):
    manifest_path = Path("data/dvc_archives/readable_artifacts_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    key = f"{prefix}/manifest.json"
    filesystem.put_file(str(manifest_path), f"{bucket}/{key}")
    print(f"Uploaded {manifest_path} -> s3://{bucket}/{key}")


def upload_readable_artifacts(
    bucket=DEFAULT_BUCKET,
    prefix=DEFAULT_PREFIX,
    remote_name="minio",
):
    access_key, secret_key, endpoint_url = load_minio_credentials(remote_name)
    filesystem = build_s3_filesystem(access_key, secret_key, endpoint_url)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket,
        "prefix": prefix,
        "endpoint_url": endpoint_url,
        "uploads": [],
    }

    upload_plan = [
        ("data/raw", f"{prefix}/raw/files"),
        ("data/processed", f"{prefix}/processed/files"),
        ("models", f"{prefix}/models/files"),
    ]
    for local_directory, s3_prefix in upload_plan:
        manifest["uploads"].extend(
            upload_directory(filesystem, local_directory, bucket, s3_prefix)
        )

    model_card = upload_file(
        filesystem,
        "reports/model_card.md",
        bucket,
        f"{prefix}/models/model_card.md",
    )
    if model_card is not None:
        manifest["uploads"].append(model_card)

    remove_stale_archive_uploads(filesystem, bucket, prefix)

    upload_manifest(filesystem, manifest, bucket, prefix)
    return manifest


def main():
    parser = ArgumentParser()
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--remote-name", default="minio")
    args = parser.parse_args()

    upload_readable_artifacts(
        bucket=args.bucket,
        prefix=args.prefix.strip("/"),
        remote_name=args.remote_name,
    )


if __name__ == "__main__":
    main()
