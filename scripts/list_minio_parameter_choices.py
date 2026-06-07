import json
from argparse import ArgumentParser
from pathlib import Path

from scripts.upload_readable_artifacts import (
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    build_s3_filesystem,
    load_minio_credentials,
)


def normalize_key(path, bucket):
    prefix = f"{bucket}/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def read_manifest(filesystem, bucket, prefix):
    manifest_path = f"{bucket}/{prefix}/manifest.json"
    if not filesystem.exists(manifest_path):
        return None

    with filesystem.open(manifest_path, "r") as manifest_file:
        return json.load(manifest_file)


def list_keys(filesystem, bucket, prefix):
    root = f"{bucket}/{prefix}"
    try:
        return [
            normalize_key(path, bucket)
            for path in filesystem.find(root)
            if not path.endswith("/")
        ]
    except FileNotFoundError:
        return []


def keys_from_manifest(manifest):
    if not manifest:
        return []

    return [
        upload["s3_key"]
        for upload in manifest.get("uploads", [])
        if isinstance(upload, dict) and upload.get("s3_key")
    ]


def dataset_choices_from_keys(keys, prefix):
    processed_prefix = f"{prefix}/processed/files/"
    choices = []
    for key in keys:
        if not key.startswith(processed_prefix):
            continue

        relative_path = key.removeprefix(processed_prefix)
        if relative_path:
            choices.append(f"data/processed/{relative_path}")

    return sorted(set(choices))


def model_version_choices_from_keys(keys, prefix):
    models_prefix = f"{prefix}/models/files/"
    choices = []
    for key in keys:
        if not key.startswith(models_prefix):
            continue

        relative_path = key.removeprefix(models_prefix)
        if relative_path:
            choices.append(Path(relative_path).stem)

    return sorted(set(choices))


def build_parameter_choices(filesystem, bucket=DEFAULT_BUCKET, prefix=DEFAULT_PREFIX):
    prefix = prefix.strip("/")
    manifest = read_manifest(filesystem, bucket, prefix)
    keys = keys_from_manifest(manifest)
    if not keys:
        keys = list_keys(filesystem, bucket, prefix)

    return {
        "datasets": dataset_choices_from_keys(keys, prefix),
        "model_versions": model_version_choices_from_keys(keys, prefix),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--remote-name", default="minio")
    parser.add_argument("--output", default="reports/minio_parameter_choices.json")
    args = parser.parse_args()

    access_key, secret_key, endpoint_url = load_minio_credentials(args.remote_name)
    filesystem = build_s3_filesystem(access_key, secret_key, endpoint_url)
    choices = build_parameter_choices(
        filesystem,
        bucket=args.bucket,
        prefix=args.prefix,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(choices, indent=2), encoding="utf-8")
    print(json.dumps(choices, indent=2))


if __name__ == "__main__":
    main()
