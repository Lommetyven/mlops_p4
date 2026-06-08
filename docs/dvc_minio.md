# DVC MinIO Setup

This project uses the MinIO bucket/prefix below as the only DVC remote target:

```text
s3://energyconsumption/dvc
```

DVC fully owns this prefix for versioning and cache objects. Do not manually add,
edit, or delete objects under `energyconsumption/dvc`.

Human-readable copies are uploaded separately under:

```text
s3://energyconsumption/readable_artifacts
```

That prefix contains named raw data, processed data, model files, tar archives,
and a `manifest.json` with sizes and SHA256 hashes.

The MinIO API endpoint is:

```text
http://172.24.198.42:9001
```

The browser UI is:

```text
http://172.24.198.42:9001/browser/energyconsumption
```

Do not use DVC garbage collection against the remote unless the team has agreed on it. `dvc push` and `dvc pull` are safe for normal use; they only operate through the configured `energyconsumption/dvc` prefix.

## Credentials

Credentials are intentionally not committed. Configure them locally with one of these options.

Option 1, local DVC config:

```powershell
python -m dvc remote modify --local minio access_key_id "<MINIO_ACCESS_KEY>"
python -m dvc remote modify --local minio secret_access_key "<MINIO_SECRET_KEY>"
```

Option 2, environment variables:

```powershell
$env:AWS_ACCESS_KEY_ID="<MINIO_ACCESS_KEY>"
$env:AWS_SECRET_ACCESS_KEY="<MINIO_SECRET_KEY>"
```

## Reproduce And Push

Build the raw, processed, and model archives locally:

```powershell
python -m dvc repro
```

Push DVC-tracked archives to MinIO:

```powershell
python -m dvc push -r minio
```

Upload named, human-readable objects to the separate prefix:

```powershell
python scripts/upload_readable_artifacts.py
```

This writes only under:

```text
s3://energyconsumption/readable_artifacts
```

## Pull And Restore

Fetch DVC-tracked archives:

```powershell
python -m dvc pull -r minio
```

Restore raw and processed data archives:

```powershell
python scripts/archive_paths.py unpack --archive data/dvc_archives/raw.tar.gz --output data
python scripts/archive_paths.py unpack --archive data/dvc_archives/processed.tar.gz --output data
```

Restore model archives:

```powershell
python scripts/archive_paths.py unpack --archive data/dvc_archives/models.tar.gz --output .
```

## After Training

After `python main.py` creates or updates `models/gru_model.pt`, refresh and push the model archive:

```powershell
python -m dvc repro archive_models
python -m dvc push -r minio
```
