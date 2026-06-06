import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class WandbMonitorConfig:
    project: str = "MLOps"
    entity: str | None = "tobiasr-aalborg-universitet"
    run_name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    mode: str = "online"
    watch_log: str = "all"
    watch_log_freq: int = 100
    hardware_log_freq: int = 50
    weight_histogram_log_freq: int = 0
    save_code: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/monitering_config.yaml"):
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}

        monitoring_config = raw_config.get("monitoring", raw_config)
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_config = {
            key: value for key, value in monitoring_config.items() if key in valid_keys
        }

        return cls(**filtered_config)


class WandbMonitor:
    def __init__(self, config: WandbMonitorConfig | None = None):
        self.config = config or WandbMonitorConfig()
        self.run = None
        self._wandb = None
        self._started_at = time.time()
        self._last_hardware_step = None

    def start(
        self,
        model=None,
        training_config: Mapping[str, Any] | None = None,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for monitoring. Install it with `pip install wandb`."
            ) from exc

        self._wandb = wandb
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            notes=self.config.notes,
            tags=self.config.tags,
            mode=self.config.mode,
            config=dict(training_config or {}),
            save_code=self.config.save_code,
        )

        if model is not None:
            self.watch_model(model)

        self.log_hardware(step=0, force=True)
        return self

    def watch_model(self, model):
        self._require_run()
        self._wandb.watch(
            model,
            log=self.config.watch_log,
            log_freq=self.config.watch_log_freq,
        )

    def log_training_step(
        self,
        step: int,
        epoch: int | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        learning_rate: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        model=None,
    ):
        self._require_run()

        payload = {}
        if epoch is not None:
            payload["train/epoch"] = epoch
        if train_loss is not None:
            payload["train/loss"] = float(train_loss)
        if val_loss is not None:
            payload["val/loss"] = float(val_loss)
        if learning_rate is not None:
            payload["train/learning_rate"] = float(learning_rate)
        if metrics:
            payload.update(metrics)

        if payload:
            self._wandb.log(payload, step=step)

        self.log_hardware(step=step)

        if (
            model is not None
            and self.config.weight_histogram_log_freq > 0
            and step % self.config.weight_histogram_log_freq == 0
        ):
            self.log_weight_histograms(model, step=step)

    def log_hardware(self, step: int | None = None, force: bool = False):
        self._require_run()

        if not force and step is not None:
            if step % self.config.hardware_log_freq != 0:
                return
            if self._last_hardware_step == step:
                return

        metrics = collect_hardware_metrics(started_at=self._started_at)
        self._wandb.log(metrics, step=step)
        self._last_hardware_step = step

    def log_weight_histograms(self, model, step: int | None = None):
        self._require_run()

        histograms = {}
        for name, parameter in model.named_parameters():
            histograms[f"weights/{name}"] = self._wandb.Histogram(
                parameter.detach().cpu().flatten().numpy()
            )

        self._wandb.log(histograms, step=step)

    def log_dataset_artifact(
        self,
        dataset_path,
        artifact_name,
        artifact_type="dataset",
        metadata: Mapping[str, Any] | None = None,
        upload_dataset=False,
        aliases=None,
        repo_path=".",
    ):
        self._require_run()

        dataset_metadata = collect_dataset_metadata(
            dataset_path=dataset_path,
            repo_path=repo_path,
        )
        if metadata:
            dataset_metadata.update(metadata)

        artifact = self._wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=dataset_metadata,
        )

        dataset_path = Path(dataset_path)
        if upload_dataset and dataset_path.exists():
            artifact.add_file(str(dataset_path), name=dataset_path.name)
        else:
            manifest_path = Path(self.run.dir) / "dataset_metadata.json"
            manifest_path.write_text(
                json.dumps(dataset_metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifact.add_file(str(manifest_path), name="dataset_metadata.json")

        self.run.log_artifact(artifact, aliases=list(aliases or []))
        return artifact

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None):
        self._require_run()
        if metrics:
            self._wandb.log(dict(metrics), step=step)

    def update_summary(self, values: Mapping[str, Any]):
        self._require_run()
        if values:
            self.run.summary.update(dict(values))

    def log_files_artifact(
        self,
        files,
        artifact_name,
        artifact_type="run-logs",
        metadata: Mapping[str, Any] | None = None,
        aliases=None,
    ):
        self._require_run()

        artifact = self._wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=dict(metadata or {}),
        )
        added = 0
        for file_path in files:
            path = Path(file_path)
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
                added += 1

        if added == 0:
            return None

        self.run.log_artifact(artifact, aliases=list(aliases or []))
        return artifact

    def log_model_artifact(
        self,
        checkpoint_path,
        artifact_name,
        artifact_type="model",
        metadata: Mapping[str, Any] | None = None,
        aliases=None,
    ):
        self._require_run()

        path = Path(checkpoint_path)
        if not path.is_file():
            return None

        artifact = self._wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=dict(metadata or {}),
        )
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=list(aliases or []))
        return artifact

    def log_confusion_matrix(
        self,
        confusion_matrix,
        class_names,
        title="confusion_matrix",
        step: int | None = None,
    ):
        self._require_run()

        y_true = []
        preds = []
        for true_index, row in enumerate(confusion_matrix):
            for predicted_index, count in enumerate(row):
                y_true.extend([true_index] * int(count))
                preds.extend([predicted_index] * int(count))

        if not y_true:
            return None

        plot = self._wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=preds,
            class_names=list(class_names),
        )
        self._wandb.log({title: plot}, step=step)
        return plot

    def finish(self):
        if self.run is not None and self._wandb is not None:
            self._wandb.finish()
            self.run = None

    def _require_run(self):
        if self.run is None or self._wandb is None:
            raise RuntimeError("Call `start()` before logging monitoring data.")

    def __enter__(self):
        if self.run is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.finish()


def collect_hardware_metrics(started_at: float | None = None):
    metrics = {}

    if started_at is not None:
        metrics["hardware/runtime_seconds"] = time.time() - started_at

    if psutil is not None:
        memory = psutil.virtual_memory()
        metrics.update(
            {
                "hardware/cpu_percent": psutil.cpu_percent(interval=None),
                "hardware/ram_percent": memory.percent,
                "hardware/ram_used_gb": memory.used / 1024**3,
                "hardware/ram_available_gb": memory.available / 1024**3,
            }
        )

    metrics.update(_collect_torch_cuda_metrics())
    metrics.update(_collect_nvidia_smi_metrics())

    return metrics


def _collect_torch_cuda_metrics():
    try:
        import torch
    except ImportError:
        return {"hardware/gpu_available": 0}

    if not torch.cuda.is_available():
        return {"hardware/gpu_available": 0}

    metrics = {"hardware/gpu_available": 1}
    for device_index in range(torch.cuda.device_count()):
        metrics.update(
            {
                f"hardware/gpu_{device_index}_memory_allocated_gb": (
                    torch.cuda.memory_allocated(device_index) / 1024**3
                ),
                f"hardware/gpu_{device_index}_memory_reserved_gb": (
                    torch.cuda.memory_reserved(device_index) / 1024**3
                ),
                f"hardware/gpu_{device_index}_max_memory_allocated_gb": (
                    torch.cuda.max_memory_allocated(device_index) / 1024**3
                ),
            }
        )

    return metrics


def _collect_nvidia_smi_metrics():
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    if result.returncode != 0:
        return {}

    metrics = {}
    for device_index, line in enumerate(result.stdout.strip().splitlines()):
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            continue

        gpu_util, memory_used, memory_total, temperature, power_draw = values
        prefix = f"hardware/gpu_{device_index}"
        metrics.update(
            {
                f"{prefix}_util_percent": _to_float(gpu_util),
                f"{prefix}_memory_used_mb": _to_float(memory_used),
                f"{prefix}_memory_total_mb": _to_float(memory_total),
                f"{prefix}_temperature_c": _to_float(temperature),
                f"{prefix}_power_watts": _to_float(power_draw),
            }
        )

    return metrics


def _to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_dataset_metadata(dataset_path, repo_path="."):
    path = Path(dataset_path)
    metadata = {
        "dataset/path": str(path),
        "dataset/exists": path.exists(),
        "git": collect_git_metadata(repo_path=repo_path),
        "dvc": collect_dvc_metadata(repo_path=repo_path),
    }

    if path.exists():
        metadata.update(
            {
                "dataset/size_bytes": path.stat().st_size,
                "dataset/sha256": file_sha256(path),
            }
        )

    return metadata


def collect_git_metadata(repo_path="."):
    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"], cwd=repo_path),
        "branch": _run_command(["git", "branch", "--show-current"], cwd=repo_path),
        "status_short": _run_command(["git", "status", "--short"], cwd=repo_path),
    }


def collect_dvc_metadata(repo_path="."):
    version = _run_dvc_command(["--version"], cwd=repo_path)
    status = _run_dvc_command(["status"], cwd=repo_path)

    metadata = {
        "available": version is not None,
        "version": version,
        "status": status,
        "remote_list": _run_dvc_command(["remote", "list"], cwd=repo_path),
        "dvc_yaml_sha256": _optional_file_sha256(Path(repo_path) / "dvc.yaml"),
        "dvc_lock_sha256": _optional_file_sha256(Path(repo_path) / "dvc.lock"),
    }
    return metadata


def _run_dvc_command(arguments, cwd="."):
    output = _run_command(["dvc", *arguments], cwd=cwd)
    if output is not None:
        return output

    return _run_command([sys.executable, "-m", "dvc", *arguments], cwd=cwd)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path):
    path = Path(path)
    if not path.exists():
        return None
    return file_sha256(path)


def _run_command(command, cwd="."):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0:
        return output or None

    return output


def init_wandb_monitor(
    model=None,
    training_config: Mapping[str, Any] | None = None,
    config_path: str | Path = "configs/monitering_config.yaml",
):
    monitor_config = WandbMonitorConfig.from_yaml(config_path)
    return WandbMonitor(monitor_config).start(
        model=model,
        training_config=training_config,
    )
