#!/usr/bin/env bash
#SBATCH --job-name=energy-gru-monitoring
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4
#SBATCH --output=reports/monitoring-slurm-%j.out
#SBATCH --error=reports/monitoring-slurm-%j.err

set -eu

cd "${AI_LAB_REPO_PATH:-/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-reports/runtime_train_config.yaml}"
MONITORING_CONFIG_PATH="${MONITORING_CONFIG_PATH:-reports/runtime_monitoring_config.yaml}"
DRIFT_CONFIG_PATH="${DRIFT_CONFIG_PATH:-configs/drift_monitoring_config.yaml}"
EVIDENTLY_OUTPUT_DIR="${EVIDENTLY_OUTPUT_DIR:-reports/evidently}"

echo "Running Evidently monitoring on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("The AI Lab monitoring job requires a CUDA GPU.")
print("cuda_device", torch.cuda.get_device_name(0))
PY

WANDB_ARGS=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS="--wandb"
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/evidently_monitoring.py \
    --checkpoint models/gru_model.pt \
    --config "$TRAIN_CONFIG_PATH" \
    --drift-config "$DRIFT_CONFIG_PATH" \
    --monitoring-config "$MONITORING_CONFIG_PATH" \
    --output-dir "$EVIDENTLY_OUTPUT_DIR" \
    --device cuda \
    $WANDB_ARGS
