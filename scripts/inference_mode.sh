#!/usr/bin/env bash
#SBATCH --job-name=energy-gru-inference
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4
#SBATCH --output=reports/inference-slurm-%j.out
#SBATCH --error=reports/inference-slurm-%j.err

set -eu

cd "${AI_LAB_REPO_PATH:-/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INFERENCE_SEQUENCE_LENGTH="${INFERENCE_SEQUENCE_LENGTH:-$(cat reports/inference_sequence_length.txt)}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-FP32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1024}"

echo "Running inference on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

WANDB_ARGS=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS="--wandb"
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/run_torchscript_inference.py \
    --model models/gru_model_torchscript.pt \
    --input reports/inference_window.csv \
    --predictions reports/inference_predictions.txt \
    --metrics reports/inference_metrics.json \
    --precision "$INFERENCE_PRECISION" \
    --sequence-length "$INFERENCE_SEQUENCE_LENGTH" \
    --batch-size "$INFERENCE_BATCH_SIZE" \
    --config reports/runtime_train_config.yaml \
    --monitoring-config reports/runtime_monitoring_config.yaml \
    --runner AI_LAB \
    $WANDB_ARGS
