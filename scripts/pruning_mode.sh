#!/usr/bin/env bash
#SBATCH --job-name=energy-gru-pruning
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4
#SBATCH --output=reports/pruning-slurm-%j.out
#SBATCH --error=reports/pruning-slurm-%j.err

set -eu

cd "${AI_LAB_REPO_PATH:-/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PRUNING_LEVELS="${PRUNING_LEVELS:-0,10,25,50,75,90}"
PRUNING_FINETUNE_EPOCHS="${PRUNING_FINETUNE_EPOCHS:-5}"
PRUNING_LEARNING_RATE="${PRUNING_LEARNING_RATE:-0.0001}"

echo "Running pruning experiment on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Pruning levels: $PRUNING_LEVELS"
echo "Fine-tune epochs per level: $PRUNING_FINETUNE_EPOCHS"
echo "Fine-tune learning rate: $PRUNING_LEARNING_RATE"
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
"$PYTHON_BIN" scripts/pruning_experiment.py \
    --checkpoint models/gru_model.pt \
    --config reports/runtime_train_config.yaml \
    --monitoring-config reports/runtime_monitoring_config.yaml \
    --levels "$PRUNING_LEVELS" \
    --fine-tune-epochs "$PRUNING_FINETUNE_EPOCHS" \
    --fine-tune-learning-rate "$PRUNING_LEARNING_RATE" \
    $WANDB_ARGS
