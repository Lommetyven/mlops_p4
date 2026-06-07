#!/usr/bin/env bash
#SBATCH --job-name=energy-gru
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.err

set -eu

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load python || true
fi

echo "Running on host: $(hostname)"
echo "Repository root: $REPO_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-not set}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-not set}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-not set}"
echo "SLURM_NODEID: ${SLURM_NODEID:-not set}"

if command -v python3.13 >/dev/null 2>&1; then
    BASE_PYTHON=python3.13
elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON=python3
else
    BASE_PYTHON=python
fi

if [ ! -x .venv/bin/python ] || [ ! -x .venv/bin/dvc ]; then
    echo "Preparing .venv on Slurm node..."
    "$BASE_PYTHON" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

PYTHON_BIN=.venv/bin/python
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/train_config.yaml}"
TRAIN_DISTRIBUTED="${TRAIN_DISTRIBUTED:-true}"
TORCHRUN_NNODES="${TORCHRUN_NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHON_BIN TRAIN_CONFIG_PATH TRAIN_DISTRIBUTED TORCHRUN_NNODES MASTER_PORT

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Training config: $TRAIN_CONFIG_PATH"

if [ -f data/dvc_archives/raw.tar.gz ]; then
    "$PYTHON_BIN" scripts/archive_paths.py unpack \
        --archive data/dvc_archives/raw.tar.gz \
        --output data
fi

detect_cuda_device_count() {
    "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

resolve_processes_per_node() {
    if [ -n "${TORCHRUN_NPROC_PER_NODE:-}" ]; then
        echo "$TORCHRUN_NPROC_PER_NODE"
    elif [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "$SLURM_GPUS_ON_NODE" -eq "$SLURM_GPUS_ON_NODE" ] 2>/dev/null; then
        NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
        echo "$NPROC_PER_NODE"
    else
        detect_cuda_device_count
    fi
}

CUDA_DEVICE_COUNT="$(detect_cuda_device_count)"

if [ "$TRAIN_DISTRIBUTED" = "true" ] && [ "$TORCHRUN_NNODES" -gt 1 ]; then
    NPROC_PER_NODE="$(resolve_processes_per_node)"
    if ! command -v srun >/dev/null 2>&1; then
        echo "srun is required for multi-node torchrun." >&2
        exit 1
    fi
    if ! command -v scontrol >/dev/null 2>&1; then
        echo "scontrol is required to resolve the torchrun rendezvous host." >&2
        exit 1
    fi

    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    export MASTER_ADDR MASTER_PORT NPROC_PER_NODE TORCHRUN_NNODES
    echo "Launching multi-node torchrun: nnodes=$TORCHRUN_NNODES nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"
    srun \
        --nodes "$TORCHRUN_NNODES" \
        --ntasks "$TORCHRUN_NNODES" \
        --ntasks-per-node 1 \
        bash -lc '"$PYTHON_BIN" -m torch.distributed.run \
            --nnodes "$TORCHRUN_NNODES" \
            --nproc-per-node "$NPROC_PER_NODE" \
            --node-rank "$SLURM_NODEID" \
            --rdzv-backend c10d \
            --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
            main.py --config "$TRAIN_CONFIG_PATH"'
elif [ "$TRAIN_DISTRIBUTED" = "true" ] && [ "$CUDA_DEVICE_COUNT" -gt 1 ]; then
    NPROC_PER_NODE="$(resolve_processes_per_node)"
    echo "Launching DistributedDataParallel with $NPROC_PER_NODE processes"
    "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nproc-per-node "$NPROC_PER_NODE" \
        main.py --config "$TRAIN_CONFIG_PATH"
else
    echo "Launching single-process training"
    "$PYTHON_BIN" main.py --config "$TRAIN_CONFIG_PATH"
fi
