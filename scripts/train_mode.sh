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

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if [ -f data/dvc_archives/raw.tar.gz ]; then
    "$PYTHON_BIN" scripts/archive_paths.py unpack \
        --archive data/dvc_archives/raw.tar.gz \
        --output data
fi

"$PYTHON_BIN" main.py
