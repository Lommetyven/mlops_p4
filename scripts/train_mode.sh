#!/usr/bin/env bash
#SBATCH --job-name=energy-gru
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.err

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load python || true
fi

if [ -x .venv/bin/python ]; then
    PYTHON_BIN=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

"$PYTHON_BIN" -m dvc pull -r "${DVC_REMOTE:-minio}"

if [ -f data/dvc_archives/raw.tar.gz ]; then
    "$PYTHON_BIN" scripts/archive_paths.py unpack \
        --archive data/dvc_archives/raw.tar.gz \
        --output data
fi

"$PYTHON_BIN" main.py
