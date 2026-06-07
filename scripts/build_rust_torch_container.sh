#!/usr/bin/env bash
#SBATCH --job-name=build_rust_torch
#SBATCH --output=reports/build-rust-torch-%j.out
#SBATCH --error=reports/build-rust-torch-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ceph/home/student.aau.dk/sl38ze/MLOps/mlops_p4

set -eu

export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$HOME/.singularity/tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.singularity/cache}"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" reports containers/build

INPUT_DEF="${INPUT_DEF:-containers/rust_torch.def}"
OUTPUT_SIF="${OUTPUT_SIF:-containers/build/rust_torch.sif}"

echo "Building Singularity image"
echo "Definition: $INPUT_DEF"
echo "Output: $OUTPUT_SIF"

singularity build --fakeroot "$OUTPUT_SIF" "$INPUT_DEF"

echo "Built $OUTPUT_SIF"
singularity exec "$OUTPUT_SIF" rust-torch-env rustc --version
singularity exec "$OUTPUT_SIF" rust-torch-env cargo --version
singularity exec "$OUTPUT_SIF" rust-torch-env python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY
