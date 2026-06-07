#!/usr/bin/env bash
set -eu

IMAGE="${RUST_TORCH_IMAGE:-containers/build/rust_torch.sif}"
MODEL="${MODEL:-models/gru_model_torchscript.pt}"
INPUT="${INPUT:-}"
MODE="${1:-build}"

if [ ! -f "$IMAGE" ]; then
    echo "Container image not found: $IMAGE" >&2
    echo "Build it with: sbatch scripts/build_rust_torch_container.sh" >&2
    exit 1
fi

run_in_container() {
    singularity exec --nv "$IMAGE" rust-torch-env "$@"
}

case "$MODE" in
    build)
        run_in_container bash -lc 'cd rust_inference && cargo build --release'
        ;;
    run)
        if [ -z "$INPUT" ]; then
            echo "Set INPUT=/path/to/window.csv before running inference." >&2
            exit 1
        fi
        run_in_container bash -lc 'cd rust_inference && cargo build --release'
        run_in_container rust_inference/target/release/energy-gru-inference \
            --model "$MODEL" \
            --input "$INPUT"
        ;;
    shell)
        run_in_container bash
        ;;
    *)
        echo "Usage: $0 [build|run|shell]" >&2
        exit 1
        ;;
esac
