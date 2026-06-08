#!/usr/bin/env bash
set -eu

IMAGE="${RUST_TORCH_DOCKER_IMAGE:-mlops-p4-rust-inference:latest}"
DOCKERFILE="${RUST_TORCH_DOCKERFILE:-containers/rust_torch.Dockerfile}"
MODEL="${MODEL:-models/gru_model_torchscript.pt}"
INPUT="${INPUT:-reports/inference_window.csv}"

if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "input CSV not found: $INPUT" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not available on this worker" >&2
    exit 1
fi

build_image() {
    docker build -f "$DOCKERFILE" -t "$IMAGE" .
}

run_inference() {
    gpu_args=""
    if [ "${DOCKER_USE_GPUS:-auto}" != "false" ]; then
        if docker run --rm --gpus all "$IMAGE" true >/dev/null 2>&1; then
            gpu_args="--gpus all"
        elif [ "${DOCKER_USE_GPUS:-auto}" = "true" ]; then
            echo "Docker GPU runtime was requested but is not available." >&2
            exit 1
        fi
    fi

    # shellcheck disable=SC2086
    docker run --rm $gpu_args \
        -e MODEL="$MODEL" \
        -e INPUT="$INPUT" \
        -v "$PWD:/workspace" \
        -w /workspace \
        "$IMAGE" \
        rust-torch-env bash -lc '
            set -eu
            cd rust_inference
            cargo build --release
            target/release/energy-gru-inference \
                --model "../$MODEL" \
                --input "../$INPUT"
        '
}

case "${1:-run}" in
    build)
        build_image
        ;;
    run)
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            build_image
        fi
        run_inference
        ;;
    *)
        echo "Usage: $0 [build|run]" >&2
        exit 2
        ;;
esac
