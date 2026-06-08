#!/usr/bin/env bash
set -euo pipefail

IMAGE="${RUST_TORCH_DOCKER_IMAGE:-mlops-p4-rust-inference:latest}"
DOCKERFILE="${RUST_TORCH_DOCKERFILE:-containers/rust_torch.Dockerfile}"
MODEL="${MODEL:-models/gru_model_torchscript.pt}"
INPUT="${INPUT:-reports/inference_window.csv}"
REPORT_DIR="${REPORT_DIR:-reports}"
BUILD_LOG="${DOCKER_BUILD_LOG:-$REPORT_DIR/docker_rust_inference_build.txt}"
METADATA_FILE="${DOCKER_METADATA_FILE:-$REPORT_DIR/docker_rust_inference_metadata.txt}"
FINGERPRINT_LABEL="org.mlops_p4.rust_torch.dockerfile_sha"

mkdir -p "$REPORT_DIR"

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

dockerfile_fingerprint() {
    sha256sum "$DOCKERFILE" | awk '{print $1}'
}

image_fingerprint() {
    docker image inspect \
        -f "{{ index .Config.Labels \"$FINGERPRINT_LABEL\" }}" \
        "$IMAGE" 2>/dev/null || true
}

image_exists() {
    docker image inspect "$IMAGE" >/dev/null 2>&1
}

write_metadata() {
    {
        echo "image=$IMAGE"
        echo "dockerfile=$DOCKERFILE"
        echo "dockerfile_sha=$(dockerfile_fingerprint)"
        echo "image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
        echo "image_created=$(docker image inspect -f '{{.Created}}' "$IMAGE" 2>/dev/null || true)"
        echo "image_size_bytes=$(docker image inspect -f '{{.Size}}' "$IMAGE" 2>/dev/null || true)"
        echo "gpu_args=${gpu_args:-}"
        echo "build_action=${build_action:-unknown}"
        echo "build_seconds=${build_seconds:-0}"
        echo "run_seconds=${run_seconds:-0}"
    } > "$METADATA_FILE"
}

build_image() {
    build_action="built"
    build_start="$(date +%s)"
    echo "Building Docker image $IMAGE from $DOCKERFILE" >&2
    if ! docker build \
        -f "$DOCKERFILE" \
        --label "$FINGERPRINT_LABEL=$(dockerfile_fingerprint)" \
        -t "$IMAGE" \
        . > "$BUILD_LOG" 2>&1; then
        cat "$BUILD_LOG" >&2
        exit 1
    fi
    build_seconds="$(( $(date +%s) - build_start ))"
    echo "Docker image build finished in ${build_seconds}s" >&2
}

ensure_image() {
    build_action="reused"
    build_seconds=0
    expected_fingerprint="$(dockerfile_fingerprint)"
    existing_fingerprint="$(image_fingerprint)"

    if ! image_exists; then
        build_image
    elif [ "$existing_fingerprint" != "$expected_fingerprint" ]; then
        echo "Docker image fingerprint changed; rebuilding $IMAGE" >&2
        build_image
    else
        echo "Reusing existing Docker image $IMAGE" >&2
        : > "$BUILD_LOG"
    fi
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

    run_start="$(date +%s)"
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
    run_seconds="$(( $(date +%s) - run_start ))"
    write_metadata
}

case "${1:-run}" in
    build)
        build_image
        gpu_args=""
        run_seconds=0
        write_metadata
        ;;
    run)
        ensure_image
        run_inference
        ;;
    *)
        echo "Usage: $0 [build|run]" >&2
        exit 2
        ;;
esac
