#!/usr/bin/env bash
set -eu

IMAGE="${RUST_TORCH_DOCKER_IMAGE:-172.24.198.42:5000/mlops-p4/rust-inference:latest}"
PROJECT_LABEL="${PROJECT_LABEL:-org.mlops_p4.project=mlops_p4}"
REGISTRY="${DAKI_DOCKER_REGISTRY:-172.24.198.42:5000}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not available on this worker" >&2
    exit 1
fi

echo "Local mlops_p4 images:"
docker image ls --filter "label=$PROJECT_LABEL"

echo
echo "Local mlops_p4 containers:"
docker ps -a --filter "label=$PROJECT_LABEL"

echo
echo "Configured Rust inference image:"
echo "$IMAGE"

echo
echo "Registry catalog, if accessible:"
if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://$REGISTRY/v2/_catalog" || true
    echo
    curl -fsS "http://$REGISTRY/v2/mlops-p4/rust-inference/tags/list" || true
    echo
else
    echo "curl is not available"
fi
