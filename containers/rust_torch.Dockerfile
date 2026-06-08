FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL org.mlops_p4.project="mlops_p4"
LABEL org.mlops_p4.role="rust-inference"
LABEL org.mlops_p4.repository="Lommetyven/mlops_p4"

ENV LIBTORCH_USE_PYTORCH=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libssl-dev \
        pkg-config \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal \
    && /root/.cargo/bin/rustup default stable \
    && /root/.cargo/bin/cargo --version \
    && python -c "import torch; print('torch', torch.__version__)"

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -eu' \
    'export PATH="/root/.cargo/bin:${PATH}"' \
    'export LIBTORCH_USE_PYTORCH=1' \
    'TORCH_LIB="$(python -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / '\''lib'\'')")"' \
    'export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"' \
    'exec "$@"' \
    > /usr/local/bin/rust-torch-env \
    && chmod +x /usr/local/bin/rust-torch-env

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

CMD ["rust-torch-env", "cargo", "--version"]
