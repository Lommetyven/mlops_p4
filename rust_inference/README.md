# Rust Inference

This CLI runs inference with the exported TorchScript GRU model.

The training checkpoint `models/gru_model.pt` is a Python PyTorch checkpoint and
cannot be loaded directly from Rust. Export it first:

```bash
python scripts/export_torchscript.py \
  --checkpoint models/gru_model.pt \
  --output models/gru_model_torchscript.pt
```

Training also exports `models/gru_model_torchscript.pt` automatically.

Run inference with a CSV containing one sequence window: one row per time step,
16 numeric feature columns per row, matching the training feature order.

```bash
cd rust_inference
cargo run --release -- \
  --model ../models/gru_model_torchscript.pt \
  --input ../tmp/window.csv
```

The binary uses the `tch` crate and requires libtorch/PyTorch native libraries
available on the machine where it is built and run.

## DAKI Worker Docker Container

The DAKI worker can run Rust inference through Docker when `cargo` is not
installed directly on the worker:

```bash
MODEL=models/gru_model_torchscript.pt \
INPUT=reports/inference_window.csv \
bash scripts/rust_inference_docker.sh run
```

The Docker image is built from `containers/rust_torch.Dockerfile` the first time
it is needed. It mounts the current workspace, builds the Rust binary, and runs
inference against the mounted TorchScript model and CSV window.

Repeated Jenkins runs reuse the existing `mlops-p4-rust-inference:latest` image
on the DAKI worker. The script rebuilds it only when the image is missing or the
Dockerfile fingerprint changes. Build and runtime details are written to:

```text
reports/docker_rust_inference_build.txt
reports/docker_rust_inference_metadata.txt
```

## AI Lab Singularity Container

AI Lab runs containers with Singularity. Build the Rust + PyTorch image from the
project root:

```bash
sbatch scripts/build_rust_torch_container.sh
```

This creates:

```text
containers/build/rust_torch.sif
```

Build the Rust binary inside the container:

```bash
bash scripts/rust_inference_container.sh build
```

Run inference inside the container:

```bash
INPUT=tmp/window.csv \
MODEL=models/gru_model_torchscript.pt \
bash scripts/rust_inference_container.sh run
```

The container definition is `containers/rust_torch.def`. It bootstraps from a
Docker PyTorch image, installs Rust, and configures `LIBTORCH_USE_PYTORCH=1` so
the `tch` crate links against the PyTorch libraries in the container.
