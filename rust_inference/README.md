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

Run inference with a CSV containing the feature rows for the complete dataset.
The CLI creates every overlapping sequence window, processes the windows in
batches, and prints one prediction per window. The 16 numeric feature columns
must match the training feature order.

```bash
cd rust_inference
cargo run --release -- \
  --model ../models/gru_model_torchscript.pt \
  --input ../reports/inference_window.csv \
  --sequence-length 12 \
  --batch-size 1024 \
  --precision fp32
```

`fp32` works on CPU or CUDA. `fp16` uses half-precision weights and inputs and
requires CUDA. `fp8` is accepted as a Jenkins/CLI capability selection but
fails before inference because the current TorchScript GRU with `tch 0.19`
does not provide an FP8 GRU kernel. It is not silently replaced with FP16.

The binary uses the `tch` crate and requires libtorch/PyTorch native libraries
available on the machine where it is built and run.

## Jenkins Runners And W&B

Jenkins routes inference through the selected `TRAIN_RUNNER`:

- `AI_LAB` submits a separate one-GPU Slurm job and runs the TorchScript model
  with Python/PyTorch. This is the normal route for CUDA FP16 inference.
- `DAKI_WORKER` runs this Rust `tch` binary. FP32 can use its CPU fallback;
  FP16 requires Docker GPU passthrough on that worker.

Both paths write `reports/inference_metrics.json` and create a separate W&B
inference run. The W&B run logs the runner, backend, device, model version,
precision, batch size, sequence length, runtime, window count, throughput, and
available GPU memory metrics. Prediction values remain in
`reports/inference_predictions.txt` as a Jenkins artifact.

## DAKI Worker Docker Container

The DAKI worker can run Rust inference through Docker when `cargo` is not
installed directly on the worker:

```bash
MODEL=models/gru_model_torchscript.pt \
INPUT=reports/inference_window.csv \
INFERENCE_PRECISION=FP16 \
INFERENCE_SEQUENCE_LENGTH=12 \
INFERENCE_BATCH_SIZE=1024 \
bash scripts/rust_inference_docker.sh run
```

The Docker image is pulled from the DAKI registry when available:

```text
172.24.198.42:5000/mlops-p4/rust-inference:latest
```

If the image is missing or its Dockerfile fingerprint is stale, it is rebuilt
from `containers/rust_torch.Dockerfile` and pushed back to the registry. It
mounts the current workspace read-only, builds the Rust binary in the
`mlops-p4-rust-target` Docker volume, and runs inference against the mounted
TorchScript model and CSV window. Set `CARGO_TARGET_VOLUME` to override the
volume name.

Repeated Jenkins runs reuse the existing
`172.24.198.42:5000/mlops-p4/rust-inference:latest` image locally or from the
registry. The script rebuilds it only when the image is
missing or the Dockerfile fingerprint changes. Build and runtime details are
written to:

```text
reports/docker_rust_inference_build.txt
reports/docker_rust_inference_metadata.txt
```

To list only this project's local Docker images and containers on a DAKI worker:

```bash
bash scripts/list_daki_docker_resources.sh
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
INFERENCE_PRECISION=FP32 \
INFERENCE_SEQUENCE_LENGTH=12 \
INFERENCE_BATCH_SIZE=1024 \
bash scripts/rust_inference_container.sh run
```

The container definition is `containers/rust_torch.def`. It bootstraps from a
Docker PyTorch image, installs Rust, and configures `LIBTORCH_USE_PYTORCH=1` so
the `tch` crate links against the PyTorch libraries in the container.
