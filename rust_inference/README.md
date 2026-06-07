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
