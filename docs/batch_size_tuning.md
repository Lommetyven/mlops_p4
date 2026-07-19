# AI Lab Batch-Size Tuning

The Jenkins `MAXIMIZE_GPU_UTIL` parameter enables a batch-size benchmark before
AI Lab training. It does not affect the `DAKI_WORKER` training path.

The tuner runs inside the same Slurm allocation and with the same number of DDP
processes as full training. Each candidate is a per-GPU batch size. The report
therefore records both the per-GPU value and the effective global batch size:

```text
global batch size = per-GPU batch size * DDP world size
```

For each candidate, the tuner performs warm-up steps followed by timed forward,
backward, and optimizer steps using the processed training data. It selects the
successful candidate with the highest aggregate samples per second across all
selected GPUs. Candidates are rejected when they exceed 90% peak reserved GPU
memory or raise a CUDA out-of-memory error.

The search is also limited to batch sizes that retain at least ten optimizer
steps per epoch. This prevents a throughput-only result from collapsing an epoch
into one or two oversized updates.

The selected value updates only the generated runtime configuration. The
checked-in `configs/train_config.yaml` is not modified. Full training then starts
in the same Slurm job using that runtime configuration.

Jenkins archives the benchmark details from:

```text
reports/batch_size_tuning.json
```

The checkbox defaults to disabled, preserving the existing training path. The
initial `BATCH_SIZE` parameter remains the fallback when tuning is disabled and
is included as a candidate when it falls inside the safe search range.
