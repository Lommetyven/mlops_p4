# GRU Pruning Experiment

The Jenkins pruning experiment measures how global unstructured
L1-magnitude pruning changes regression quality before and after
mask-preserving fine-tuning.

## Jenkins configuration

Use these parameters for a saved-model experiment:

- `RUN_TRAINING=false`
- `RUN_INFERENCE=false`
- `RUN_PRUNING_EXPERIMENT=true`
- `RUN_DVC_REPRO=false`
- `SAVE_MODEL=false`
- `PUSH_DVC=false`
- `UPLOAD_READABLE_ARTIFACTS=false`
- `TRAIN_RUNNER=AI_LAB`
- `MODEL_VERSION=<the baseline model version>`
- `PRUNING_LEVELS=0,10,25,50,75,90`
- `PRUNING_FINETUNE_EPOCHS=5`
- `PRUNING_LEARNING_RATE=0.0001`
- `AI_LAB_CPUS=8`
- `AI_LAB_TIME_LIMIT=04:00:00`

The pruning stage always requests one AI Lab node and one GPU. The training
GPU parameters do not change this allocation.

## Method

Every pruning level starts from the same native PyTorch checkpoint. The
experiment globally ranks the absolute values of
`input_to_gru_weights` and `hidden_to_hidden_weights`, then masks the
lowest-ranked individual weights. Biases are not pruned.

This is unstructured pruning because it removes individual weights rather
than complete GRU neurons, channels, rows, or columns.

Each level is evaluated immediately after pruning. It is then fine-tuned
with the pruning masks active, which prevents masked weights from regrowing.
The state with the lowest validation loss is evaluated on the test split.
The deterministic split and random seed from the runtime training
configuration are reused for every level.

The 0% control also receives the same fine-tuning budget. Consequently, a
difference between its immediate and fine-tuned scores measures the effect
of additional training without pruning.

## Metrics and artifacts

All variants are evaluated on the same test subset. The regression metrics
are:

- RMSE, where lower is better
- MAE, where lower is better
- R2, where higher is better

Jenkins archives:

- `reports/pruning_experiment.json`
- `reports/pruning_experiment.csv`
- `reports/pruning_experiment.md`
- `reports/pruning_metrics.png`
- `reports/pruning-slurm-*.out`
- `reports/pruning-slurm-*.err`

The PNG contains RMSE, MAE, and R2 panels. Each panel contains one curve for
quality immediately after pruning and one curve for quality after
fine-tuning. The same measurements and files are logged to a dedicated W&B
pruning run.

## Interpretation limitation

Unstructured pruning creates zero-valued weights, but the normal dense
PyTorch checkpoint still stores those values and the normal dense kernels
still process the full tensor shapes. The measured sparsity is therefore a
compression opportunity, not evidence of smaller files or faster
inference. A sparse storage format and compatible sparse kernels are needed
to realize those benefits.
