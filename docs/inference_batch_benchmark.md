# Inference Batch Benchmark

The Jenkins inference benchmark measures how batch size changes latency,
throughput, and GPU memory use while keeping the model, dataset, precision, and
GPU allocation fixed.

## Jenkins parameters

Use an inference-only AI Lab build:

- `RUN_TRAINING=false`
- `RUN_INFERENCE=true`
- `TRAIN_RUNNER=AI_LAB`
- `AI_LAB_NODES=1`
- `AI_LAB_GPUS=1`
- `BENCHMARK_INFERENCE_BATCHES=true`
- `INFERENCE_BATCH_SIZES=1,8,32,128,512,1024,2048,4096`
- `INFERENCE_BENCHMARK_REPEATS=3`

Select one saved `MODEL_VERSION`, one `DATASET_PATH`, and one
`INFERENCE_PRECISION`. Do not change those values within a batch-size
comparison.

The normal inference pass still runs first and produces predictions and RMSE.
The benchmark then loads the same TorchScript model once, warms up every batch
size, and processes the complete inference dataset for every repeat.

## Definitions

- Latency is the end-to-end time required to complete one batch inference
  operation. It includes input transfer, model execution, and output transfer,
  but excludes model loading and request queueing.
- Throughput is the number of completed predictions per second.
- Bandwidth is the maximum rate at which data can be transferred.

The report also includes amortized latency per prediction. This is the inverse
of throughput and is not the same as the service latency experienced by a
request in a batch.

## Saturation

Batch size 1 is the speedup baseline. The saturation batch size is the smallest
successful batch size that reaches at least 95% of the maximum measured
throughput. This identifies the throughput knee instead of selecting a larger
batch that adds latency and memory use for little or no throughput gain.

## Compute and memory diagnosis

The benchmark samples NVIDIA SM utilization and device-memory activity during
the timed passes. It classifies the saturated region as likely compute-bound,
likely memory-bandwidth-bound, mixed, or underutilized/overhead-bound.

This classification is a heuristic. `nvidia-smi` memory utilization measures
the percentage of time with active memory traffic, not achieved DRAM GB/s. The
report includes the GPU's theoretical memory-bandwidth ceiling when CUDA
exposes the required device properties. Use NVIDIA Nsight Compute roofline
metrics when a definitive compute-versus-memory-bandwidth conclusion is
required.

## Outputs

Jenkins prints the Markdown result and archives:

- `reports/inference_batch_benchmark.json`
- `reports/inference_batch_benchmark.csv`
- `reports/inference_batch_benchmark.md`

The JSON contains raw repeat runtimes and throughputs. W&B receives one metric
row per batch size plus summary values for saturation, speedup, latency, and
the hardware-bound diagnosis.
