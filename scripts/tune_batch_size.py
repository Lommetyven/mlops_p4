import argparse
import gc
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from main import (
    autocast_dtype,
    autocast_enabled,
    build_criterion,
    build_dataloaders,
    build_optimizer,
    cleanup_distributed,
    load_train_config,
    maybe_distribute_model,
    normalize_precision,
    resolve_device,
    scaler_enabled,
    seed_everything,
    setup_distributed_if_requested,
)
from train.gru_model import GruModel


def build_candidate_batch_sizes(configured, minimum, maximum):
    configured = int(configured)
    minimum = max(1, int(minimum))
    maximum = max(1, int(maximum))
    lower_bound = min(minimum, maximum)

    candidates = set()
    value = 1
    while value < lower_bound:
        value *= 2
    while value <= maximum:
        candidates.add(value)
        value *= 2

    candidates.add(min(max(configured, lower_bound), maximum))
    candidates.add(maximum)
    return sorted(candidates)


def select_best_result(results):
    successful = [result for result in results if result["status"] == "ok"]
    if not successful:
        return None
    return max(
        successful,
        key=lambda result: (
            result["aggregate_samples_per_second"],
            -result["batch_size_per_gpu"],
        ),
    )


def update_runtime_config_batch_size(config_path, batch_size):
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    config.setdefault("training", {})["batch_size"] = int(batch_size)

    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_report(report_path, report):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _next_batch(data_loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(data_loader)
        batch = next(iterator)
    return batch, iterator


def _run_step(
    model,
    features,
    targets,
    criterion,
    optimizer,
    device,
    precision,
    amp_enabled,
    scaler,
    gradient_clip_norm,
):
    features = features.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype(precision),
        enabled=autocast_enabled(device, precision, amp_enabled),
    ):
        predictions = model(features)
        loss = criterion(predictions, targets)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

    return features.size(0)


def _build_benchmark_components(config, distributed_context, device, distributed):
    benchmark_config = deepcopy(config)
    train_loader, _, _, _, _ = build_dataloaders(
        benchmark_config,
        distributed_context=distributed_context,
    )
    model = GruModel(**benchmark_config["model"]).to(device)
    if distributed:
        model = maybe_distribute_model(model, device, distributed_context)
    criterion = build_criterion(benchmark_config["training"]["loss"])
    optimizer = build_optimizer(model, benchmark_config["training"])
    precision = normalize_precision(
        benchmark_config["training"].get("precision", "float32")
    )
    amp_enabled = bool(benchmark_config["training"].get("amp_enabled", False))
    scaler = torch.cuda.amp.GradScaler(
        enabled=scaler_enabled(device, precision, amp_enabled)
    )
    return train_loader, model, criterion, optimizer, precision, amp_enabled, scaler


def _is_cuda_oom(error):
    return (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )


def _clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def _probe_local_batch_size(config, distributed_context, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fits = True
    try:
        (
            train_loader,
            model,
            criterion,
            optimizer,
            precision,
            amp_enabled,
            scaler,
        ) = _build_benchmark_components(
            config,
            distributed_context,
            device,
            distributed=False,
        )
        batch, _ = _next_batch(train_loader, iter(train_loader))
        _run_step(
            model=model,
            features=batch[0],
            targets=batch[1],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            precision=precision,
            amp_enabled=amp_enabled,
            scaler=scaler,
            gradient_clip_norm=config["training"].get("gradient_clip_norm"),
        )
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        if not _is_cuda_oom(error):
            raise
        fits = False

    peak_memory_bytes = torch.cuda.max_memory_reserved(device)
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    memory_fraction = peak_memory_bytes / max(total_memory_bytes, 1)
    return fits, memory_fraction


def probe_batch_size(config, distributed_context, device):
    fits, memory_fraction = _probe_local_batch_size(
        config,
        distributed_context,
        device,
    )
    _clear_cuda()

    status = torch.tensor(1 if fits else 0, dtype=torch.int32, device=device)
    memory = torch.tensor(memory_fraction, dtype=torch.float64, device=device)
    if distributed_context.get("distributed"):
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    return bool(status.item()), float(memory.item())


def benchmark_batch_size(
    config,
    distributed_context,
    device,
    warmup_steps,
    benchmark_steps,
):
    torch.cuda.empty_cache()
    (
        train_loader,
        model,
        criterion,
        optimizer,
        precision,
        amp_enabled,
        scaler,
    ) = _build_benchmark_components(
        config,
        distributed_context,
        device,
        distributed=distributed_context.get("distributed", False),
    )
    iterator = iter(train_loader)
    gradient_clip_norm = config["training"].get("gradient_clip_norm")

    for _ in range(warmup_steps):
        batch, iterator = _next_batch(train_loader, iterator)
        _run_step(
            model,
            batch[0],
            batch[1],
            criterion,
            optimizer,
            device,
            precision,
            amp_enabled,
            scaler,
            gradient_clip_norm,
        )

    if distributed_context.get("distributed"):
        dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    local_samples = 0

    for _ in range(benchmark_steps):
        batch, iterator = _next_batch(train_loader, iterator)
        local_samples += _run_step(
            model,
            batch[0],
            batch[1],
            criterion,
            optimizer,
            device,
            precision,
            amp_enabled,
            scaler,
            gradient_clip_norm,
        )

    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    peak_memory_bytes = torch.cuda.max_memory_reserved(device)
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory

    elapsed = torch.tensor(elapsed_seconds, dtype=torch.float64, device=device)
    samples = torch.tensor(local_samples, dtype=torch.float64, device=device)
    memory = torch.tensor(
        peak_memory_bytes / max(total_memory_bytes, 1),
        dtype=torch.float64,
        device=device,
    )
    if distributed_context.get("distributed"):
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    result = {
        "elapsed_seconds": float(elapsed.item()),
        "aggregate_samples": int(samples.item()),
        "aggregate_samples_per_second": float(samples.item() / elapsed.item()),
        "peak_memory_fraction": float(memory.item()),
    }
    _clear_cuda()
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune the per-GPU training batch size on an AI Lab allocation."
    )
    parser.add_argument(
        "--config",
        default=os.getenv("TRAIN_CONFIG_PATH", "configs/train_config.yaml"),
    )
    parser.add_argument(
        "--report",
        default=os.getenv(
            "BATCH_TUNER_REPORT",
            "reports/batch_size_tuning.json",
        ),
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=int(os.getenv("BATCH_TUNER_MIN_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.getenv("BATCH_TUNER_MAX_BATCH_SIZE", "4096")),
    )
    parser.add_argument(
        "--minimum-steps-per-epoch",
        type=int,
        default=int(os.getenv("BATCH_TUNER_MIN_STEPS_PER_EPOCH", "10")),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=int(os.getenv("BATCH_TUNER_WARMUP_STEPS", "3")),
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=int(os.getenv("BATCH_TUNER_BENCHMARK_STEPS", "10")),
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=float(os.getenv("BATCH_TUNER_MEMORY_LIMIT", "0.9")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Batch-size tuning requires an AI Lab CUDA allocation.")
    if args.minimum_steps_per_epoch < 1:
        raise ValueError("minimum-steps-per-epoch must be at least 1.")
    if args.warmup_steps < 0 or args.benchmark_steps < 1:
        raise ValueError(
            "warmup-steps must be non-negative and benchmark-steps positive."
        )
    if not 0 < args.memory_limit <= 1:
        raise ValueError("memory-limit must be between 0 and 1.")

    config = load_train_config(args.config)
    distributed_context = setup_distributed_if_requested(config["training"])
    is_main_process = distributed_context["is_main"]
    world_size = distributed_context["world_size"]
    device = resolve_device(config["training"]["device"], distributed_context)
    seed_everything(int(config["training"]["seed"]))

    try:
        _, _, _, _, split_sizes = build_dataloaders(
            config,
            distributed_context=distributed_context,
        )
        local_training_samples = math.ceil(split_sizes["train"] / world_size)
        step_limited_maximum = max(
            1,
            local_training_samples // args.minimum_steps_per_epoch,
        )
        maximum_batch_size = min(args.max_batch_size, step_limited_maximum)
        candidates = build_candidate_batch_sizes(
            configured=config["training"]["batch_size"],
            minimum=args.min_batch_size,
            maximum=maximum_batch_size,
        )
        results = []

        if is_main_process:
            print(
                "Batch-size tuning: "
                f"gpus={world_size} candidates={candidates} "
                f"memory_limit={args.memory_limit:.0%}"
            )

        for batch_size in candidates:
            candidate_config = deepcopy(config)
            candidate_config["training"]["batch_size"] = batch_size
            fits, probe_memory_fraction = probe_batch_size(
                candidate_config,
                distributed_context,
                device,
            )
            if not fits:
                result = {
                    "batch_size_per_gpu": batch_size,
                    "global_batch_size": batch_size * world_size,
                    "status": "out_of_memory",
                    "peak_memory_fraction": probe_memory_fraction,
                }
                results.append(result)
                if is_main_process:
                    print(f"Batch size {batch_size}: out of memory")
                break
            if probe_memory_fraction > args.memory_limit:
                result = {
                    "batch_size_per_gpu": batch_size,
                    "global_batch_size": batch_size * world_size,
                    "status": "memory_limit",
                    "peak_memory_fraction": probe_memory_fraction,
                }
                results.append(result)
                if is_main_process:
                    print(
                        f"Batch size {batch_size}: probe memory "
                        f"{probe_memory_fraction:.1%} exceeds limit"
                    )
                break

            metrics = benchmark_batch_size(
                candidate_config,
                distributed_context,
                device,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
            )
            status = (
                "ok"
                if metrics["peak_memory_fraction"] <= args.memory_limit
                else "memory_limit"
            )
            result = {
                "batch_size_per_gpu": batch_size,
                "global_batch_size": batch_size * world_size,
                "status": status,
                **metrics,
            }
            results.append(result)
            if is_main_process:
                print(
                    f"Batch size {batch_size}: "
                    f"{metrics['aggregate_samples_per_second']:.1f} samples/s, "
                    f"peak memory {metrics['peak_memory_fraction']:.1%}"
                )
            if status != "ok":
                break

        selected = select_best_result(results)
        selected_batch_size = selected["batch_size_per_gpu"] if selected else -1
        selected_tensor = torch.tensor(
            selected_batch_size,
            dtype=torch.int64,
            device=device,
        )
        if distributed_context.get("distributed"):
            dist.broadcast(selected_tensor, src=0)
        selected_batch_size = int(selected_tensor.item())

        if is_main_process:
            report = {
                "selection_metric": "aggregate_samples_per_second",
                "selected_batch_size_per_gpu": (
                    selected_batch_size if selected_batch_size > 0 else None
                ),
                "selected_global_batch_size": (
                    selected_batch_size * world_size
                    if selected_batch_size > 0
                    else None
                ),
                "gpu_count": world_size,
                "gpu_model": torch.cuda.get_device_name(device),
                "memory_limit_fraction": args.memory_limit,
                "minimum_steps_per_epoch": args.minimum_steps_per_epoch,
                "local_training_samples": local_training_samples,
                "warmup_steps": args.warmup_steps,
                "benchmark_steps": args.benchmark_steps,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "candidates": results,
            }
            write_report(args.report, report)
            if selected_batch_size > 0:
                update_runtime_config_batch_size(args.config, selected_batch_size)
                print(
                    "Selected per-GPU batch size "
                    f"{selected_batch_size} "
                    f"(global batch {selected_batch_size * world_size})"
                )

        if distributed_context.get("distributed"):
            dist.barrier()
        if selected_batch_size < 1:
            raise RuntimeError("No batch-size candidate completed successfully.")
    finally:
        cleanup_distributed(distributed_context)


if __name__ == "__main__":
    main()
