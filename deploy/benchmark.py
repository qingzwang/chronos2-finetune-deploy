#!/usr/bin/env python3
"""
Chronos-2 Inference Benchmark

Measures model loading time, single-sample latency, batch throughput,
concurrent request performance, and GPU memory usage.

Usage:
    python deploy/benchmark.py \
        --model_path output/chronos2-336-48-covariant-full-100k/finetuned-ckpt \
        --data_dir data/ \
        --split_csv zero-shot/data_split.csv \
        --use_covariates

    python deploy/benchmark.py \
        --model_path amazon/chronos-2 \
        --num_samples 10 --warmup_runs 2
"""

import sys
import os
import time
import json
import argparse
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "chronos-forecasting" / "src"))

from finetune.data.dataset import load_train_files, load_data_from_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Chronos-2 Inference Benchmark")
    parser.add_argument("--model_path", type=str, default="amazon/chronos-2",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--data_dir", type=str,
                        default="/home/ec2-user/efs/to_customer/zendure/data/")
    parser.add_argument("--split_csv", type=str,
                        default="/home/ec2-user/efs/to_customer/zendure/zero-shot/data_split.csv")
    parser.add_argument("--context_length", type=int, default=336)
    parser.add_argument("--prediction_length", type=int, default=48)
    parser.add_argument("--use_covariates", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of test samples to use")
    parser.add_argument("--n_windows", type=int, default=10,
                        help="Number of sliding windows per series for MAE eval")
    parser.add_argument("--stride_days", type=float, default=1,
                        help="Stride in days between sliding windows")
    parser.add_argument("--forecast_days", type=float, default=1,
                        help="Forecast horizon in days for MAE eval")
    parser.add_argument("--warmup_runs", type=int, default=5,
                        help="Warmup iterations before timing")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8,16,32,64",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--concurrency_levels", type=str, default="1,2,4,8",
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--output", type=str, default="deploy/benchmark_results.json",
                        help="Output JSON file path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run on (auto=cuda if available, else cpu)")
    return parser.parse_args()


def get_gpu_memory_mb():
    """Get current and peak GPU memory in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "peak_mb": 0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }


def compute_latency_stats(latencies):
    """Compute latency statistics from a list of times (seconds)."""
    if not latencies:
        return {}
    latencies_ms = [t * 1000 for t in latencies]
    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(sorted_ms), 2),
        "std_ms": round(statistics.stdev(sorted_ms), 2) if n > 1 else 0,
        "min_ms": round(sorted_ms[0], 2),
        "p50_ms": round(sorted_ms[n // 2], 2),
        "p90_ms": round(sorted_ms[int(n * 0.9)], 2),
        "p99_ms": round(sorted_ms[min(int(n * 0.99), n - 1)], 2),
        "max_ms": round(sorted_ms[-1], 2),
    }


def resample_and_fill(df, freq='30min'):
    """Resample to regular frequency and fill missing values (match zero-shot)."""
    df = df.copy()
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    elif 'timestamp' not in df.columns:
        return df

    df = df.sort_values('timestamp').set_index('timestamp')
    start, end = df.index.min(), df.index.max()
    regular_index = pd.date_range(start=start, end=end, freq=freq)
    df = df.reindex(regular_index)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    df[numeric_cols] = df[numeric_cols].bfill().ffill()
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    return df


def generate_sliding_windows(series_len, forecast_len, stride, n_windows, min_context):
    """Generate cutoff indices for sliding windows (from end to start)."""
    # First cutoff: leave forecast_len at the end
    first_cutoff = series_len - forecast_len
    cutoffs = []
    current = first_cutoff
    while current >= min_context:
        cutoffs.append(current)
        if n_windows is not None and len(cutoffs) >= n_windows:
            break
        current -= stride
    return cutoffs


def prepare_samples(args):
    """Load and prepare test samples (single window at end, for latency benchmarks)."""
    print(f"Loading test data from {args.split_csv} ...")
    test_files = load_train_files(args.split_csv, args.data_dir, split="test")

    if not test_files:
        raise RuntimeError("No test files found. Check --split_csv and --data_dir.")

    # Limit samples
    test_files = test_files[: args.num_samples]

    samples = []
    min_len = args.context_length + args.prediction_length

    for fp in test_files:
        try:
            df = load_data_from_csv(fp)
            if len(df) < min_len:
                continue

            # Take the last context_length points as context
            ctx_end = len(df) - args.prediction_length
            ctx_start = max(0, ctx_end - args.context_length)

            target = torch.tensor(df["power"].values[ctx_start:ctx_end], dtype=torch.float32)

            if args.use_covariates:
                temp = torch.tensor(df["temperature"].values[ctx_start:ctx_end], dtype=torch.float32)
                irr = torch.tensor(df["irradiance"].values[ctx_start:ctx_end], dtype=torch.float32)
                future_temp = torch.tensor(
                    df["temperature"].values[ctx_end:ctx_end + args.prediction_length],
                    dtype=torch.float32,
                )
                future_irr = torch.tensor(
                    df["irradiance"].values[ctx_end:ctx_end + args.prediction_length],
                    dtype=torch.float32,
                )
                sample = {
                    "target": target,
                    "past_covariates": {"temperature": temp, "irradiance": irr},
                    "future_covariates": {"temperature": future_temp, "irradiance": future_irr},
                }
            else:
                sample = target

            samples.append(sample)
        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    print(f"Prepared {len(samples)} samples (context={args.context_length}, "
          f"pred={args.prediction_length}, covariates={args.use_covariates})")
    return samples


def prepare_sliding_window_samples(args):
    """
    Load and prepare sliding window samples for MAE evaluation.
    Matches zero-shot's sliding window + resample + interpolation approach.
    """
    print(f"\nLoading test data for MAE eval (sliding windows) ...")
    test_files = load_train_files(args.split_csv, args.data_dir, split="test")

    if not test_files:
        raise RuntimeError("No test files found.")

    test_files = test_files[: args.num_samples]

    forecast_len = int(48 * args.forecast_days)
    stride = int(48 * args.stride_days)
    min_context = args.context_length

    all_samples = []
    all_ground_truths = []
    total_windows = 0

    for fp in test_files:
        try:
            df = load_data_from_csv(fp)
            df = resample_and_fill(df, freq='30min')

            min_len = min_context + forecast_len
            if len(df) < min_len:
                continue

            cutoffs = generate_sliding_windows(
                len(df), forecast_len, stride, args.n_windows, min_context
            )

            for cutoff_idx in cutoffs:
                ctx_start = max(0, cutoff_idx - args.context_length)
                ctx_end = cutoff_idx
                fc_end = cutoff_idx + forecast_len

                if fc_end > len(df):
                    continue

                target = torch.tensor(
                    df["power"].values[ctx_start:ctx_end], dtype=torch.float32
                )
                gt = torch.tensor(
                    df["power"].values[ctx_end:fc_end], dtype=torch.float32
                )

                if args.use_covariates:
                    temp = torch.tensor(df["temperature"].values[ctx_start:ctx_end], dtype=torch.float32)
                    irr = torch.tensor(df["irradiance"].values[ctx_start:ctx_end], dtype=torch.float32)
                    future_temp = torch.tensor(
                        df["temperature"].values[ctx_end:fc_end], dtype=torch.float32
                    )
                    future_irr = torch.tensor(
                        df["irradiance"].values[ctx_end:fc_end], dtype=torch.float32
                    )
                    sample = {
                        "target": target,
                        "past_covariates": {"temperature": temp, "irradiance": irr},
                        "future_covariates": {"temperature": future_temp, "irradiance": future_irr},
                    }
                else:
                    sample = target

                all_samples.append(sample)
                all_ground_truths.append(gt)
                total_windows += 1

        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    print(f"Prepared {total_windows} sliding window samples "
          f"(n_windows={args.n_windows}, stride={args.stride_days}d, "
          f"forecast={args.forecast_days}d)")
    return all_samples, all_ground_truths


def load_model(args):
    """Load model and return pipeline + load time."""
    from chronos import BaseChronosPipeline

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    print(f"Loading model: {args.model_path} (dtype={args.dtype}, device={device})")
    t0 = time.perf_counter()
    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=dtype,
    )
    load_time = time.perf_counter() - t0

    mem = get_gpu_memory_mb()
    print(f"Model loaded in {load_time:.2f}s | GPU memory: {mem['allocated_mb']} MB")
    return pipeline, load_time, mem


def run_single_predict(pipeline, sample, prediction_length):
    """Run a single prediction and return elapsed time in seconds."""
    inputs = [sample]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    pipeline.predict(inputs, prediction_length=prediction_length)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - t0


def benchmark_mae(pipeline, mae_samples, mae_ground_truths, prediction_length):
    """
    Compute MAE (zero-shot accuracy) using sliding windows.
    Matches zero-shot evaluation methodology.
    """
    print("\n--- Zero-Shot MAE (Sliding Windows) ---")

    if not mae_samples:
        print("  No MAE samples available, skipping.")
        return {}

    # Find the median quantile index
    quantile_levels = pipeline.quantiles
    if 0.5 in quantile_levels:
        median_idx = quantile_levels.index(0.5)
    else:
        median_idx = len(quantile_levels) // 2
        print(f"  Warning: 0.5 not in quantiles {quantile_levels}, using index {median_idx}")

    maes = []
    per_step_abs_errors = []

    for i, (sample, gt) in enumerate(zip(mae_samples, mae_ground_truths)):
        preds = pipeline.predict([sample], prediction_length=prediction_length)
        # preds[0] shape: (n_variates, n_quantiles, prediction_length)
        median_pred = preds[0][0, median_idx, :].cpu().float()
        gt_cpu = gt.cpu().float()

        # Align lengths (in case of mismatch)
        min_len = min(len(median_pred), len(gt_cpu))
        abs_error = (median_pred[:min_len] - gt_cpu[:min_len]).abs()
        mae_val = abs_error.mean().item()
        maes.append(mae_val)
        per_step_abs_errors.append(abs_error)

    per_step_mae = torch.stack(per_step_abs_errors).mean(dim=0)

    overall_mae = statistics.mean(maes)
    results = {
        "num_windows": len(maes),
        "overall_mae": round(overall_mae, 4),
        "std_mae": round(statistics.stdev(maes), 4) if len(maes) > 1 else 0,
        "min_mae": round(min(maes), 4),
        "max_mae": round(max(maes), 4),
        "median_mae": round(sorted(maes)[len(maes) // 2], 4),
        "per_step_mae": [round(v, 4) for v in per_step_mae.tolist()],
    }

    print(f"  Windows:    {results['num_windows']}")
    print(f"  Overall MAE:{results['overall_mae']:.4f}")
    print(f"  Std MAE:    {results['std_mae']:.4f}")
    print(f"  Min/Max MAE:{results['min_mae']:.4f} / {results['max_mae']:.4f}")
    print(f"  Median MAE: {results['median_mae']:.4f}")
    print(f"  Step MAE (first 5): {results['per_step_mae'][:5]}")
    print(f"  Step MAE (last 5):  {results['per_step_mae'][-5:]}")

    return results


def benchmark_single_latency(pipeline, samples, prediction_length, warmup_runs):
    """Benchmark single-sample inference latency."""
    print("\n--- Single Sample Latency ---")

    # Warmup
    print(f"  Warmup ({warmup_runs} runs) ...")
    for i in range(warmup_runs):
        run_single_predict(pipeline, samples[i % len(samples)], prediction_length)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    latencies = []
    for i, sample in enumerate(samples):
        elapsed = run_single_predict(pipeline, sample, prediction_length)
        latencies.append(elapsed)

    stats = compute_latency_stats(latencies)
    mem = get_gpu_memory_mb()
    stats["gpu_peak_mb"] = mem["peak_mb"]

    print(f"  Samples: {stats['count']}")
    print(f"  Mean:    {stats['mean_ms']:.2f} ms")
    print(f"  P50:     {stats['p50_ms']:.2f} ms")
    print(f"  P90:     {stats['p90_ms']:.2f} ms")
    print(f"  P99:     {stats['p99_ms']:.2f} ms")
    print(f"  Min/Max: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
    print(f"  GPU peak: {mem['peak_mb']} MB")
    return stats


def benchmark_batch_throughput(pipeline, samples, prediction_length, batch_sizes):
    """Benchmark batch inference at different batch sizes."""
    print("\n--- Batch Throughput ---")
    results = {}

    for bs in batch_sizes:
        if bs > len(samples):
            print(f"  Batch {bs}: skipped (only {len(samples)} samples)")
            continue

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        batch = samples[:bs]
        # Warmup
        try:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            pipeline.predict(batch, prediction_length=prediction_length)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch {bs}: OOM, skipping")
                torch.cuda.empty_cache()
                results[bs] = {"error": "OOM"}
                continue
            raise

        # Timed runs — process all samples in batches
        latencies = []
        n_batches = max(1, len(samples) // bs)
        for i in range(n_batches):
            batch = samples[i * bs: (i + 1) * bs]
            if not batch:
                break
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            pipeline.predict(batch, prediction_length=prediction_length)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append(time.perf_counter() - t0)

        total_time = sum(latencies)
        total_samples = n_batches * bs
        throughput = total_samples / total_time if total_time > 0 else 0
        mem = get_gpu_memory_mb()

        batch_stats = compute_latency_stats(latencies)
        batch_stats["throughput_samples_per_sec"] = round(throughput, 2)
        batch_stats["gpu_peak_mb"] = mem["peak_mb"]

        print(f"  Batch {bs:>3d}: {batch_stats['mean_ms']:>8.2f} ms/batch | "
              f"{throughput:>6.1f} samples/s | GPU peak: {mem['peak_mb']} MB")
        results[bs] = batch_stats

    return results


def benchmark_concurrency(pipeline, samples, prediction_length, concurrency_levels):
    """Benchmark concurrent inference requests using thread pool."""
    print("\n--- Concurrency ---")
    results = {}

    for n_threads in concurrency_levels:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        # Each thread sends single-sample requests
        n_requests = min(len(samples), max(n_threads * 4, 20))
        request_samples = [samples[i % len(samples)] for i in range(n_requests)]

        def do_predict(sample):
            t0 = time.perf_counter()
            pipeline.predict([sample], prediction_length=prediction_length)
            return time.perf_counter() - t0

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall_start = time.perf_counter()

        per_request_latencies = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(do_predict, s) for s in request_samples]
            for f in as_completed(futures):
                try:
                    per_request_latencies.append(f.result())
                except Exception as e:
                    print(f"    Thread error: {e}")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall_time = time.perf_counter() - wall_start

        qps = len(per_request_latencies) / wall_time if wall_time > 0 else 0
        mem = get_gpu_memory_mb()

        stats = compute_latency_stats(per_request_latencies)
        stats["wall_time_sec"] = round(wall_time, 3)
        stats["qps"] = round(qps, 2)
        stats["total_requests"] = len(per_request_latencies)
        stats["gpu_peak_mb"] = mem["peak_mb"]

        print(f"  Threads {n_threads:>2d}: QPS={qps:>6.2f} | "
              f"mean={stats['mean_ms']:>8.2f} ms | p99={stats['p99_ms']:>8.2f} ms | "
              f"wall={wall_time:.2f}s | GPU peak: {mem['peak_mb']} MB")
        results[n_threads] = stats

    return results


def main():
    args = parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",")]

    print("=" * 70)
    print("Chronos-2 Inference Benchmark")
    print("=" * 70)
    print(f"  Model:            {args.model_path}")
    print(f"  Context length:   {args.context_length}")
    print(f"  Prediction length:{args.prediction_length}")
    print(f"  Covariates:       {args.use_covariates}")
    if args.device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = args.device
    print(f"  Dtype:            {args.dtype}")
    print(f"  Device:           {actual_device}")
    if actual_device == "cuda" and torch.cuda.is_available():
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Num samples:      {args.num_samples}")
    print(f"  MAE windows:      {args.n_windows} per series")
    print(f"  MAE stride:       {args.stride_days} days")
    print(f"  MAE forecast:     {args.forecast_days} days ({int(48 * args.forecast_days)} steps)")
    print(f"  Batch sizes:      {batch_sizes}")
    print(f"  Concurrency:      {concurrency_levels}")
    print("=" * 70)

    # 1. Load model
    pipeline, load_time, load_mem = load_model(args)

    # 2. Prepare data
    samples = prepare_samples(args)
    if not samples:
        print("ERROR: No valid samples. Exiting.")
        sys.exit(1)

    # Prepare sliding window samples for MAE eval (matching zero-shot methodology)
    mae_samples, mae_ground_truths = prepare_sliding_window_samples(args)

    all_results = {
        "config": {
            "model_path": args.model_path,
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "use_covariates": args.use_covariates,
            "dtype": args.dtype,
            "device": actual_device,
            "gpu_name": torch.cuda.get_device_name(0) if actual_device == "cuda" and torch.cuda.is_available() else None,
            "num_samples": len(samples),
            "mae_n_windows": args.n_windows,
            "mae_stride_days": args.stride_days,
            "mae_forecast_days": args.forecast_days,
            "mae_total_windows": len(mae_samples),
        },
        "model_load": {
            "time_sec": round(load_time, 3),
            "gpu_mb": load_mem,
        },
    }

    # 3. Zero-shot MAE (sliding windows, matching zero-shot eval)
    mae_prediction_length = int(48 * args.forecast_days)
    all_results["zero_shot_mae"] = benchmark_mae(
        pipeline, mae_samples, mae_ground_truths, mae_prediction_length
    )

    # 4. Single latency
    all_results["single_latency"] = benchmark_single_latency(
        pipeline, samples, args.prediction_length, args.warmup_runs
    )

    # 5. Batch throughput
    all_results["batch_throughput"] = benchmark_batch_throughput(
        pipeline, samples, args.prediction_length, batch_sizes
    )

    # 6. Concurrency
    all_results["concurrency"] = benchmark_concurrency(
        pipeline, samples, args.prediction_length, concurrency_levels
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model load time:       {load_time:.2f} s")
    mae_res = all_results["zero_shot_mae"]
    print(f"Zero-shot MAE:         {mae_res['overall_mae']:.4f} (std={mae_res['std_mae']:.4f})")
    sl = all_results["single_latency"]
    print(f"Single inference:      {sl['mean_ms']:.2f} ms mean, {sl['p99_ms']:.2f} ms p99")

    best_tp = 0
    best_bs = 0
    for bs, stats in all_results["batch_throughput"].items():
        if isinstance(stats, dict) and "throughput_samples_per_sec" in stats:
            if stats["throughput_samples_per_sec"] > best_tp:
                best_tp = stats["throughput_samples_per_sec"]
                best_bs = bs
    if best_bs:
        print(f"Best batch throughput: {best_tp:.1f} samples/s (batch_size={best_bs})")

    best_qps = 0
    best_threads = 0
    for nt, stats in all_results["concurrency"].items():
        if isinstance(stats, dict) and stats.get("qps", 0) > best_qps:
            best_qps = stats["qps"]
            best_threads = nt
    if best_threads:
        print(f"Best concurrency QPS:  {best_qps:.2f} (threads={best_threads})")

    print("=" * 70)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON
    if "batch_throughput" in all_results:
        all_results["batch_throughput"] = {
            str(k): v for k, v in all_results["batch_throughput"].items()
        }
    if "concurrency" in all_results:
        all_results["concurrency"] = {
            str(k): v for k, v in all_results["concurrency"].items()
        }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
