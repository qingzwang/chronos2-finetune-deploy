#!/usr/bin/env python3
"""
Test script for Chronos-2 SageMaker endpoint.

Runs accuracy (MAE) and latency benchmarks against a deployed endpoint,
using the same data pipeline as benchmark.py.

Usage:
    # Basic test (univariate)
    python deploy/test_sagemaker_endpoint.py \
        --endpoint_name chronos2-finetuned-zendure

    # With covariates (matching finetuned model)
    python deploy/test_sagemaker_endpoint.py \
        --endpoint_name chronos2-finetuned-zendure \
        --use_covariates

    # Full evaluation
    python deploy/test_sagemaker_endpoint.py \
        --endpoint_name chronos2-finetuned-zendure \
        --use_covariates \
        --num_samples 50 \
        --n_windows 10
"""

import sys
import time
import json
import argparse
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Add project paths for data loading
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "chronos-forecasting" / "src"))

from finetune.data.dataset import load_train_files, load_data_from_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Test Chronos-2 SageMaker Endpoint")
    parser.add_argument("--endpoint_name", type=str, default="chronos2-finetuned-zendure")
    parser.add_argument("--data_dir", type=str,
                        default="data/")
    parser.add_argument("--split_csv", type=str,
                        default="zero-shot/data_split.csv")
    parser.add_argument("--context_length", type=int, default=336)
    parser.add_argument("--prediction_length", type=int, default=48)
    parser.add_argument("--use_covariates", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of test series to use")
    parser.add_argument("--n_windows", type=int, default=5,
                        help="Number of sliding windows per series for MAE eval")
    parser.add_argument("--stride_days", type=float, default=1,
                        help="Stride in days between sliding windows")
    parser.add_argument("--forecast_days", type=float, default=1,
                        help="Forecast horizon in days")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Number of time series per endpoint request")
    parser.add_argument("--quantile_levels", type=str, default="0.1,0.5,0.9",
                        help="Comma-separated quantile levels")
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region (default: from env/config)")
    parser.add_argument("--output", type=str, default="deploy/endpoint_test_results.json")
    return parser.parse_args()


def resample_and_fill(df, freq='30min'):
    """Resample to regular frequency and fill missing values."""
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
    """Generate cutoff indices for sliding windows."""
    first_cutoff = series_len - forecast_len
    cutoffs = []
    current = first_cutoff
    while current >= min_context:
        cutoffs.append(current)
        if n_windows is not None and len(cutoffs) >= n_windows:
            break
        current -= stride
    return cutoffs


def build_payload_entry(df, ctx_start, ctx_end, fc_end, use_covariates):
    """Build a single time series entry for the endpoint payload."""
    target = df["power"].values[ctx_start:ctx_end].tolist()
    entry = {"target": target}

    if use_covariates:
        past_temp = df["temperature"].values[ctx_start:ctx_end].tolist()
        past_irr = df["irradiance"].values[ctx_start:ctx_end].tolist()
        entry["past_covariates"] = {
            "temperature": past_temp,
            "irradiance": past_irr,
        }
        future_temp = df["temperature"].values[ctx_end:fc_end].tolist()
        future_irr = df["irradiance"].values[ctx_end:fc_end].tolist()
        entry["future_covariates"] = {
            "temperature": future_temp,
            "irradiance": future_irr,
        }

    return entry


def prepare_sliding_window_data(args):
    """
    Load test data and prepare sliding window samples.
    Returns list of (payload_entry, ground_truth) tuples.
    """
    print(f"Loading test data from {args.split_csv} ...")
    test_files = load_train_files(args.split_csv, args.data_dir, split="test")
    if not test_files:
        raise RuntimeError("No test files found.")

    test_files = test_files[:args.num_samples]

    forecast_len = int(48 * args.forecast_days)
    stride = int(48 * args.stride_days)
    min_context = args.context_length

    samples = []

    for fp in test_files:
        try:
            df = load_data_from_csv(fp)
            df = resample_and_fill(df, freq='30min')

            if len(df) < min_context + forecast_len:
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

                entry = build_payload_entry(df, ctx_start, ctx_end, fc_end, args.use_covariates)
                ground_truth = df["power"].values[ctx_end:fc_end].tolist()
                samples.append((entry, ground_truth))

        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    print(f"Prepared {len(samples)} sliding window samples")
    return samples


def prepare_single_samples(args):
    """Prepare single-window samples for latency testing."""
    print(f"Loading latency test data ...")
    test_files = load_train_files(args.split_csv, args.data_dir, split="test")
    if not test_files:
        raise RuntimeError("No test files found.")

    test_files = test_files[:args.num_samples]
    min_len = args.context_length + args.prediction_length
    entries = []

    for fp in test_files:
        try:
            df = load_data_from_csv(fp)
            if len(df) < min_len:
                continue

            ctx_end = len(df) - args.prediction_length
            ctx_start = max(0, ctx_end - args.context_length)
            fc_end = ctx_end + args.prediction_length

            entry = build_payload_entry(df, ctx_start, ctx_end, fc_end, args.use_covariates)
            entries.append(entry)

        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    print(f"Prepared {len(entries)} latency test samples")
    return entries


def invoke_endpoint(predictor, inputs, prediction_length, quantile_levels):
    """Send a batch of inputs to the endpoint and return predictions, elapsed time, and model latency.

    Returns:
        response: parsed JSON response
        elapsed: total round-trip time in seconds (network + model)
        model_latency: model inference time in seconds (from SageMaker header), or None
    """
    payload = json.dumps({
        "inputs": inputs,
        "parameters": {
            "prediction_length": prediction_length,
            "quantile_levels": quantile_levels,
        },
    })

    sm_runtime = predictor.sagemaker_session.sagemaker_runtime_client

    t0 = time.perf_counter()
    raw_response = sm_runtime.invoke_endpoint(
        EndpointName=predictor.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    elapsed = time.perf_counter() - t0

    model_latency = None

    response = json.loads(raw_response["Body"].read().decode("utf-8"))

    return response, elapsed, model_latency


def query_cloudwatch_latency(endpoint_name, region, start_time, end_time):
    """Query CloudWatch for SageMaker endpoint ModelLatency and OverheadLatency.

    ModelLatency: time spent inside the model container (inference).
    OverheadLatency: SageMaker overhead (routing, serialization, etc.).
    Both are in microseconds in CloudWatch.
    """
    import boto3
    from datetime import timedelta

    cw = boto3.client("cloudwatch", region_name=region)
    # Add buffer to capture all metrics
    start = start_time - timedelta(minutes=1)
    end = end_time + timedelta(minutes=1)

    results = {}
    for metric_name in ["ModelLatency", "OverheadLatency"]:
        resp = cw.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            StartTime=start,
            EndTime=end,
            Period=60,  # 1-minute granularity
            Statistics=["Average", "Minimum", "Maximum", "SampleCount"],
        )
        datapoints = resp.get("Datapoints", [])
        if datapoints:
            # Aggregate across all periods
            total_samples = sum(d["SampleCount"] for d in datapoints)
            weighted_avg = sum(d["Average"] * d["SampleCount"] for d in datapoints) / total_samples
            overall_min = min(d["Minimum"] for d in datapoints)
            overall_max = max(d["Maximum"] for d in datapoints)
            results[metric_name] = {
                "mean_ms": round(weighted_avg / 1000, 2),   # us -> ms
                "min_ms": round(overall_min / 1000, 2),
                "max_ms": round(overall_max / 1000, 2),
                "sample_count": int(total_samples),
            }
        else:
            results[metric_name] = None

    return results


def compute_latency_stats(latencies):
    """Compute latency statistics."""
    if not latencies:
        return {}
    latencies_ms = sorted([t * 1000 for t in latencies])
    n = len(latencies_ms)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "std_ms": round(statistics.stdev(latencies_ms), 2) if n > 1 else 0,
        "min_ms": round(latencies_ms[0], 2),
        "p50_ms": round(latencies_ms[n // 2], 2),
        "p90_ms": round(latencies_ms[int(n * 0.9)], 2),
        "p99_ms": round(latencies_ms[min(int(n * 0.99), n - 1)], 2),
        "max_ms": round(latencies_ms[-1], 2),
    }


def benchmark_mae(predictor, samples, prediction_length, quantile_levels, batch_size):
    """Compute MAE using sliding windows via endpoint."""
    print("\n--- MAE Evaluation (Sliding Windows) ---")
    if not samples:
        print("  No samples, skipping.")
        return {}

    all_predictions = []
    all_ground_truths = [gt for _, gt in samples]
    all_entries = [entry for entry, _ in samples]

    # Send in batches
    total_time = 0
    total_model_time = 0
    for i in range(0, len(all_entries), batch_size):
        batch = all_entries[i:i + batch_size]
        response, elapsed, model_latency = invoke_endpoint(
            predictor, batch, prediction_length, quantile_levels
        )
        total_time += elapsed
        if model_latency is not None:
            total_model_time += model_latency
        all_predictions.extend(response["predictions"])

        done = min(i + batch_size, len(all_entries))
        model_str = f", model={model_latency*1000:.1f}ms" if model_latency else ""
        print(f"  Processed {done}/{len(all_entries)} samples ({elapsed:.2f}s{model_str})")

    # Find median quantile key
    median_key = "0.5"
    if median_key not in all_predictions[0]:
        # Fallback: use mean
        median_key = "mean"

    maes = []
    per_step_errors = []

    for pred, gt in zip(all_predictions, all_ground_truths):
        forecast = np.array(pred[median_key])
        actual = np.array(gt)
        min_len = min(len(forecast), len(actual))
        abs_error = np.abs(forecast[:min_len] - actual[:min_len])
        maes.append(float(abs_error.mean()))
        per_step_errors.append(abs_error)

    # Per-step MAE
    max_steps = max(len(e) for e in per_step_errors)
    per_step_mae = []
    for step in range(max_steps):
        step_errors = [e[step] for e in per_step_errors if step < len(e)]
        per_step_mae.append(round(float(np.mean(step_errors)), 4))

    overall_mae = statistics.mean(maes)
    results = {
        "num_windows": len(maes),
        "overall_mae": round(overall_mae, 4),
        "std_mae": round(statistics.stdev(maes), 4) if len(maes) > 1 else 0,
        "min_mae": round(min(maes), 4),
        "max_mae": round(max(maes), 4),
        "median_mae": round(sorted(maes)[len(maes) // 2], 4),
        "per_step_mae": per_step_mae,
        "total_inference_time_sec": round(total_time, 2),
        "total_model_time_sec": round(total_model_time, 2),
        "total_network_overhead_sec": round(total_time - total_model_time, 2),
    }

    print(f"  Windows:     {results['num_windows']}")
    print(f"  Overall MAE: {results['overall_mae']:.4f}")
    print(f"  Std MAE:     {results['std_mae']:.4f}")
    print(f"  Median MAE:  {results['median_mae']:.4f}")
    print(f"  Min/Max MAE: {results['min_mae']:.4f} / {results['max_mae']:.4f}")
    print(f"  Step MAE (first 5): {results['per_step_mae'][:5]}")
    print(f"  Step MAE (last 5):  {results['per_step_mae'][-5:]}")
    print(f"  Total inference time: {total_time:.2f}s")

    return results


def benchmark_latency(predictor, entries, prediction_length, quantile_levels):
    """Benchmark single-request and batch latency."""
    print("\n--- Latency Benchmark ---")
    if not entries:
        print("  No samples, skipping.")
        return {}

    # Warmup
    print("  Warmup (3 requests) ...")
    for i in range(min(3, len(entries))):
        invoke_endpoint(predictor, [entries[i]], prediction_length, quantile_levels)

    # Single-sample latency
    print("  Single-sample latency ...")
    single_latencies = []
    single_model_latencies = []
    for entry in entries:
        _, elapsed, model_latency = invoke_endpoint(predictor, [entry], prediction_length, quantile_levels)
        single_latencies.append(elapsed)
        if model_latency is not None:
            single_model_latencies.append(model_latency)

    single_stats = compute_latency_stats(single_latencies)
    model_stats = compute_latency_stats(single_model_latencies) if single_model_latencies else {}
    network_latencies = [e - m for e, m in zip(single_latencies, single_model_latencies)] if single_model_latencies else []
    network_stats = compute_latency_stats(network_latencies) if network_latencies else {}

    print(f"    Samples:        {single_stats['count']}")
    print(f"    Total RTT:      {single_stats['mean_ms']:.2f} ms mean, {single_stats['p99_ms']:.2f} ms p99")
    if model_stats:
        print(f"    Model latency:  {model_stats['mean_ms']:.2f} ms mean, {model_stats['p99_ms']:.2f} ms p99")
    if network_stats:
        print(f"    Network overhead:{network_stats['mean_ms']:.2f} ms mean, {network_stats['p99_ms']:.2f} ms p99")

    # Batch latency (use reasonable batch sizes to avoid payload limit)
    batch_sizes = [4, 8, 16, 32, 64, 128]
    n_repeats = 5
    batch_results = {}
    for bs in batch_sizes:
        print(f"  Batch latency (batch_size={bs}, {n_repeats} repeats) ...")

        # Build batch by cycling through entries if not enough samples
        def make_batch(size, offset=0):
            return [entries[i % len(entries)] for i in range(offset, offset + size)]

        # Warmup 1 run
        invoke_endpoint(predictor, make_batch(bs), prediction_length, quantile_levels)

        latencies = []
        model_lats = []
        for r in range(n_repeats):
            batch = make_batch(bs, offset=r * bs)
            _, elapsed, model_latency = invoke_endpoint(predictor, batch, prediction_length, quantile_levels)
            latencies.append(elapsed)
            if model_latency is not None:
                model_lats.append(model_latency)

        mean_elapsed = statistics.mean(latencies)
        mean_model = statistics.mean(model_lats) if model_lats else None
        batch_throughput = bs / mean_elapsed if mean_elapsed > 0 else 0

        batch_stats = {
            "batch_size": bs,
            "repeats": n_repeats,
            "mean_total_ms": round(mean_elapsed * 1000, 2),
            "std_total_ms": round(statistics.stdev(latencies) * 1000, 2) if len(latencies) > 1 else 0,
            "per_sample_ms": round(mean_elapsed * 1000 / bs, 2),
            "throughput_samples_per_sec": round(batch_throughput, 2),
        }
        if mean_model is not None:
            batch_stats["mean_model_ms"] = round(mean_model * 1000, 2)
            batch_stats["mean_network_ms"] = round((mean_elapsed - mean_model) * 1000, 2)

        model_str = f", model={batch_stats['mean_model_ms']:.1f}ms, network={batch_stats['mean_network_ms']:.1f}ms" if mean_model else ""
        print(f"    Total: {batch_stats['mean_total_ms']:.2f} ms (std={batch_stats['std_total_ms']:.2f}){model_str}")
        print(f"    Per sample: {batch_stats['per_sample_ms']:.2f} ms")
        print(f"    Throughput: {batch_stats['throughput_samples_per_sec']:.2f} samples/s")
        batch_results[bs] = batch_stats

    # Concurrent batch throughput
    # Test different batch sizes x concurrency combinations to find optimal config
    # Small batch + high concurrency can hide network latency
    # Large batch + low concurrency saturates GPU directly
    from concurrent.futures import ThreadPoolExecutor, as_completed

    concurrent_batch_sizes = [8, 16, 32]
    concurrency_levels = [1, 2, 4, 8]
    concurrent_results = {}

    print(f"\n  Concurrent batch throughput (batch_size x workers) ...")
    for conc_bs in concurrent_batch_sizes:
        for n_workers in concurrency_levels:
            n_requests = max(n_workers * 3, 10)

            def do_batch_request(_idx, _bs=conc_bs):
                batch = make_batch(_bs, offset=_idx * _bs)
                _, elapsed, _ = invoke_endpoint(predictor, batch, prediction_length, quantile_levels)
                return elapsed

            # Warmup
            do_batch_request(0)

            wall_start = time.perf_counter()
            request_latencies = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(do_batch_request, i) for i in range(n_requests)]
                for f in as_completed(futures):
                    try:
                        request_latencies.append(f.result())
                    except Exception as e:
                        print(f"    Worker error: {e}")

            wall_time = time.perf_counter() - wall_start
            total_samples = len(request_latencies) * conc_bs
            throughput = total_samples / wall_time if wall_time > 0 else 0
            mean_req_ms = statistics.mean(request_latencies) * 1000

            key = f"bs{conc_bs}_w{n_workers}"
            conc_stats = {
                "workers": n_workers,
                "batch_size": conc_bs,
                "n_requests": len(request_latencies),
                "total_samples": total_samples,
                "wall_time_sec": round(wall_time, 3),
                "throughput_samples_per_sec": round(throughput, 2),
                "mean_request_ms": round(mean_req_ms, 2),
            }
            print(f"    batch={conc_bs:>3d} x workers={n_workers}: "
                  f"{throughput:>7.1f} samples/s | mean_req={mean_req_ms:.1f}ms")
            concurrent_results[key] = conc_stats

    result = {
        "single_sample": single_stats,
        "batch": batch_results,
        "concurrent": concurrent_results,
    }
    if model_stats:
        result["single_sample_model"] = model_stats
    if network_stats:
        result["single_sample_network"] = network_stats
    return result


def smoke_test(predictor):
    """Quick smoke test with synthetic data."""
    print("\n--- Smoke Test ---")
    payload = {
        "inputs": [
            {"target": [0.0, 4.0, 5.0, 1.5, -3.0, -5.0, -3.0, 1.5, 5.0, 4.0,
                         0.0, -4.0, -5.0, -1.5, 3.0, 5.0, 3.0, -1.5, -5.0, -4.0]},
        ],
        "parameters": {"prediction_length": 10},
    }

    t0 = time.perf_counter()
    response = predictor.predict(payload)
    elapsed = time.perf_counter() - t0

    predictions = response["predictions"][0]
    print(f"  Mean forecast: {[round(v, 2) for v in predictions['mean']]}")
    print(f"  Latency: {elapsed * 1000:.1f} ms")
    print("  Smoke test passed!")
    return True


def main():
    args = parse_args()
    quantile_levels = [float(q) for q in args.quantile_levels.split(",")]

    print("=" * 70)
    print("Chronos-2 SageMaker Endpoint Test")
    print("=" * 70)
    print(f"  Endpoint:          {args.endpoint_name}")
    print(f"  Context length:    {args.context_length}")
    print(f"  Prediction length: {args.prediction_length}")
    print(f"  Covariates:        {args.use_covariates}")
    print(f"  Num samples:       {args.num_samples}")
    print(f"  MAE windows:       {args.n_windows} per series")
    print(f"  Quantile levels:   {quantile_levels}")
    print(f"  Batch size:        {args.batch_size}")
    print("=" * 70)

    # Connect to endpoint
    if args.region:
        import boto3
        boto_session = boto3.Session(region_name=args.region)
        import sagemaker
        sm_session = sagemaker.Session(boto_session=boto_session)
    else:
        sm_session = None

    predictor = Predictor(
        args.endpoint_name,
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    # Record start time for CloudWatch query
    from datetime import datetime, timezone
    benchmark_start_time = datetime.now(timezone.utc)

    # 1. Smoke test
    smoke_test(predictor)

    # 2. Prepare data
    mae_samples = prepare_sliding_window_data(args)
    latency_entries = prepare_single_samples(args)

    all_results = {
        "config": {
            "endpoint_name": args.endpoint_name,
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "use_covariates": args.use_covariates,
            "num_samples": args.num_samples,
            "n_windows": args.n_windows,
            "stride_days": args.stride_days,
            "forecast_days": args.forecast_days,
            "batch_size": args.batch_size,
            "quantile_levels": quantile_levels,
            "mae_total_windows": len(mae_samples),
            "latency_total_samples": len(latency_entries),
        },
    }

    # 3. MAE evaluation
    forecast_len = int(48 * args.forecast_days)
    all_results["mae"] = benchmark_mae(
        predictor, mae_samples, forecast_len, quantile_levels, args.batch_size
    )

    # 4. Latency benchmark
    all_results["latency"] = benchmark_latency(
        predictor, latency_entries, args.prediction_length, quantile_levels
    )

    # 5. Query CloudWatch for model vs overhead latency
    benchmark_end_time = datetime.now(timezone.utc)
    region = args.region or (sm_session.boto_region_name if sm_session else "us-east-1")
    print("\n--- CloudWatch Latency (Model vs Overhead) ---")
    print("  Querying CloudWatch metrics (waiting 10s for propagation) ...")
    time.sleep(10)
    cw_metrics = query_cloudwatch_latency(
        args.endpoint_name, region, benchmark_start_time, benchmark_end_time
    )
    all_results["cloudwatch_latency"] = cw_metrics

    if cw_metrics.get("ModelLatency"):
        ml = cw_metrics["ModelLatency"]
        print(f"  ModelLatency:    {ml['mean_ms']:.2f} ms mean, {ml['min_ms']:.2f}-{ml['max_ms']:.2f} ms range ({ml['sample_count']} samples)")
        print(f"    (Note: averaged across ALL requests including large batches)")
    else:
        print("  ModelLatency:    no data (metrics may need more time to propagate)")
    if cw_metrics.get("OverheadLatency"):
        ol = cw_metrics["OverheadLatency"]
        print(f"  OverheadLatency: {ol['mean_ms']:.2f} ms mean, {ol['min_ms']:.2f}-{ol['max_ms']:.2f} ms range ({ol['sample_count']} samples)")
    else:
        print("  OverheadLatency: no data")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_results.get("mae"):
        mae_res = all_results["mae"]
        print(f"MAE:              {mae_res['overall_mae']:.4f} (std={mae_res['std_mae']:.4f})")
    if all_results.get("latency", {}).get("single_sample"):
        sl = all_results["latency"]["single_sample"]
        print(f"Single RTT:       {sl['mean_ms']:.2f} ms mean, {sl['p99_ms']:.2f} ms p99")
    if cw_metrics.get("OverheadLatency"):
        ol = cw_metrics["OverheadLatency"]
        print(f"SM overhead:      {ol['mean_ms']:.2f} ms mean (CloudWatch)")
    if all_results.get("latency", {}).get("batch"):
        for bs, bl in all_results["latency"]["batch"].items():
            print(f"Batch {bs} throughput: {bl['throughput_samples_per_sec']:.2f} samples/s")
    if all_results.get("latency", {}).get("concurrent"):
        # Find best concurrent config
        best_key = max(
            all_results["latency"]["concurrent"],
            key=lambda k: all_results["latency"]["concurrent"][k]["throughput_samples_per_sec"]
        )
        best = all_results["latency"]["concurrent"][best_key]
        print(f"Best concurrent:  {best['throughput_samples_per_sec']:.1f} samples/s "
              f"(batch={best['batch_size']} x workers={best['workers']})")
    print("=" * 70)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
