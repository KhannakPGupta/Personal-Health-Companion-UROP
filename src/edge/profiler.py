"""CPU inference and feature extraction latency profiling for edge deployment."""

from __future__ import annotations

import time
from typing import Callable, Dict, Tuple
import numpy as np
import torch

from src.features.extractor import extract_window_features_and_tensor


def benchmark_inference_latency(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    n_warmup: int = 20,
    n_runs: int = 100,
) -> Dict[str, float]:
    """Measure single-window CPU inference latency statistics in milliseconds."""
    model.eval()
    sample = sample_tensor.cpu()
    if sample.dim() == 2:
        sample = sample.unsqueeze(0) # (1, 6, 1920)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)
    return {
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
    }


def benchmark_feature_extraction_latency(
    bvp: np.ndarray,
    eda: np.ndarray,
    temp: np.ndarray,
    acc: np.ndarray,
    n_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark end-to-end signal processing and feature calculation latency per window."""
    # Warm-up
    _ = extract_window_features_and_tensor(bvp, eda, temp, acc)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = extract_window_features_and_tensor(bvp, eda, temp, acc)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "feature_extraction_mean_ms": round(float(np.mean(arr)), 3),
        "feature_extraction_p50_ms": round(float(np.percentile(arr, 50)), 3),
        "feature_extraction_p95_ms": round(float(np.percentile(arr, 95)), 3),
    }
