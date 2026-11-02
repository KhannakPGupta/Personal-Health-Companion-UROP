"""Post-training int8 quantization and edge memory profiling."""

from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, Tuple
import torch
import numpy as np


def quantize_model_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic post-training int8 quantization to linear/dense layers."""
    model_cpu = model.cpu()
    model_cpu.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def measure_model_size_kb(model: torch.nn.Module) -> float:
    """Measure serialized model checkpoint size in kilobytes (KB)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_bytes = buf.getbuffer().nbytes
    return round(size_bytes / 1024.0, 2)


def profile_quantization(
    float_model: torch.nn.Module,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> Dict[str, float]:
    """Profile float32 vs int8 parameter count, memory footprint, and accuracy delta."""
    float_model.eval()
    val_x_cpu = val_x.cpu()
    val_y_cpu = val_y.cpu().numpy()

    # Parameter count
    param_count = sum(p.numel() for p in float_model.parameters())

    # Float model size and accuracy
    float_size_kb = measure_model_size_kb(float_model)
    with torch.no_grad():
        float_logits = float_model(val_x_cpu).squeeze(-1)
        float_preds = (torch.sigmoid(float_logits).numpy() >= 0.5).astype(int)
    float_acc = float(np.mean(float_preds == val_y_cpu))

    # Quantize to int8
    int8_model = quantize_model_int8(float_model)
    int8_size_kb = measure_model_size_kb(int8_model)
    with torch.no_grad():
        int8_logits = int8_model(val_x_cpu).squeeze(-1)
        int8_preds = (torch.sigmoid(int8_logits).numpy() >= 0.5).astype(int)
    int8_acc = float(np.mean(int8_preds == val_y_cpu))

    delta_acc = round(int8_acc - float_acc, 4)

    return {
        "parameters": param_count,
        "float_size_kb": float_size_kb,
        "int8_size_kb": int8_size_kb,
        "compression_ratio": round(float_size_kb / max(1e-4, int8_size_kb), 2),
        "float_accuracy": round(float_acc, 4),
        "int8_accuracy": round(int8_acc, 4),
        "delta_accuracy": delta_acc,
    }
