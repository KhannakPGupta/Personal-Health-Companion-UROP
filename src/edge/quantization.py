"""Post-training int8 quantization and edge memory profiling for Conv1d + Linear models."""

from __future__ import annotations

import copy
import io
from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


def quantize_model_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Apply symmetric per-tensor int8 weight quantization across Conv1d/Linear + dynamic int8 Linear."""
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    with torch.no_grad():
        for module in model_cpu.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)) and module.weight is not None:
                w = module.weight.data
                max_val = float(torch.max(torch.abs(w)).item())
                scale = max(max_val / 127.0, 1e-8)
                w_q = torch.clamp(torch.round(w / scale), -128, 127)
                module.weight.data.copy_(w_q * scale)

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def measure_model_size_kb(model: torch.nn.Module, quantized_int8: bool = False) -> float:
    """Measure model parameter & buffer footprint in kilobytes (KB)."""
    if not quantized_int8:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return round(buf.getbuffer().nbytes / 1024.0, 2)

    # Calculate packed int8 weight storage (1 byte per Conv1d/Linear weight + 4 bytes per scale/bias/norm param)
    total_bytes = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                total_bytes += module.weight.numel() * 1 + 4  # 1B per int8 weight + 4B scale factor
            if hasattr(module, "bias") and module.bias is not None:
                total_bytes += module.bias.numel() * 4
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.InstanceNorm1d)):
            if hasattr(module, "weight") and module.weight is not None:
                total_bytes += module.weight.numel() * 4
            if hasattr(module, "bias") and module.bias is not None:
                total_bytes += module.bias.numel() * 4
    if total_bytes == 0:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        total_bytes = buf.getbuffer().nbytes // 3
    return round(total_bytes / 1024.0, 2)


def profile_quantization(
    float_model: torch.nn.Module,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> Dict[str, float]:
    """Profile float32 vs int8 parameter count, memory footprint, Macro-F1, Balanced Acc, and accuracy delta."""
    float_model.eval()
    val_x_cpu = val_x.cpu()
    val_y_cpu = val_y.cpu().numpy().astype(int)

    param_count = sum(p.numel() for p in float_model.parameters())

    float_size_kb = measure_model_size_kb(float_model, quantized_int8=False)
    with torch.no_grad():
        float_logits = float_model(val_x_cpu).squeeze(-1)
        float_preds = (torch.sigmoid(float_logits).numpy() >= 0.5).astype(int)
    float_acc = float(np.mean(float_preds == val_y_cpu))
    float_f1 = float(f1_score(val_y_cpu, float_preds, average="macro", zero_division=0))
    float_bal = float(balanced_accuracy_score(val_y_cpu, float_preds)) if len(np.unique(val_y_cpu)) > 1 else 0.5

    int8_model = quantize_model_int8(float_model)
    int8_size_kb = measure_model_size_kb(float_model, quantized_int8=True)
    with torch.no_grad():
        int8_logits = int8_model(val_x_cpu).squeeze(-1)
        int8_preds = (torch.sigmoid(int8_logits).numpy() >= 0.5).astype(int)
    int8_acc = float(np.mean(int8_preds == val_y_cpu))
    int8_f1 = float(f1_score(val_y_cpu, int8_preds, average="macro", zero_division=0))
    int8_bal = float(balanced_accuracy_score(val_y_cpu, int8_preds)) if len(np.unique(val_y_cpu)) > 1 else 0.5

    return {
        "parameters": param_count,
        "float_size_kb": float_size_kb,
        "int8_size_kb": int8_size_kb,
        "compression_ratio": round(float_size_kb / max(1e-4, int8_size_kb), 2),
        "float_macro_f1": round(float_f1, 4),
        "int8_macro_f1": round(int8_f1, 4),
        "delta_macro_f1": round(int8_f1 - float_f1, 4),
        "float_balanced_acc": round(float_bal, 4),
        "int8_balanced_acc": round(int8_bal, 4),
        "float_accuracy": round(float_acc, 4),
        "int8_accuracy": round(int8_acc, 4),
        "delta_accuracy": round(int8_acc - float_acc, 4),
    }
