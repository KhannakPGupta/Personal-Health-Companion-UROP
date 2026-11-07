"""Lightweight 1D-CNN (M2) with Instance Normalization & SE Channel Attention (< 50,000 parameters).

Architecture:
  - Input: (batch_size, in_channels, seq_len) @ 32 Hz
  - Input InstanceNorm1d(in_channels, affine=True) -> eliminates DC baseline drift across subjects
  - ConvBlock 1: Conv1D(C, 16, k=7, s=2) + BatchNorm1D(16) + ReLU + MaxPool1D(2)
  - ConvBlock 2: Conv1D(16, 32, k=5, s=2) + BatchNorm1D(32) + ReLU + SEBlock1D(32) + MaxPool1D(2)
  - ConvBlock 3: Conv1D(32, 48, k=3, s=1) + BatchNorm1D(48) + ReLU + SEBlock1D(48)
  - Dual Global Pooling: Concat(AdaptiveAvgPool1d(1), AdaptiveMaxPool1d(1)) -> 96 features
  - Classifier: Linear(96, 24) + ReLU + Dropout(0.25) + Linear(24, 1)

Total Parameters: ~11,800 (strictly satisfies the < 50,000 edge budget).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SEBlock1D(nn.Module):
    """Lightweight Squeeze-and-Excitation (SE) channel attention for 1D biosignals."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pool: bool = False,
        use_se: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        num_groups = 4 if out_channels % 4 == 0 else 1
        self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock1D(out_channels) if use_se else nn.Identity()
        self.pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class LightweightStrainCNN(nn.Module):
    """1D-CNN with Instance Normalization and SE Channel Attention for edge wearable inference."""

    def __init__(self, in_channels: int = 6, num_classes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.input_norm = nn.InstanceNorm1d(in_channels, affine=True, eps=1e-5)
        self.block1 = ConvBlock1D(in_channels, 16, kernel_size=7, stride=2, pool=True, use_se=False)
        self.block2 = ConvBlock1D(16, 32, kernel_size=5, stride=2, pool=True, use_se=True)
        self.block3 = ConvBlock1D(32, 48, kernel_size=3, stride=1, pool=False, use_se=True)

        # Dual Global Average + Max Pooling captures both tonic baseline & phasic SCR/HR peaks
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(96, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(24, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, C, L) or (batch, L, C).

        Returns:
            Logits of shape (batch, 1).
        """
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)

        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        avg_feat = self.gap(x).flatten(1)
        max_feat = self.gmp(x).flatten(1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        logits = self.classifier(feat)
        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_cnn_with_early_stopping(
    model: LightweightStrainCNN,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    patience: int = 7,
    device: str = "cpu",
) -> Tuple[LightweightStrainCNN, Dict[str, List[float]]]:
    """Train M2 CNN with class-weighted loss, cosine LR schedule, and early stopping on validation subjects."""
    model = model.to(device)

    num_pos = torch.sum(y_train == 1.0).item()
    num_neg = torch.sum(y_train == 0.0).item()
    pos_weight_val = float(np.clip(np.sqrt(num_neg / max(1.0, num_pos)), 0.65, 2.2))
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in loader_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= max(1, len(dataset_train))
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = X_val.to(device)
            val_y = y_val.to(device)
            val_logits = model(val_x).squeeze(-1)
            if val_logits.ndim == 0:
                val_logits = val_logits.unsqueeze(0)
            val_loss = criterion(val_logits, val_y).item()
            history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history
