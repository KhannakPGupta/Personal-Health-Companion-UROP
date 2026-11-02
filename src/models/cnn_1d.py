"""Lightweight 1D-CNN (M2) for on-device edge execution (< 50,000 parameters).

Architecture:
  - Input: (batch_size, 6_channels, 1920_samples) @ 32 Hz
  - ConvBlock 1: Conv1D(6, 16, k=7, s=2) + BatchNorm1D(16) + ReLU + MaxPool1D(2)
  - ConvBlock 2: Conv1D(16, 32, k=5, s=2) + BatchNorm1D(32) + ReLU + MaxPool1D(2)
  - ConvBlock 3: Conv1D(32, 48, k=3, s=1) + BatchNorm1D(48) + ReLU
  - GlobalAveragePooling1D -> 48 features
  - Dense(48, 16) + ReLU + Dropout(0.2)
  - Dense(16, 1) + Sigmoid output

Total Parameters: ~10,000 to 15,000 (strictly satisfies the < 50,000 edge budget).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, pool: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class LightweightStrainCNN(nn.Module):
    """1D-CNN designed for ultra-low latency wearable edge inference."""

    def __init__(self, in_channels: int = 6, num_classes: int = 1):
        super().__init__()
        self.block1 = ConvBlock1D(in_channels, 16, kernel_size=7, stride=2, pool=True)
        self.block2 = ConvBlock1D(16, 32, kernel_size=5, stride=2, pool=True)
        self.block3 = ConvBlock1D(32, 48, kernel_size=3, stride=1, pool=False)
        
        # Global Average Pooling collapses (B, 48, L) -> (B, 48)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 6, 1920) or (batch, 1920, 6).
            
        Returns:
            Logits of shape (batch, 1).
        """
        if x.dim() == 3 and x.shape[1] != 6 and x.shape[2] == 6:
            # Transpose (B, L, C) -> (B, C, L)
            x = x.transpose(1, 2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        logits = self.classifier(x)
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
    lr: float = 1e-3,
    patience: int = 7,
    device: str = "cpu",
) -> Tuple[LightweightStrainCNN, Dict[str, List[float]]]:
    """Train M2 CNN with class-weighted loss and early stopping on validation subjects."""
    model = model.to(device)
    
    # Class-weighted binary cross entropy
    num_pos = torch.sum(y_train == 1.0).item()
    num_neg = torch.sum(y_train == 0.0).item()
    pos_weight_val = max(1.0, num_neg / max(1.0, num_pos))
    pos_weight = torch.tensor([pos_weight_val], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(dataset_train)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = X_val.to(device)
            val_y = y_val.to(device)
            val_logits = model(val_x).squeeze(-1)
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
