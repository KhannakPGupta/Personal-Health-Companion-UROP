"""Phase 3 — Subject-Isolated Federated Averaging (FedAvg) & FedProx for Edge Wearables.

Models each wearable participant as an isolated on-device edge client:
  - Raw physiological time-series (BVP, EDA, TEMP, ACC) never leave the client.
  - Only local model weight updates are transmitted to the coordinator.
  - Supports sample-weighted FedAvg (McMahan et al., 2017) and FedProx (Li et al., 2020)
    with proximal regularization mu >= 0 to handle non-IID inter-subject drift.
  - Tracks uplink/downlink communication overhead in kilobytes (float32 vs int8 quantized).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cnn_1d import LightweightStrainCNN
from tests.test_leakage_assertion import assert_no_subject_leakage


@dataclass
class EdgeClientData:
    """Encapsulates a single wearable subject's local on-device dataset."""
    client_id: str
    X: torch.Tensor
    y: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.X.shape[0])


def federated_weighted_average(
    client_state_dicts: Sequence[Dict[str, torch.Tensor]],
    client_sample_counts: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Compute sample-weighted FedAvg aggregation over client state dictionaries."""
    if not client_state_dicts or len(client_state_dicts) != len(client_sample_counts):
        raise ValueError("client_state_dicts and client_sample_counts must be non-empty and equal length.")

    total_samples = float(sum(client_sample_counts))
    if total_samples <= 0:
        raise ValueError("Sum of client_sample_counts must be positive.")

    weights = [count / total_samples for count in client_sample_counts]
    aggregated: Dict[str, torch.Tensor] = {}

    for key in client_state_dicts[0].keys():
        first_tensor = client_state_dicts[0][key]
        if first_tensor.is_floating_point():
            acc = torch.zeros_like(first_tensor, dtype=torch.float32)
            for state, w in zip(client_state_dicts, weights):
                acc += state[key].cpu().float() * w
            aggregated[key] = acc.to(dtype=first_tensor.dtype)
        else:
            # Integer tracking buffers (e.g., BatchNorm num_batches_tracked)
            aggregated[key] = first_tensor.cpu().clone()

    return aggregated


def train_local_edge_client(
    global_model: nn.Module,
    client_data: EdgeClientData,
    local_epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-3,
    fedprox_mu: float = 0.0,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Train a local copy of global_model on a single subject's on-device windows.

    Args:
        global_model: Current global model broadcast from server.
        client_data: Subject-specific local tensor dataset.
        local_epochs: Number of local SGD/Adam epochs E.
        batch_size: Local mini-batch size B.
        lr: Learning rate.
        fedprox_mu: Proximal penalty coefficient (0.0 = standard FedAvg; > 0.0 = FedProx).
        device: Target compute device ('cpu').

    Returns:
        Tuple of (updated_state_dict, average_local_loss).
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    global_params = [p.detach().clone().to(device) for p in global_model.parameters() if p.requires_grad]

    y_local = client_data.y.to(device)
    num_pos = torch.sum(y_local == 1.0).item()
    num_neg = torch.sum(y_local == 0.0).item()
    pos_weight_val = max(1.0, num_neg / max(1.0, num_pos))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    optimizer = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(client_data.X, client_data.y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    total_steps = 0

    for _ in range(local_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = local_model(bx).squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            loss = criterion(logits, by)

            if fedprox_mu > 0.0:
                prox_term = torch.tensor(0.0, device=device)
                for p_local, p_global in zip(
                    (p for p in local_model.parameters() if p.requires_grad),
                    global_params,
                ):
                    prox_term = prox_term + torch.sum((p_local - p_global) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox_term

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    updated_state = {k: v.cpu().clone() for k, v in local_model.state_dict().items()}
    return updated_state, avg_loss


def compute_communication_cost_kb(
    num_parameters: int,
    num_rounds: int,
    clients_per_round: int,
    quantized_int8: bool = False,
) -> Dict[str, float]:
    """Calculate uplink + downlink model payload communication overhead in KB."""
    bytes_per_param = 1.0 if quantized_int8 else 4.0
    single_payload_kb = (num_parameters * bytes_per_param) / 1024.0
    # Each participating client downloads global model (downlink) and uploads local update (uplink)
    per_round_kb = 2.0 * clients_per_round * single_payload_kb
    total_kb = num_rounds * per_round_kb
    return {
        "single_model_payload_kb": round(single_payload_kb, 2),
        "per_round_traffic_kb": round(per_round_kb, 2),
        "total_communication_kb": round(total_kb, 2),
    }


def run_federated_training(
    clients: Sequence[EdgeClientData],
    test_subjects: Sequence[str],
    in_channels: int = 6,
    rounds: int = 8,
    local_epochs: int = 2,
    client_fraction: float = 1.0,
    batch_size: int = 16,
    lr: float = 1e-3,
    fedprox_mu: float = 0.0,
    random_seed: int = 42,
) -> Tuple[LightweightStrainCNN, List[Dict[str, float]]]:
    """Run simulated FedAvg/FedProx across subject edge clients with strict leakage verification."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    client_ids = [c.client_id for c in clients]
    assert_no_subject_leakage(client_ids, test_subjects)

    global_model = LightweightStrainCNN(in_channels=in_channels, num_classes=1)
    n_params = global_model.count_parameters()
    round_logs: List[Dict[str, float]] = []

    n_clients = len(clients)
    m_select = max(1, int(np.ceil(client_fraction * n_clients)))

    for r_idx in range(1, rounds + 1):
        selected_indices = rng.choice(n_clients, size=m_select, replace=False)
        selected_clients = [clients[i] for i in selected_indices]

        local_states = []
        sample_counts = []
        round_losses = []

        for client in selected_clients:
            state_dict, c_loss = train_local_edge_client(
                global_model=global_model,
                client_data=client,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                fedprox_mu=fedprox_mu,
            )
            local_states.append(state_dict)
            sample_counts.append(client.num_samples)
            round_losses.append(c_loss)

        new_global_state = federated_weighted_average(local_states, sample_counts)
        global_model.load_state_dict(new_global_state)

        comm_f32 = compute_communication_cost_kb(n_params, r_idx, m_select, quantized_int8=False)
        comm_int8 = compute_communication_cost_kb(n_params, r_idx, m_select, quantized_int8=True)

        round_logs.append({
            "round": r_idx,
            "participating_clients": m_select,
            "mean_client_loss": round(float(np.mean(round_losses)), 4),
            "cumulative_comm_float32_kb": comm_f32["total_communication_kb"],
            "cumulative_comm_int8_kb": comm_int8["total_communication_kb"],
        })

    return global_model, round_logs
