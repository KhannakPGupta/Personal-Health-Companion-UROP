"""Phase 3 — Federated Edge Learning Simulation (FedAvg & FedProx)."""

from src.federated.fedavg import (
    EdgeClientData,
    federated_weighted_average,
    train_local_edge_client,
    compute_communication_cost_kb,
    run_federated_training,
)
from src.federated.federated_evaluator import run_federated_loso_evaluation

__all__ = [
    "EdgeClientData",
    "federated_weighted_average",
    "train_local_edge_client",
    "compute_communication_cost_kb",
    "run_federated_training",
    "run_federated_loso_evaluation",
]
