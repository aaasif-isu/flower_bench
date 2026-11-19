"""Deterministic seeding utilities for reproducible FL experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def client_round_seed(training_seed: int, round_id: int, client_id: int) -> int:
    """Deterministic per-client, per-round seed matching Track 1 & Track 4."""
    return int(training_seed) + int(round_id) * 10000 + int(client_id)
