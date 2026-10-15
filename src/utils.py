"""Shared utilities: seeding, device, temperature softmax, metrics."""
from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_with_temperature(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return F.softmax(logits / T, dim=-1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
