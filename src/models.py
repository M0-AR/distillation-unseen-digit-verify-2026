"""Models with EXACT video parameter counts (verified by execution).

Teacher: 64 -> 400 -> 400 -> 10
  params = (64*400+400) + (400*400+400) + (400*10+10) = 190410
Student: 64 -> 128 -> 10
  params = (64*128+128) + (128*10+10) = 9610
"""
from __future__ import annotations

import torch.nn as nn


class TeacherMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )

    def forward(self, x):
        return self.net(x)


class StudentMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
