"""Fully-connected movement-intention classifiers.

Three variants differing only in which modalities they consume:

===================  =================================================
``FNN_OnlyEEG``      EEG features only
``FNN_NonEEG``       motion + EMG features (no brain signal)
``FNN_Fusion``       EEG + motion + EMG features
===================  =================================================

Each ``forward`` concatenates its inputs and passes them through a small MLP
with ReLU activations to a 7-way logit vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import N_CLASSES


def _mlp(in_dim: int, hidden: list[int], n_class: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_class))
    return nn.Sequential(*layers)


class FNN_OnlyEEG(nn.Module):
    """EEG-only classifier."""

    def __init__(self, d_eeg: int, n_class: int = N_CLASSES, hidden=(128, 64, 32)):
        super().__init__()
        self.net = _mlp(d_eeg, list(hidden), n_class)

    def forward(self, x_eeg: torch.Tensor) -> torch.Tensor:
        return self.net(x_eeg)


class FNN_NonEEG(nn.Module):
    """Motion + EMG classifier (no EEG)."""

    def __init__(self, d_motion: int, d_emg: int, n_class: int = N_CLASSES, hidden=(128, 64, 32)):
        super().__init__()
        self.net = _mlp(d_motion + d_emg, list(hidden), n_class)

    def forward(self, x_motion: torch.Tensor, x_emg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_motion, x_emg], dim=1))


class FNN_Fusion(nn.Module):
    """EEG + motion + EMG multimodal classifier."""

    def __init__(
        self,
        d_eeg: int,
        d_motion: int,
        d_emg: int,
        n_class: int = N_CLASSES,
        hidden=(256, 128, 64, 32),
    ):
        super().__init__()
        self.net = _mlp(d_eeg + d_motion + d_emg, list(hidden), n_class)

    def forward(self, x_eeg: torch.Tensor, x_motion: torch.Tensor, x_emg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_eeg, x_motion, x_emg], dim=1))
