"""torch ``Dataset`` wrappers around the extracted feature matrices."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Serve one or more aligned feature matrices plus the integer label.

    Pass whichever of ``X_eeg`` / ``X_motion`` / ``X_emg`` a given model consumes;
    ``__getitem__`` yields ``(*present_features, y)`` in that fixed order, so the
    training loop can branch on ``len(batch)``.
    """

    def __init__(self, y, X_eeg=None, X_motion=None, X_emg=None):
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.X_eeg = _as_float(X_eeg)
        self.X_motion = _as_float(X_motion)
        self.X_emg = _as_float(X_emg)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        parts = [
            X[idx] for X in (self.X_eeg, self.X_motion, self.X_emg) if X is not None
        ]
        return (*parts, self.y[idx])


class ConcatFeatureDataset(Dataset):
    """Concatenate every provided feature matrix into a single flat vector."""

    def __init__(self, y, *feature_matrices):
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        mats = [torch.as_tensor(np.asarray(m), dtype=torch.float32) for m in feature_matrices]
        self.X = torch.cat([m.reshape(len(m), -1) for m in mats], dim=1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _as_float(x):
    if x is None:
        return None
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)
