"""Generic train / evaluate loop with early stopping.

The models take a variable number of feature tensors, so batches from
:class:`~motion_intent.datasets.FeatureDataset` are ``(*features, y)``. Both
loops below dispatch on ``len(batch)``: 2 -> single input, 3 -> two inputs,
4 -> three inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _forward_batch(model, batch, device):
    *xs, y = batch
    xs = [x.to(device) for x in xs]
    y = y.to(device)
    return model(*xs), y


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits, y = _forward_batch(model, batch, device)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Return ``(accuracy, y_true, y_pred)`` over ``loader``."""
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        logits, y = _forward_batch(model, batch, device)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return acc, y_true, y_pred


def weight_update_norm(model, prev_params) -> float:
    """L2 norm of the parameter change since ``prev_params`` (a convergence probe)."""
    return float(
        sum(torch.norm(p - q, p=2).item() for p, q in zip(model.parameters(), prev_params))
    )


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    weight_norm: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = -1


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 30,
    min_delta: float = 1e-3,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> History:
    """Adam training with early stopping on validation accuracy.

    Restores the best-seen weights into ``model`` in place and returns the
    :class:`History` (loss / val-acc / weight-update-norm per epoch).
    """
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    hist = History()
    best_state = deepcopy(model.state_dict())
    prev_params = [p.detach().clone() for p in model.parameters()]
    stale = 0

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc, _, _ = eval_epoch(model, val_loader, device)
        wnorm = weight_update_norm(model, prev_params)
        prev_params = [p.detach().clone() for p in model.parameters()]

        hist.train_loss.append(loss)
        hist.val_acc.append(acc)
        hist.weight_norm.append(wnorm)

        if acc > hist.best_val_acc + min_delta:
            hist.best_val_acc = acc
            hist.best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if verbose:
            print(f"epoch {epoch:03d} | loss {loss:.4f} | val_acc {acc:.3f} | dW {wnorm:.3e}")

        if stale >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best {hist.best_val_acc:.3f} @ {hist.best_epoch})")
            break

    model.load_state_dict(best_state)
    return hist
