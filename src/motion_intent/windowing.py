"""Time-block train/val/test split and sliding-window extraction.

The split is by time (not random) so that windows near a boundary never leak
between train and test. Windows are labelled with the class at their centre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #

def split_by_time_block(
    df: pd.DataFrame,
    t_col: str = "t_sec",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Chronological train / val / test split of a single session.

    Returns ``(df_train, df_val, df_test)``. ``test_ratio`` is the remainder.
    """
    t_min, t_max = df[t_col].min(), df[t_col].max()
    span = t_max - t_min
    t_train_end = t_min + span * train_ratio
    t_val_end = t_min + span * (train_ratio + val_ratio)

    df_train = df[df[t_col] < t_train_end]
    df_val = df[(df[t_col] >= t_train_end) & (df[t_col] < t_val_end)]
    df_test = df[df[t_col] >= t_val_end]
    return df_train, df_val, df_test


def downsample_rows(
    df: pd.DataFrame, keep_ratio: float, random_state: int | None = None
) -> pd.DataFrame:
    """Randomly keep a fraction of rows (index order preserved).

    Handy for shrinking a session during quick experiments; use
    ``keep_ratio=1.0`` for the real run.
    """
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    n_keep = int(len(df) * keep_ratio)
    if n_keep < 1:
        raise ValueError("keep_ratio too small: 0 rows would remain")
    return df.sample(n=n_keep, replace=False, random_state=random_state).sort_index()


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #

def make_reference_indices(t_sec: np.ndarray, win_sec: float, step_sec: float) -> np.ndarray:
    """Sample indices of the window *end* points, spaced ``step_sec`` apart.

    The first reference is ``win_sec`` after the recording start so every window
    is fully populated.
    """
    t_sec = np.asarray(t_sec)
    t_refs = np.arange(t_sec[0] + win_sec, t_sec[-1], step_sec)
    return np.searchsorted(t_sec, t_refs)


def windows_at(
    X: np.ndarray, ref_idx: np.ndarray, win: int
) -> np.ndarray:
    """Extract ``(n_ref, n_channels, win)`` windows ending at each ``ref_idx``.

    References with ``idx < win`` are skipped. ``X`` is ``(n_time, n_channels)``.
    """
    out = [X[i - win: i].T for i in ref_idx if i >= win]
    return np.stack(out) if out else np.empty((0, X.shape[1], win))


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    channel_indices,
    win: int,
    step: int,
):
    """Dense sliding windows over ``X`` with the centre-sample label.

    Parameters
    ----------
    X : (n_time, n_channels)
    y : (n_time,)
    channel_indices : columns of ``X`` to keep
    win, step : window length and hop, in samples

    Returns
    -------
    X_win : (n_windows, len(channel_indices), win)
    y_win : (n_windows,)
    """
    Xc = X[:, channel_indices]
    xs, ys = [], []
    for start in range(0, len(X) - win + 1, step):
        xs.append(Xc[start: start + win].T)
        ys.append(y[start + win // 2])
    return np.stack(xs), np.asarray(ys)


def transition_indices(labels: np.ndarray) -> np.ndarray:
    """Sample indices where the class label changes."""
    return np.where(labels[1:] != labels[:-1])[0] + 1


def transition_offset_refs(
    labels: np.ndarray,
    offset_samples: int,
    n_samples: int,
    min_ref: int,
):
    """Window end-points placed at ``(each transition + offset_samples)``.

    Used by the "how early can the upcoming movement be read out" analysis:
    every label change is a candidate onset, and windows are re-centred around
    it at a range of offsets.

    Parameters
    ----------
    labels : (n_time,) class ids of one session
    offset_samples : shift applied to every transition index (can be negative)
    n_samples : session length, so refs past the end are dropped
    min_ref : smallest valid end-point (usually the largest window length)

    Returns
    -------
    refs : (k,) end-point sample indices, kept only if ``min_ref <= ref <= n_samples``
    y : (k,) the **post-transition** class for each ref (the movement being started)
    """
    tr = transition_indices(labels)
    refs = tr + int(offset_samples)
    ok = (refs >= min_ref) & (refs <= n_samples)
    return refs[ok], np.asarray(labels)[tr[ok]]


# --------------------------------------------------------------------------- #
# ragged-sequence padding (for the sequence models)
# --------------------------------------------------------------------------- #

def pad_time_axis(x: np.ndarray, t_max: int) -> np.ndarray:
    """Zero-pad axis 0 of ``x`` (T, C, W) up to ``t_max``."""
    pad = t_max - x.shape[0]
    if pad <= 0:
        return x
    return np.concatenate([x, np.zeros((pad, *x.shape[1:]), dtype=x.dtype)], axis=0)


def pad_labels(y: np.ndarray, t_max: int, pad_value: int = -1) -> np.ndarray:
    """Pad a 1-D label vector up to ``t_max`` with ``pad_value``."""
    pad = t_max - len(y)
    if pad <= 0:
        return y
    return np.concatenate([y, np.full(pad, pad_value, dtype=y.dtype)])


def make_mask(length: int, t_max: int) -> np.ndarray:
    """Boolean mask, ``True`` for the first ``length`` of ``t_max`` steps."""
    mask = np.zeros(t_max, dtype=bool)
    mask[:length] = True
    return mask
