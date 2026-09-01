"""Derive per-sample movement labels from EMG and motion-capture traces.

The experiment protocol cues one movement at a time, but the exact onset/offset
of each phase has to be recovered from the signals. The helpers here cover the
pieces reused across the per-movement labelling notebooks:

* EMG activation envelope + onset detection (sit-to-stand, stepping)
* marker-height thresholding (standing vs. sitting, stair contact)
* label post-processing: short-gap filling and majority smoothing
* transition indexing for review / evaluation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert


# --------------------------------------------------------------------------- #
# EMG activation
# --------------------------------------------------------------------------- #

def emg_envelope(x: np.ndarray, fs: float, band=(20.0, 450.0), smooth_hz: float = 5.0) -> np.ndarray:
    """Linear envelope of an EMG channel: band-pass -> Hilbert magnitude -> low-pass."""
    nyq = fs / 2.0
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="band")
    bp = filtfilt(b, a, x)
    env = np.abs(hilbert(bp))
    bl, al = butter(2, smooth_hz / nyq, btype="low")
    return filtfilt(bl, al, env)


def onset_from_envelope(env: np.ndarray, fs: float, k: float = 3.0, min_gap_sec: float = 0.5) -> np.ndarray:
    """Sample indices where ``env`` rises above ``mean + k * std`` (rest baseline).

    Onsets closer than ``min_gap_sec`` are collapsed to the first.
    """
    thr = env.mean() + k * env.std()
    above = env > thr
    rising = np.where((~above[:-1]) & (above[1:]))[0] + 1

    min_gap = int(min_gap_sec * fs)
    out: list[int] = []
    for idx in rising:
        if not out or idx - out[-1] >= min_gap:
            out.append(int(idx))
    return np.asarray(out, dtype=int)


# --------------------------------------------------------------------------- #
# marker-height thresholding
# --------------------------------------------------------------------------- #

def threshold_state(series: np.ndarray, thr: float, above_label: str, below_label: str) -> np.ndarray:
    """Binary state string per sample from a single scalar trace (e.g. hip height)."""
    return np.where(np.asarray(series) >= thr, above_label, below_label)


# --------------------------------------------------------------------------- #
# label post-processing
# --------------------------------------------------------------------------- #

def fill_short_gaps(labels: pd.Series, max_gap: int) -> pd.Series:
    """Replace runs of NaN/None no longer than ``max_gap`` with the preceding label."""
    labels = labels.copy()
    is_na = labels.isna().to_numpy()
    i = 0
    n = len(labels)
    while i < n:
        if is_na[i]:
            j = i
            while j < n and is_na[j]:
                j += 1
            if j - i <= max_gap and i > 0:
                labels.iloc[i:j] = labels.iloc[i - 1]
            i = j
        else:
            i += 1
    return labels


def majority_smooth(labels: np.ndarray, win: int) -> np.ndarray:
    """Sliding-window mode filter that removes single-sample label flicker."""
    labels = np.asarray(labels, dtype=object)
    out = labels.copy()
    half = win // 2
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def label_transitions(labels: np.ndarray) -> np.ndarray:
    """Sample indices where the label changes value."""
    labels = np.asarray(labels)
    return np.where(labels[:-1] != labels[1:])[0] + 1


def labels_to_ids(labels: pd.Series, class_names: list[str]) -> np.ndarray:
    """Map string labels to integer class ids (``-1`` for anything unmapped)."""
    lut = {name: i for i, name in enumerate(class_names)}
    return labels.map(lambda s: lut.get(s, -1)).to_numpy()
