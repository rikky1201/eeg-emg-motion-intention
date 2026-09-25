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
    """Remove label flicker shorter than ``win`` samples by merging each such
    run into the run before it (the first run, if it is itself too short,
    merges into the one after).

    Runs found via a vectorised run-length encoding rather than a per-sample
    sliding-window vote: with long, mostly-stable label tracks (a resting
    plateau lasting thousands of samples) a windowed vote re-examines that
    same stable stretch at every single sample for no benefit, which made
    this the dominant cost of labelling a session at 1kHz.
    """
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    if n == 0:
        return labels.copy()
    change = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    values = labels[starts].copy()
    lengths = ends - starts

    for i in range(len(values)):
        if lengths[i] < win:
            neighbor = i - 1 if i > 0 else i + 1
            if 0 <= neighbor < len(values):
                values[i] = values[neighbor]

    out = np.empty(n, dtype=object)
    for s, e, v in zip(starts, ends, values):
        out[s:e] = v
    return out


# --------------------------------------------------------------------------- #
# hysteresis phase labelling (repeating low/high-plateau sessions)
# --------------------------------------------------------------------------- #
#
# Several sessions repeat a cycle that alternates between two stable levels of
# a motion feature (sit/stand hip height, stair-bottom/stair-top hip height,
# still/oscillating heel height for stand/walk) joined by a fast transition.
# A generic change-point search (fixed number of equal-cost segments) turned
# out to be unreliable here: the two transitions in a cycle are rarely the
# same duration (e.g. climbing stairs is quicker than descending them), so a
# fixed per-cycle segment budget puts boundaries in the wrong place. Simple
# hysteresis thresholding matches the data's actual shape instead, and
# self-determines how many cycles there are rather than needing that as
# an input.

def auto_thresholds(signal: np.ndarray, margin: float = 0.3) -> tuple[float, float]:
    """Low/high hysteresis thresholds for a bimodal signal via 2-means clustering.

    ``margin`` (0-0.5) sets how far inside the gap between the two cluster
    centers each threshold sits: 0 puts both thresholds at the midpoint
    (no hysteresis band), 0.5 puts them right at the cluster centers.
    """
    from sklearn.cluster import KMeans

    signal = np.asarray(signal, dtype=float).reshape(-1, 1)
    centers = sorted(KMeans(n_clusters=2, n_init=10, random_state=0).fit(signal).cluster_centers_.ravel())
    low_center, high_center = centers
    gap = high_center - low_center
    return low_center + margin * gap, high_center - margin * gap


def hysteresis_phases(
    signal: np.ndarray,
    low_thresh: float,
    high_thresh: float,
    low: str,
    high: str,
    rising: str | None = None,
    falling: str | None = None,
) -> np.ndarray:
    """Label a signal that rests at two levels (``low``/``high``) joined by
    transitions, via hysteresis: the current state persists until the signal
    crosses the *other* threshold, so brief noise around one threshold can't
    cause flicker.

    With ``rising``/``falling`` given, whichever level (``low``/``high``) is
    last *confirmed* (by fully crossing to the other side's own threshold)
    also decides the label of the "in-between" zone: ``rising`` while last
    confirmed at ``low`` and not yet back down to it, ``falling`` while last
    confirmed at ``high`` and not yet back up to it. Without them, this is a
    plain 2-state (``low``/``high``) thresholding with hysteresis.

    This is intentionally *not* sticky in the sense of "once it starts
    rising, keep calling it rising until it reaches high": a real recording
    can have a partial excursion (e.g. a landing bounce, a hesitation) that
    pokes past ``low_thresh`` without ever completing the climb, and a
    design that commits to ``rising`` on the first crossing would then
    mislabel everything up to the *next real* ascent. Recomputing the label
    from the confirmed state on every sample means an aborted excursion is
    self-correcting: as soon as the signal settles back below
    ``low_thresh``, the label reverts to ``low`` on its own.

    ``low`` and ``high`` may be the *same* label (e.g. session ``176``: both
    the bottom and top of the staircase are ``stand``) - confirmed state is
    tracked as a bool, never by comparing to the label strings, so a
    repeated label can't make two logically distinct states collide.
    """
    signal = np.asarray(signal, dtype=float)
    two_state = rising is None and falling is None
    confirmed_high = signal[0] >= (low_thresh + high_thresh) / 2
    out = np.empty(len(signal), dtype=object)
    for i, v in enumerate(signal):
        if not confirmed_high and v >= high_thresh:
            confirmed_high = True
        elif confirmed_high and v <= low_thresh:
            confirmed_high = False

        if two_state:
            out[i] = high if confirmed_high else low
        elif confirmed_high:
            out[i] = high if v >= high_thresh else falling
        else:
            out[i] = low if v <= low_thresh else rising
    return out


def pad_transition_labels(labels: np.ndarray, low: str, high: str, pad: int) -> np.ndarray:
    """Extend every transition run (any label other than ``low``/``high``) by
    ``pad`` samples on each side, eating into its neighbouring low/high run
    (never into another transition run).

    Widening a transition by loosening ``hysteresis_phases``'s thresholds is
    tempting but unsafe: a session can have a resting plateau that drifts to
    sit slightly closer to the *other* threshold for a while (e.g. a chair
    sit that settles a bit higher on the first rep than later ones), and a
    looser threshold then mistakes that whole flat stretch for an in-progress
    transition since it never completes the climb. Padding a safely-detected
    transition after the fact widens it without that risk.
    """
    labels = np.asarray(labels, dtype=object).copy()
    n = len(labels)
    if n == 0:
        return labels
    change = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    values = labels[starts]
    is_transition = ~np.isin(values, [low, high])
    for i in np.where(is_transition)[0]:
        s, e = starts[i], ends[i]
        lo_bound = starts[i - 1] if i > 0 else 0
        hi_bound = ends[i + 1] if i + 1 < len(starts) else n
        labels[max(s - pad, lo_bound):s] = values[i]
        labels[e:min(e + pad, hi_bound)] = values[i]
    return labels


def label_transitions(labels: np.ndarray) -> np.ndarray:
    """Sample indices where the label changes value."""
    labels = np.asarray(labels)
    return np.where(labels[:-1] != labels[1:])[0] + 1


def labels_to_ids(labels: pd.Series, class_names: list[str]) -> np.ndarray:
    """Map string labels to integer class ids (``-1`` for anything unmapped)."""
    lut = {name: i for i, name in enumerate(class_names)}
    return labels.map(lambda s: lut.get(s, -1)).to_numpy()
