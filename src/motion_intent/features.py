"""Feature extraction for windowed EEG / EMG / motion segments.

All extractors take a batch of windows shaped ``(n_samples, n_channels, n_time)``
and return a 2-D feature matrix ``(n_samples, n_features)``, so they can be
concatenated column-wise before the classifier.

Branches
--------
EEG   Hjorth parameters per band  (:func:`extract_hjorth_fast`)
      CSP spatial filter -> Hjorth (:func:`extract_csp_hjorth`)
      CSP spatial filter -> delta slope (:func:`extract_csp_slope`)
EMG   root-mean-square amplitude   (:func:`extract_rms`)
motion mean marker position        (:func:`extract_mean_position`)
env   task geometry (chair / stair height) (:func:`build_env_features`)
"""
from __future__ import annotations

import numpy as np

from .config import DYNAMIC_CLASSES

# --------------------------------------------------------------------------- #
# static vs. dynamic helper (used to fit the CSP)
# --------------------------------------------------------------------------- #

def is_dynamic(y: np.ndarray) -> np.ndarray:
    """Map class ids to a binary static(0) / dynamic(1) label.

    Parameters
    ----------
    y : (n_samples,) int array of class ids.

    Returns
    -------
    (n_samples,) int array, 1 where the class involves movement.
    """
    return np.isin(y, DYNAMIC_CLASSES).astype(int)


# --------------------------------------------------------------------------- #
# EEG - Hjorth parameters
# --------------------------------------------------------------------------- #

def hjorth_parameters(x: np.ndarray, eps: float = 1e-8):
    """Hjorth activity, mobility and complexity of a 1-D signal ``x`` (time,)."""
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + eps))
    complexity = np.sqrt(var_ddx / (var_dx + eps)) / (mobility + eps)
    return activity, mobility, complexity


def extract_hjorth(X: np.ndarray) -> np.ndarray:
    """Per-channel [activity, mobility] via an explicit loop (reference impl).

    Parameters
    ----------
    X : (n_samples, n_channels, n_time)

    Returns
    -------
    (n_samples, n_channels * 2)
    """
    n_samples, n_ch, _ = X.shape
    feats = np.empty((n_samples, n_ch * 2), dtype=float)
    for i in range(n_samples):
        row = []
        for ch in range(n_ch):
            a, m, _ = hjorth_parameters(X[i, ch])
            row.extend([a, m])
        feats[i] = row
    return feats


def extract_hjorth_fast(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Vectorised [activity, mobility] over the time axis.

    Parameters
    ----------
    X : (n_samples, n_channels, n_time)

    Returns
    -------
    (n_samples, n_channels * 2) -- ``[activity(all ch), mobility(all ch)]``
    """
    var_x = np.var(X, axis=2)                 # (n_samples, n_ch)
    var_dx = np.var(np.diff(X, axis=2), axis=2)

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + eps))
    return np.concatenate([activity, mobility], axis=1)


# --------------------------------------------------------------------------- #
# EEG - CSP spatial filtering
# --------------------------------------------------------------------------- #

def apply_csp_filters(X: np.ndarray, csp) -> np.ndarray:
    """Apply a fitted ``mne.decoding.CSP`` as a spatial filter, keeping time.

    Parameters
    ----------
    X   : (n_samples, n_channels, n_time)
    csp : fitted ``mne.decoding.CSP`` (``filters_`` and ``n_components`` set)

    Returns
    -------
    (n_samples, n_components, n_time)
    """
    W = csp.filters_[: csp.n_components]      # (n_components, n_channels)
    return np.einsum("kc,sct->skt", W, X)


def extract_csp_hjorth(X: np.ndarray, csp) -> np.ndarray:
    """CSP spatial filter, then Hjorth features on the component time-series.

    Returns ``(n_samples, n_components * 2)``.
    """
    return extract_hjorth_fast(apply_csp_filters(X, csp))


def extract_delta_slope(X: np.ndarray, fs: float) -> np.ndarray:
    """Least-squares linear slope of each channel over the window.

    Intended for slow (delta-band) EEG. ``X`` is ``(n_samples, n_channels, n_time)``;
    returns ``(n_samples, n_channels)``.
    """
    n_time = X.shape[2]
    t = np.arange(n_time) / fs
    t = t - t.mean()                          # numerical stability
    denom = np.sum(t ** 2)
    return np.sum(X * t[None, None, :], axis=2) / denom


def extract_csp_slope(X: np.ndarray, csp, fs: float) -> np.ndarray:
    """CSP spatial filter, then per-component linear slope."""
    return extract_delta_slope(apply_csp_filters(X, csp), fs=fs)


def get_hjorth_feature_indices(
    n_csp_models: int,
    n_components: int,
    param_names=("activity", "mobility"),
):
    """Describe each column of a stacked CSP->Hjorth feature matrix.

    Returns a list of dicts ``{index, csp_model, component, param}`` for tracing
    which physical quantity a given feature column corresponds to.
    """
    out, idx = [], 0
    for m in range(n_csp_models):
        for c in range(n_components):
            for pname in param_names:
                out.append(
                    {"index": idx, "csp_model": m, "component": c, "param": pname}
                )
                idx += 1
    return out


# --------------------------------------------------------------------------- #
# EMG
# --------------------------------------------------------------------------- #

def extract_rms(X: np.ndarray) -> np.ndarray:
    """Root-mean-square amplitude per channel.

    ``X`` is ``(n_samples, n_channels, n_time)``; returns ``(n_samples, n_channels)``.
    """
    return np.sqrt(np.mean(X ** 2, axis=2))


# --------------------------------------------------------------------------- #
# Motion capture
# --------------------------------------------------------------------------- #

def extract_mean_position(X: np.ndarray) -> np.ndarray:
    """Mean marker position over the window.

    ``X`` is ``(n_samples, n_channels, n_time)``; returns ``(n_samples, n_channels)``.
    """
    return np.mean(X, axis=2)


# --------------------------------------------------------------------------- #
# Environment (task geometry)
# --------------------------------------------------------------------------- #

def build_env_features(n_samples: int, chair_h: float, stair_h: float) -> np.ndarray:
    """Broadcast the (constant-within-session) chair and stair heights.

    Returns ``(n_samples, 2)``.
    """
    return np.tile(np.array([[chair_h, stair_h]], dtype=float), (n_samples, 1))
