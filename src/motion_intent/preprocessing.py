"""Signal preprocessing: marker-axis merge, resampling, band-pass, EEG cleaning.

The heavy EEG artefact-removal step (ICA + ICLabel + ASR) is wrapped in
:func:`clean_eeg`, which is a thin adapter over ``mne`` / ``mne_icalabel`` /
``asrpy``. Those imports are deferred so the rest of the module works without a
full MNE stack installed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt, lfilter

from .config import EEG_BANDS

# --------------------------------------------------------------------------- #
# marker X/Z axis merge
# --------------------------------------------------------------------------- #

def merge_xz_all(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the X and Z ground-plane axes of each motion marker into ``*_XZ``.

    Only one of the X / Z streams is populated at a time (the capture volume is
    rotated per session), so :meth:`~pandas.Series.combine_first` recovers a
    single continuous trace. Velocity / acceleration / lag suffixes are kept.
    """
    df = df.copy()
    pattern = re.compile(r"(Markers_.+)_([XZ])(.*)")

    pairs: dict[str, dict[str, str]] = {}
    for c in df.columns:
        m = pattern.match(c)
        if m:
            base, axis, suffix = m.groups()
            pairs.setdefault(base + suffix, {})[axis] = c

    for key, axes in pairs.items():
        if "X" in axes and "Z" in axes:
            if "_vel" in key:
                new_col = key.replace("_vel", "_XZ_vel")
            elif "_acc" in key:
                new_col = key.replace("_acc", "_XZ_acc")
            else:
                new_col = key + "_XZ"
            df[new_col] = df[axes["X"]].combine_first(df[axes["Z"]])
    return df


def drop_single_axis_marker_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the original ``*_X`` / ``*_Z`` marker columns after :func:`merge_xz_all`."""
    cols = [
        c for c in df.columns
        if ("_X" in c or "_Z" in c) and "_XZ" not in c and c.startswith("Markers_")
    ]
    return df.drop(columns=cols)


# --------------------------------------------------------------------------- #
# sampling-rate helpers
# --------------------------------------------------------------------------- #

def estimate_fs_from_time(t_sec: np.ndarray) -> float:
    """Estimate the sampling rate [Hz] from a (possibly NaN-holed) time column."""
    t = np.asarray(t_sec, dtype=float)
    t = t[~np.isnan(t)]
    dt = np.diff(t)
    dt = dt[dt > 0]
    return 1.0 / np.median(dt)


def decimate_to(df: pd.DataFrame, cols, q: int, t_col: str = "t_sec") -> pd.DataFrame:
    """Anti-alias downsample ``cols`` by integer factor ``q`` (FIR, causal).

    NaN gaps are linearly interpolated first. The time column is sub-sampled to
    match. Returns a new DataFrame with ``cols`` + ``t_col``.
    """
    filled = df[cols].interpolate(method="linear", limit_direction="forward")
    x = decimate(filled.to_numpy(), q=q, axis=0, ftype="fir", zero_phase=False)
    t = df[t_col].to_numpy()[::q][: len(x)]
    out = pd.DataFrame(x, columns=list(cols))
    out[t_col] = t
    return out


# --------------------------------------------------------------------------- #
# band-pass filtering
# --------------------------------------------------------------------------- #

def bandpass_by_session(
    df: pd.DataFrame,
    use_cols: list[str],
    session_col: str,
    fs: float,
    band: tuple[float, float],
    order: int = 2,
) -> pd.DataFrame:
    """Causal Butterworth band-pass applied independently per session.

    Filtering per session avoids ringing across recording boundaries. Only
    ``use_cols`` are modified; every other column is passed through untouched.
    """
    low, high = band
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")

    out = df.copy()
    for _, df_sess in df.groupby(session_col):
        idx = df_sess.index
        for col in use_cols:
            out.loc[idx, col] = lfilter(b, a, df_sess[col].to_numpy())
    return out


def append_band_columns(
    df: pd.DataFrame,
    use_cols: list[str],
    session_col: str,
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    order: int = 2,
    drop_original: bool = True,
) -> pd.DataFrame:
    """Add one ``<col>_<band>`` column per (channel, band) pair.

    ``bands`` defaults to :data:`motion_intent.config.EEG_BANDS`
    (delta / alpha / beta).
    """
    bands = bands or EEG_BANDS
    out = df.copy()
    for band_name, band_range in bands.items():
        df_band = bandpass_by_session(df, use_cols, session_col, fs, band_range, order)
        for col in use_cols:
            out[f"{col}_{band_name}"] = df_band[col]
    if drop_original:
        out = out.drop(columns=use_cols)
    return out


def lowpass(
    df: pd.DataFrame,
    cols: list[str],
    fs: float,
    cutoff_hz: float,
    order: int = 4,
    zero_phase: bool = True,
) -> pd.DataFrame:
    """Butterworth low-pass for motion-capture channels (default zero-phase)."""
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        raise ValueError("cutoff_hz must be < fs / 2")
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    apply = filtfilt if zero_phase else (lambda bb, aa, x: lfilter(bb, aa, x))

    out = df.copy()
    for col in cols:
        out[col] = apply(b, a, df[col].to_numpy())
    return out


# --------------------------------------------------------------------------- #
# EEG artefact cleaning (ICA + ICLabel + ASR)
# --------------------------------------------------------------------------- #

def extract_static_segments(
    df: pd.DataFrame,
    label_col: str,
    static_labels=("sit", "stand"),
    fs: float = 1000.0,
    min_duration_sec: float = 1.0,
    trim_sec: float = 0.3,
) -> pd.DataFrame | None:
    """Return the concatenation of stable static-posture segments.

    Used to build a clean calibration set for ICA / ASR (no movement artefacts).
    Segments shorter than ``min_duration_sec`` are discarded; ``trim_sec`` is
    removed from both ends of every kept segment.
    """
    mask = df[label_col].isin(static_labels).to_numpy()
    if not mask.any():
        return None

    trim = int(trim_sec * fs)
    min_len = int(min_duration_sec * fs)

    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]

    keep = []
    for s, e in zip(starts, ends):
        s, e = s + trim, e - trim
        if e - s >= min_len:
            keep.append(df.iloc[s:e])
    if not keep:
        return None
    return pd.concat(keep, ignore_index=True)


def clean_eeg(
    df: pd.DataFrame,
    eeg_cols: list[str],
    fs: float = 1000.0,
    df_calib: pd.DataFrame | None = None,
    n_components: int = 30,
    asr_cutoff: float = 10.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Remove non-brain components (ICA + ICLabel) and burst artefacts (ASR).

    ``df`` holds EEG in microvolts. If ``df_calib`` is given (e.g. the output of
    :func:`extract_static_segments`) ICA / ASR are fitted on it and applied to
    ``df``; otherwise they are fitted on ``df`` itself.

    Returns a copy of ``df`` with ``eeg_cols`` replaced by the cleaned signal.
    Requires ``mne``, ``mne_icalabel`` and ``asrpy``.
    """
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    import asrpy

    info = mne.create_info(ch_names=list(eeg_cols), sfreq=fs, ch_types="eeg")

    fit_src = df_calib if df_calib is not None else df
    raw_fit = mne.io.RawArray(fit_src[eeg_cols].to_numpy().T * 1e-6, info)
    raw_fit.set_montage("standard_1020", on_missing="ignore")
    raw_fit.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", phase="minimum")

    ica = ICA(n_components=n_components, method="fastica", random_state=random_state)
    ica.fit(raw_fit)
    labels = label_components(raw_fit, ica, method="iclabel")["labels"]
    ica.exclude = [i for i, lab in enumerate(labels) if lab != "brain"]

    asr = asrpy.ASR(sfreq=fs, cutoff=asr_cutoff)
    asr.fit(ica.apply(raw_fit.copy()))

    raw = mne.io.RawArray(df[eeg_cols].to_numpy().T * 1e-6, info)
    raw.set_montage("standard_1020", on_missing="ignore")
    raw = ica.apply(raw)
    raw = asr.transform(raw)

    out = df.copy()
    out[eeg_cols] = raw.get_data().T * 1e6
    return out
