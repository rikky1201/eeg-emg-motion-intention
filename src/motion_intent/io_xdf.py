"""Load an XDF recording and merge its LSL streams onto one time base.

A session file contains several streams captured together:

===========================  ===================================
``EmotivDataStream-EEG``      32-channel EEG + device markers
``EMG_Stream``               8-channel surface EMG
``OptiTrack_BiomechIDs``     motion-capture marker coordinates
===========================  ===================================

:func:`xdf_to_dataframe` writes one wide CSV per session, indexed by the shared
LSL timestamp, with columns prefixed by stream type (``EEG_``, ``EMG_``,
``Markers_``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# LSL channel labels that carry no signal
META_LABELS = {"Timestamp", "Counter", "Interpolate", "HardwareMarker", "Markers"}


def channel_labels(stream: dict) -> list[str]:
    """Pull channel names from an XDF stream's metadata, falling back to ``chN``."""
    n = int(stream["info"].get("channel_count", ["0"])[0])
    try:
        chans = (
            stream["info"].get("desc", [{}])[0]
            .get("channels", [{}])[0]
            .get("channel", [])
        )
        return [
            (chans[i].get("label", [""])[0] or f"ch{i + 1}") if i < len(chans) else f"ch{i + 1}"
            for i in range(n)
        ]
    except Exception:
        return [f"ch{i + 1}" for i in range(n)]


def summarize_xdf(xdf_path: str | Path) -> None:
    """Print a one-line summary (name, type, shape, rate) for every stream."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    print(f"{Path(xdf_path).name}: {len(streams)} streams")
    for s in streams:
        info = s["info"]
        data = np.asarray(s["time_series"])
        print(
            f"  {info.get('type', [''])[0]:24s} "
            f"name={info.get('name', [''])[0]:26s} "
            f"shape={data.shape} "
            f"srate={info.get('nominal_srate', ['?'])[0]}"
        )


def xdf_to_dataframe(xdf_path: str | Path) -> pd.DataFrame:
    """Merge all non-empty streams of an XDF file on their LSL ``Timestamp``.

    Returns a wide DataFrame; each stream contributes ``<type>_<label>`` columns.
    Streams are outer-joined on timestamp, so cross-rate gaps appear as NaN
    (resolved later by interpolation / resampling in
    :mod:`motion_intent.preprocessing`).
    """
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    merged: pd.DataFrame | None = None

    for s in streams:
        stype = s["info"].get("type", [""])[0]
        data = np.asarray(s["time_series"])
        ts = np.asarray(s["time_stamps"])
        if data.size == 0 or ts.size == 0:
            continue

        cols = [f"{stype}_{lab}" for lab in channel_labels(s)]
        df = pd.DataFrame(data, index=ts, columns=cols)
        df.index.name = "Timestamp"
        df = df[~df.index.duplicated(keep="first")]
        merged = df if merged is None else merged.join(df, how="outer")

    if merged is None:
        raise ValueError(f"no usable streams in {xdf_path}")

    merged = merged.sort_index()
    merged["t_sec"] = merged.index - merged.index[0]
    return merged.reset_index()


def export_session_csv(xdf_path: str | Path, out_dir: str | Path) -> Path:
    """Convert one XDF file to ``<out_dir>/<same-stem>.csv`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(xdf_path).stem + ".csv")
    xdf_to_dataframe(xdf_path).to_csv(out_path, index=False)
    return out_path
