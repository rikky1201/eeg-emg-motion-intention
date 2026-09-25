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

Some recordings never got a working LSL clock-offset handshake for one or
more streams (network hiccup during that particular take): ``pyxdf``'s
built-in ``synchronize_clocks`` then gives up on that stream and leaves it on
its own device-local clock, so it ends up hundreds of thousands of seconds
away from the others once merged. Each physical stream's offset from the
common clock is otherwise stable for an entire recording day (same host,
same LSL outlet, not rebooted between takes) with sub-second scatter, so
:func:`pool_fallback_offsets` recovers a per-stream-name reference offset from
whichever recordings *did* sync cleanly, and :func:`xdf_to_dataframe` falls
back to it for streams whose own measurement looks unusable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# LSL channel labels that carry no signal
META_LABELS = {"Timestamp", "Counter", "Interpolate", "HardwareMarker", "Markers"}

# A stream's own clock-offset measurement is considered for trust only if
# there are at least this many samples with less than this much scatter [s].
MIN_OFFSET_SAMPLES = 3
MAX_OFFSET_STD_SEC = 1.0
# Even a low-scatter measurement is rejected if it disagrees with the pooled
# cross-recording reference by more than this [s]: a failed handshake can
# report a perfectly self-consistent (std == 0) but wrong value (e.g. a stuck
# 0.0) instead of raising, so internal consistency alone isn't enough.
MAX_OFFSET_DEVIATION_SEC = 2.0


def _offset_samples(stream: dict) -> np.ndarray:
    """Every clock-offset value ``pyxdf`` recorded for this stream (in-body + footer)."""
    vals = [float(v) for v in stream.get("clock_values", [])]
    off_info = stream.get("footer", {}).get("info", {}).get("clock_offsets", [{}])[0]
    for entry in (off_info.get("offset", []) if off_info else []):
        vals.append(float(entry["value"][0]))
    return np.asarray(vals, dtype=float)


def _offset_candidate(stream: dict) -> float | None:
    """This stream's own median clock offset, or ``None`` if there are too few
    samples to say anything (no scatter/plausibility check)."""
    vals = _offset_samples(stream)
    if len(vals) < MIN_OFFSET_SAMPLES:
        return None
    return float(np.median(vals))


def _trusted_offset(stream: dict, reference: float | None = None) -> float | None:
    """This stream's own clock offset, or ``None`` if the measurement looks
    unusable: too few samples, too much internal scatter, or (when
    ``reference`` - typically a pooled cross-recording value - is given)
    implausibly far from it. The last check matters because a failed
    handshake can report a perfectly self-consistent wrong value (e.g. stuck
    at 0.0) rather than noisy garbage.
    """
    vals = _offset_samples(stream)
    if len(vals) < MIN_OFFSET_SAMPLES or np.std(vals) > MAX_OFFSET_STD_SEC:
        return None
    median = float(np.median(vals))
    if reference is not None and abs(median - reference) > MAX_OFFSET_DEVIATION_SEC:
        return None
    return median


def pool_fallback_offsets(xdf_dir: str | Path) -> dict[str, float]:
    """Median clock offset per LSL stream name, pooled across every ``*.xdf``
    file in ``xdf_dir``.

    Uses each recording's own offset candidate without a plausibility check,
    but the pooled median is robust to the rare recording where a stream's
    handshake failed and reported an implausible (e.g. ~0) value instead of
    its real, physically-stable-for-the-day offset, as long as it's a
    minority of the recordings for that stream.

    Pass the result as ``fallback_offsets`` to :func:`xdf_to_dataframe` /
    :func:`export_session_csv` to repair recordings where the offset
    handshake failed for a stream.
    """
    import pyxdf

    pooled: dict[str, list[float]] = {}
    for xdf_path in sorted(Path(xdf_dir).glob("*.xdf")):
        streams, _ = pyxdf.load_xdf(str(xdf_path), synchronize_clocks=False, dejitter_timestamps=False)
        for s in streams:
            offset = _offset_candidate(s)
            if offset is not None:
                pooled.setdefault(s["info"]["name"][0], []).append(offset)
    return {name: float(np.median(vals)) for name, vals in pooled.items()}


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


def _drop_duplicate_streams(streams: list[dict]) -> list[dict]:
    """Keep only the largest stream for each (name, type) pair.

    Some recordings ended up with the same physical device broadcasting on
    two separate LSL outlets under the identical name (e.g. subject lee's
    ``EMG_Stream`` appearing twice with near-identical timestamps and
    values - a duplicate transmission, not two different signals). Left in,
    both would produce the same ``<type>_<label>`` columns and the outer
    join would fail with "columns overlap but no suffix specified".
    """
    best: dict[tuple[str, str], dict] = {}
    for s in streams:
        key = (s["info"]["name"][0], s["info"].get("type", [""])[0])
        if key not in best or len(s["time_series"]) > len(best[key]["time_series"]):
            best[key] = s
    return list(best.values())


def xdf_to_dataframe(xdf_path: str | Path, fallback_offsets: dict[str, float] | None = None) -> pd.DataFrame:
    """Merge all non-empty streams of an XDF file onto one clock-corrected time base.

    Returns a wide DataFrame; each stream contributes ``<type>_<label>`` columns.
    Streams are outer-joined on timestamp, so cross-rate gaps appear as NaN
    (resolved later by interpolation / resampling in
    :mod:`motion_intent.preprocessing`).

    Clock offsets are resolved ourselves (rather than via ``pyxdf``'s
    ``synchronize_clocks``) so a stream whose handshake failed for this one
    recording can be repaired with ``fallback_offsets`` (see
    :func:`pool_fallback_offsets`) instead of silently staying on its own
    device-local clock.
    """
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path), synchronize_clocks=False, dejitter_timestamps=True)
    streams = _drop_duplicate_streams(streams)
    merged: pd.DataFrame | None = None

    for s in streams:
        stype = s["info"].get("type", [""])[0]
        name = s["info"]["name"][0]
        data = np.asarray(s["time_series"])
        ts = np.asarray(s["time_stamps"])
        if data.size == 0 or ts.size == 0:
            continue

        reference = fallback_offsets.get(name) if fallback_offsets else None
        offset = _trusted_offset(s, reference=reference)
        if offset is None:
            if fallback_offsets is None or name not in fallback_offsets:
                raise ValueError(
                    f"{xdf_path}: stream {name!r} has no trustworthy clock offset "
                    f"(handshake likely failed for this recording) - pass "
                    f"fallback_offsets=pool_fallback_offsets(xdf_dir) computed "
                    f"from this subject's other recordings"
                )
            offset = fallback_offsets[name]
        ts = ts + offset

        labels = channel_labels(s)
        keep = [i for i, lab in enumerate(labels) if lab not in META_LABELS]
        data = data[:, keep]
        cols = [f"{stype}_{labels[i]}" for i in keep]
        df = pd.DataFrame(data, index=ts, columns=cols)
        df.index.name = "Timestamp"
        df = df[~df.index.duplicated(keep="first")]
        merged = df if merged is None else merged.join(df, how="outer")

    if merged is None:
        raise ValueError(f"no usable streams in {xdf_path}")

    merged = merged.sort_index()
    merged["t_sec"] = merged.index - merged.index[0]
    return merged.reset_index()


def export_session_csv(
    xdf_path: str | Path, out_dir: str | Path, fallback_offsets: dict[str, float] | None = None
) -> Path:
    """Convert one XDF file to ``<out_dir>/<same-stem>.csv`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(xdf_path).stem + ".csv")
    xdf_to_dataframe(xdf_path, fallback_offsets=fallback_offsets).to_csv(out_path, index=False)
    return out_path
