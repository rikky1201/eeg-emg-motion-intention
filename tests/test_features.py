"""Shape / sanity checks for the feature extractors on synthetic arrays.

No experiment data is needed; these guard the array contracts the notebooks rely
on. Run with ``pytest``.
"""
import numpy as np
import pytest

from motion_intent import features, windowing
from motion_intent.config import N_CLASSES

rng = np.random.default_rng(0)


@pytest.fixture
def eeg_windows():
    # (n_samples, n_channels, n_time)
    return rng.standard_normal((32, 11, 500))


def test_hjorth_fast_shape(eeg_windows):
    out = features.extract_hjorth_fast(eeg_windows)
    assert out.shape == (32, 11 * 2)
    assert np.isfinite(out).all()


def test_hjorth_fast_matches_reference(eeg_windows):
    fast = features.extract_hjorth_fast(eeg_windows)
    ref = features.extract_hjorth(eeg_windows)
    # reference interleaves [act, mob] per channel; fast stacks [act..., mob...]
    n_ch = eeg_windows.shape[1]
    ref_stacked = np.concatenate([ref[:, 0::2], ref[:, 1::2]], axis=1)
    assert np.allclose(fast, ref_stacked, rtol=1e-6)


def test_rms_shape():
    emg = rng.standard_normal((16, 8, 200))
    out = features.extract_rms(emg)
    assert out.shape == (16, 8)
    assert (out >= 0).all()


def test_mean_position_shape():
    motion = rng.standard_normal((16, 12, 200))
    assert features.extract_mean_position(motion).shape == (16, 12)


def test_env_features():
    out = features.build_env_features(10, 0.6, 0.3)
    assert out.shape == (10, 2)
    assert (out[:, 0] == 0.6).all() and (out[:, 1] == 0.3).all()


def test_make_windows_labels_centre():
    n_time, win, step = 100, 10, 5
    X = np.arange(n_time)[:, None] * np.ones((1, 3))
    y = np.arange(n_time)
    Xw, yw = windowing.make_windows(X, y, [0, 1, 2], win=win, step=step)
    assert Xw.shape == (yw.shape[0], 3, win)
    assert yw[0] == win // 2


def test_transition_offset_refs():
    # labels: 0 for 100 samples, then 1 for 100 -> single transition at idx 100
    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    refs, y = windowing.transition_offset_refs(labels, offset_samples=-30,
                                               n_samples=len(labels), min_ref=50)
    assert refs.tolist() == [70]          # 100 + (-30)
    assert y.tolist() == [1]              # post-transition class
    # ref below min_ref is dropped
    refs, y = windowing.transition_offset_refs(labels, offset_samples=-60,
                                               n_samples=len(labels), min_ref=50)
    assert refs.size == 0 and y.size == 0


def test_split_by_time_block_is_chronological():
    import pandas as pd
    df = pd.DataFrame({"t_sec": np.linspace(0, 10, 1000), "v": rng.standard_normal(1000)})
    tr, va, te = windowing.split_by_time_block(df, train_ratio=0.6, val_ratio=0.2)
    assert tr["t_sec"].max() <= va["t_sec"].min() <= te["t_sec"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_class_count():
    assert N_CLASSES == 7
