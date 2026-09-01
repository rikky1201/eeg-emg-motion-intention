"""窓化した EEG / EMG / モーション区間からの特徴量抽出.

各抽出関数は ``(n_samples, n_channels, n_time)`` 形状の窓バッチを受け取り，
2 次元の特徴量行列 ``(n_samples, n_features)`` を返す．分類器の前段で列方向に
連結できるようにするため．

ブランチ
--------
EEG     帯域ごとの Hjorth パラメータ (:func:`extract_hjorth_fast`)
EMG     二乗平均平方根 (RMS) 振幅       (:func:`extract_rms`)
motion  マーカー平均位置                (:func:`extract_mean_position`)
env     タスク幾何（椅子高・段差高）    (:func:`build_env_features`)
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# EEG - Hjorth パラメータ
# --------------------------------------------------------------------------- #

def hjorth_parameters(x: np.ndarray, eps: float = 1e-8):
    """1 次元信号 ``x`` (time,) の Hjorth activity / mobility / complexity."""
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
    """明示ループ版．チャネルごとに [activity, mobility] を並べる（参照実装）.

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
    """時間軸に沿ってベクトル化した [activity, mobility].

    Parameters
    ----------
    X : (n_samples, n_channels, n_time)

    Returns
    -------
    (n_samples, n_channels * 2) -- ``[activity(全ch), mobility(全ch)]``
    """
    var_x = np.var(X, axis=2)                 # (n_samples, n_ch)
    var_dx = np.var(np.diff(X, axis=2), axis=2)

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + eps))
    return np.concatenate([activity, mobility], axis=1)


# --------------------------------------------------------------------------- #
# EMG
# --------------------------------------------------------------------------- #

def extract_rms(X: np.ndarray) -> np.ndarray:
    """チャネルごとの二乗平均平方根振幅.

    ``X`` は ``(n_samples, n_channels, n_time)``，戻り値は ``(n_samples, n_channels)``.
    """
    return np.sqrt(np.mean(X ** 2, axis=2))


# --------------------------------------------------------------------------- #
# モーションキャプチャ
# --------------------------------------------------------------------------- #

def extract_mean_position(X: np.ndarray) -> np.ndarray:
    """窓内のマーカー平均位置.

    ``X`` は ``(n_samples, n_channels, n_time)``，戻り値は ``(n_samples, n_channels)``.
    """
    return np.mean(X, axis=2)


# --------------------------------------------------------------------------- #
# 環境（タスク幾何）
# --------------------------------------------------------------------------- #

def build_env_features(n_samples: int, chair_h: float, stair_h: float) -> np.ndarray:
    """セッション内で一定の椅子高・段差高をブロードキャストする.

    戻り値は ``(n_samples, 2)``.
    """
    return np.tile(np.array([[chair_h, stair_h]], dtype=float), (n_samples, 1))
