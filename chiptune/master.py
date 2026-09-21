"""マスタリング: 高域の補正、ソフトリミッタ、ラウドネス正規化、フェードアウト。"""
from __future__ import annotations

import numpy as np
from scipy import signal


def high_shelf(x: np.ndarray, sr: int, freq: float = 4000.0, gain_db: float = 3.0) -> np.ndarray:
    """簡易ハイシェルフ: freq 以上を gain_db 持ち上げる (2 次)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cosw = np.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cosw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
    b2 = A * ((A + 1) + (A - 1) * cosw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cosw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cosw)
    a2 = (A + 1) - (A - 1) * cosw - 2 * np.sqrt(A) * alpha
    return signal.lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], x).astype(np.float32)


def soft_limit(x: np.ndarray, knee: float = 0.7) -> np.ndarray:
    a = np.abs(x)
    over = a > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return y


def normalize_loudness(x: np.ndarray, target_db: float = -14.0) -> np.ndarray:
    """1 秒 RMS の上位 30% の平均を target_db に合わせる (サビ基準のラウドネス)."""
    n = len(x) // 44100 * 44100
    if n < 44100:
        return x
    r = np.sqrt((x[:n].reshape(-1, 44100) ** 2).mean(axis=1))
    loud = np.sort(r)[int(len(r) * 0.7):]
    cur = 20 * np.log10(loud.mean() + 1e-9)
    return x * 10 ** ((target_db - cur) / 20)


def fade_out(x: np.ndarray, sr: int, seconds: float = 2.0, tail_db: float = -30.0) -> np.ndarray:
    """曲末が唐突なら (最後 1 秒がまだ鳴っていれば) フェードアウトを掛ける."""
    n = int(seconds * sr)
    if len(x) <= n:
        return x
    tail = x[-sr:]
    if 20 * np.log10(np.sqrt((tail ** 2).mean()) + 1e-9) > tail_db:
        x = x.copy()
        x[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return x


def finalize(mix: np.ndarray, sr: int, crunch: bool = True) -> np.ndarray:
    y = high_shelf(mix, sr, 4000.0, 3.0)
    y = normalize_loudness(y, -14.0)
    y = soft_limit(y, 0.75)
    y = fade_out(y, sr)
    peak = np.abs(y).max() + 1e-9
    if peak > 0.95:
        y = y / peak * 0.95
    if crunch:
        y = np.round(y * 127) / 127
    return y.astype(np.float32)
