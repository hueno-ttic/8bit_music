"""ファイルで入力した曲を Songle の登録曲と照合する (曲名で検索 → 音で照合 → 時間のずれを求める).

歌詞タイミングが無いと、音節の区切りを歌の音だけから当てることになり、同じ高さの歌い直しの多くを取りこぼす
(楽譜の分かっている評価曲で、音符の正解率が Songle あり 85% に対しファイルのみ 58%)。
ファイルでも曲名が分かれば、Songle の人手修正済みの歌詞タイミングと拍を使える。
  1. 曲名で Songle を検索し、歌詞タイミングのある候補を選ぶ (長さが大きく違うものは除く)
  2. 候補の元動画 (YouTube) の音声を取得し、音の立ち上がりの包絡の相互相関で、同じ曲かと時間のずれを求める
  3. 相関がはっきり高いときだけ採用する (違う曲・別アレンジでは使わない)
曲名は利用者が入力したときだけ外部 (Songle の検索, YouTube) に送る。
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

log = logging.getLogger(__name__)

SR = 11025
HOP = 110            # 約 10 ms
MAX_LAG = 60.0       # 秒: 手元の音声と元動画の頭のずれの上限 (MV の前置きなど)
MIN_SCORE = 0.25     # 採用する正規化相互相関の下限 (違う曲では 0.1 未満, 同じ音源では 0.4 以上だった)


@dataclass
class Match:
    url: str
    title: str
    offset: float      # 手元の音声の時刻 = 元動画の時刻 + offset
    score: float


def _envelope(y: np.ndarray, sr: int) -> np.ndarray:
    x = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else y.astype(np.float32)
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP)
    env = env - np.convolve(env, np.ones(51) / 51, mode="same")      # 緩やかな変化を除く
    return (env - env.mean()) / (env.std() + 1e-9)


def align(y_file: np.ndarray, y_ref: np.ndarray, sr: int) -> tuple[float, float]:
    """(offset 秒, 正規化相互相関). offset > 0 なら手元の音声が元動画より遅れている."""
    a, b = _envelope(y_file, sr), _envelope(y_ref, sr)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    maxlag = int(MAX_LAG * SR / HOP)
    lags = np.concatenate([np.arange(0, maxlag + 1), np.arange(-maxlag, 0)])
    vals = np.concatenate([cc[:maxlag + 1], cc[-maxlag:]])
    k = int(np.argmax(vals))
    lag = int(lags[k])
    # 重なり部分で正規化した相関
    if lag >= 0:
        aa, bb = a[lag:], b[:len(a) - lag]
    else:
        aa, bb = a[:len(b) + lag], b[-lag:]
    m = min(len(aa), len(bb))
    score = float(np.dot(aa[:m], bb[:m]) / (np.linalg.norm(aa[:m]) * np.linalg.norm(bb[:m]) + 1e-9)) if m > 100 else 0.0
    return lag * HOP / SR, score


def find(y: np.ndarray, sr: int, title: str, progress=None, max_candidates: int = 4) -> Match | None:
    """曲名 title で Songle を検索し、手元の音声 y と同じ曲で歌詞タイミングのある候補を返す (無ければ None)."""
    from . import songle as SG
    from .audio_io import load_mono
    from .youtube import download_audio

    title = (title or "").strip()
    if not title:
        return None
    if progress:
        progress(f"Songle で「{title}」を探しています…")
    dur = len(y) / sr
    # 入力全体と、空白で区切った各語 (曲名だけ・作者だけ) で検索する (Songle の検索は語の組み合わせで結果が大きく変わる)
    queries = [title] + [w for w in title.split() if len(w) >= 2 and w != title][:3]
    cands, seen = [], set()
    for q in queries:
        for c in SG.search(q):
            if c["url"] in seen or not ("youtube.com" in c["url"] or "youtu.be" in c["url"]):
                continue
            seen.add(c["url"])
            if not c["duration"] or abs(c["duration"] - dur) <= max(30.0, 0.2 * dur):
                cands.append(c)
    best = None
    tried = 0
    for c in cands:
        if tried >= max_candidates:
            break
        if not SG.lyrics_chars(c["url"]):
            continue
        tried += 1
        try:
            with tempfile.TemporaryDirectory() as d:
                src, _ = download_audio(c["url"], Path(d))
                ref = load_mono(src, sr)
        except Exception as e:  # noqa: BLE001
            log.info("候補 %s の音声を取得できませんでした: %s", c["url"], e)
            continue
        off, score = align(y, ref, sr)
        log.info("Songle 候補 %s (%s): ずれ %+.2f 秒, 相関 %.2f", c["title"], c["url"], off, score)
        if best is None or score > best.score:
            best = Match(c["url"], c["title"], off, score)
        if score >= 2 * MIN_SCORE:
            break
    if best is None or best.score < MIN_SCORE:
        log.info("Songle に一致する曲が見つかりませんでした")
        return None
    if progress:
        progress(f"Songle の「{best.title}」と一致しました (ずれ {best.offset:+.2f} 秒)。歌詞タイミングと拍を使います")
    return best
