"""歌ステムから音節 (歌詞の 1 文字 = 1 モーラ) の始まりを検出する小さな 1 次元 CNN.

Songle / TextAlive の歌詞タイミングが無い曲 (ファイル入力・未登録曲) では、音符の区切りの手掛かりが無く、
f0 の変化と音量の谷だけでは音節の切れ目を取りこぼす (同じ高さで歌い直す箇所がつながり、リズムが崩れる)。
そこで、Songle の人手修正済みの文字タイミングを教師に、歌ステムのメルスペクトログラム + f0 + 音量から
「ここで文字が始まる」確率を 10 ms ごとに出すモデルを学習した (tools/train_onset.py)。
検出した始まりを歌詞の文字境界の代わりに楽譜モデル (score_model) に渡す。

入力特徴量 (10 ms ごと, RMVPE の時刻格子):
  log メル 48 帯域 (80 Hz〜8 kHz, 曲ごとに標準化) / 相対音高 (曲の中央値からのオクターブ) / 有声フラグ / log 音量 (標準化)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np

log = logging.getLogger(__name__)

HOP_S = 0.01
N_MELS = 48
WEIGHTS = Path(__file__).with_name("onset_model.pt")
_NET = None
_THR = 0.5


def is_available() -> bool:
    if not WEIGHTS.exists():
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def features(voc: np.ndarray, sr: int, t: np.ndarray, f0: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """[51, T] の特徴量. t, f0 (Hz, 無声=0), rms は RMVPE の 10 ms 格子 (長さ T)."""
    hop = int(round(HOP_S * sr))
    S = librosa.feature.melspectrogram(y=voc.astype(np.float32), sr=sr, n_fft=1024, hop_length=hop,
                                       n_mels=N_MELS, fmin=80, fmax=8000)
    L = np.log(S + 1e-6)
    idx = np.clip(np.round(np.asarray(t) / (hop / sr)).astype(int), 0, L.shape[1] - 1)
    L = L[:, idx]
    L = (L - L.mean()) / (L.std() + 1e-9)
    voiced = np.asarray(f0) > 0
    midi = np.where(voiced, librosa.hz_to_midi(np.maximum(f0, 1e-3)), 0.0)
    mref = float(np.median(midi[voiced])) if voiced.any() else 60.0
    mrel = np.where(voiced, (midi - mref) / 12.0, 0.0)
    lr = np.log(np.asarray(rms) + 1e-6)
    lr = (lr - lr.mean()) / (lr.std() + 1e-9)
    return np.vstack([L, mrel[None], voiced[None].astype(float), lr[None]]).astype(np.float32)


def _build_net(ch: int, drop: float = 0.0, gru: int = 0):
    """畳み込み 4 層 (受容野 約 ±0.3 秒)。gru > 0 なら双方向 GRU を重ねてフレーズ全体の文脈 (音節の間隔) を見る."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, cin=N_MELS + 3):
            super().__init__()
            self.f = nn.Sequential(
                nn.Dropout(drop / 3),
                nn.Conv1d(cin, ch, 5, padding=2), nn.ReLU(), nn.Dropout(drop),
                nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.ReLU(), nn.Dropout(drop),
                nn.Conv1d(ch, ch, 5, padding=8, dilation=4), nn.ReLU(), nn.Dropout(drop),
                nn.Conv1d(ch, ch, 3, padding=8, dilation=8), nn.ReLU(), nn.Dropout(drop),
                *([] if gru else [nn.Conv1d(ch, 1, 1)]))
            if gru:
                self.rnn = nn.GRU(ch, gru, batch_first=True, bidirectional=True)
                self.drop = nn.Dropout(drop)
                self.out = nn.Linear(2 * gru, 1)
            self.gru = gru

        def forward(self, x):
            h = self.f(x)
            if not self.gru:
                return h[:, 0]
            h, _ = self.rnn(h.transpose(1, 2))
            return self.out(self.drop(h))[..., 0]

    return Net()


def _load():
    global _NET, _THR
    if _NET is None:
        import torch
        meta = json.loads(WEIGHTS.with_suffix(".json").read_text())
        net = _build_net(int(meta["ch"]), gru=int(meta.get("gru", 0)))
        net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        net.eval()
        _NET, _THR = net, float(meta["thr"])
    return _NET, _THR


def posterior(X: np.ndarray, net=None) -> np.ndarray:
    """特徴量 [51, T] → 各フレームが文字の始まりである確率 [T]."""
    import torch
    net = net or _load()[0]
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(X)[None]))[0].numpy()


def pick_peaks(p: np.ndarray, thr: float, min_dist: int = 6) -> np.ndarray:
    """確率の山 (thr 以上, 互いに min_dist フレーム以上離れる) の時刻 [秒]."""
    pk: list[int] = []
    for i in range(1, len(p) - 1):
        if p[i] >= thr and p[i] >= p[i - 1] and p[i] > p[i + 1]:
            if pk and i - pk[-1] < min_dist:
                if p[i] > p[pk[-1]]:
                    pk[-1] = i
                continue
            pk.append(i)
    return np.asarray(pk, dtype=float) * HOP_S


def detect(voc: np.ndarray, sr: int, t: np.ndarray, f0: np.ndarray, rms: np.ndarray, thr: float | None = None):
    """(文字の始まりの時刻 [秒], 確率 [T]) を返す."""
    net, thr0 = _load()
    p = posterior(features(voc, sr, t, f0, rms), net)
    return pick_peaks(p, thr0 if thr is None else thr), p


def pseudo_chars(onsets: np.ndarray, t: np.ndarray, voiced: np.ndarray, max_len: float = 1.5):
    """検出した文字の始まりから、歌詞の文字区間の代わり [(start, end)] を作る.
    文字は次の始まりか、歌が途切れる (有声が 100 ms 以上切れる) ところで終わる."""
    voiced = np.asarray(voiced, bool)
    n = len(voiced)
    chars = []
    for i, s in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else s + max_len
        a = int(s / HOP_S)
        e = min(nxt, s + max_len)
        # 有声区間の終わりで打ち切る
        f, last_v = a, a
        while f < min(n, int(e / HOP_S)):
            if voiced[f]:
                last_v = f
            elif f - last_v >= 10:
                break
            f += 1
        e = min(e, max(last_v + 1, a + 4) * HOP_S)
        if a < n and voiced[a:min(n, a + 15)].sum() >= 3 and e - s >= 0.04:
            chars.append((float(s), float(e)))
    return chars
