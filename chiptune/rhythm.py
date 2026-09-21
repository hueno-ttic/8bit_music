"""リズム: テンポの倍/半分の補正、小節頭 (ダウンビート) の推定、ドラムパターンの生成。

ドラムは「ヒットを拾う」のではなく、ドラムステムの帯域ごとの活動を小節単位で見て、
定番パターン (4 つ打ち / 8 ビート / ハーフタイム / 16 ビート) から一番近いものを選んで格子上に打ち込む。
"""
from __future__ import annotations

import librosa
import numpy as np

from .analysis import Hit

# パターン: 16 分 16 個 (1 小節) の (kick, snare, hat) 各ステップの強さ (0..1)
PATTERNS = {
    "rock8": {
        "kick":  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    },
    "rock8_kick2": {
        "kick":  [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    },
    "four": {
        "kick":  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat":   [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    },
    "beat16": {
        "kick":  [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hat":   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    "half": {
        "kick":  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "snare": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "hat":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    },
    "hat_only": {
        "kick":  [0] * 16,
        "snare": [0] * 16,
        "hat":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    },
}
BANDS = {"kick": (20, 150), "snare": (150, 2500), "hat": (5000, 11000)}


def band_activity(drm: np.ndarray, sr: int, cells: np.ndarray) -> dict[str, np.ndarray]:
    """16 分セルごとの帯域別の立ち上がり強さ (0..1 に正規化)."""
    S = librosa.feature.melspectrogram(y=drm, sr=sr, hop_length=256, n_mels=64, fmax=sr / 2)
    S_db = np.maximum(librosa.power_to_db(S, ref=np.max), -70.0)
    mf = librosa.mel_frequencies(n_mels=64, fmax=sr / 2)
    out = {}
    fr = np.clip(np.floor(cells * sr / 256).astype(int), 0, S_db.shape[1])
    for name, (lo, hi) in BANDS.items():
        band = S_db[(mf >= lo) & (mf < hi)].mean(axis=0)
        flux = np.maximum(0, np.diff(band, prepend=band[0]))
        flux = flux / (np.percentile(flux, 99) + 1e-9)
        act = np.zeros(len(cells) - 1)
        for i in range(len(cells) - 1):
            a, b = fr[i], max(fr[i + 1], fr[i] + 1)
            act[i] = flux[a:b].max() if b > a else 0.0
        out[name] = np.clip(act, 0, 1)
    return out


def fix_tempo_octave(tempo: float, cells: np.ndarray, act: dict[str, np.ndarray]) -> int:
    """ビート検出の倍/半分を、キックとスネアの周期から判定. 戻り値: 1 (そのまま) / 2 (倍) / 0.5 相当は -1."""
    kick = act["kick"]; snare = act["snare"]
    # 1 拍 = 4 セル。キックが「2 セルごと」に強ければ実テンポは倍
    def periodic_strength(x, period):
        n = len(x) // period * period
        if n < period * 8:
            return 0.0
        m = x[:n].reshape(-1, period).mean(axis=0)
        return float(m.max() - np.median(m))
    if tempo < 105 and len(kick) >= 64:
        # キックの自己相関: 2 セル周期 (= 8 分) が 4 セル周期 (= 拍) と同じくらい強ければ、実テンポは倍
        k = kick - kick.mean()
        ac = np.correlate(k, k, mode="full")[len(k) - 1:]
        ac = ac / (ac[0] + 1e-9)
        if ac[2] > 0.3 and ac[2] >= 0.9 * ac[4]:
            return 2
    if tempo > 170:
        if periodic_strength(kick, 8) > 1.3 * periodic_strength(kick, 4):
            return -1
    return 1


def estimate_downbeat_phase(act: dict[str, np.ndarray], beats_per_bar: int = 4) -> int:
    """小節頭の位相 (0..3 拍) を、スネアが 2・4 拍目に来る仮定で推定."""
    kick, snare = act["kick"], act["snare"]
    n_beats = len(kick) // 4
    best, best_score = 0, -1e9
    for phase in range(beats_per_bar):
        score = 0.0
        for b in range(n_beats):
            pos = (b - phase) % beats_per_bar
            cell = b * 4
            if cell >= len(kick):
                break
            if pos in (1, 3):
                score += snare[cell] - 0.5 * kick[cell]
            else:
                score += kick[cell] - 0.5 * snare[cell]
        if score > best_score:
            best_score, best = score, phase
    return best


def choose_patterns(act: dict[str, np.ndarray], bar_starts: list[int], intensity: np.ndarray) -> list[str | None]:
    """小節ごとに一番近いパターンを選ぶ. ドラムがほぼ無い小節は None."""
    names = list(PATTERNS)
    # 帯域ごとの「はっきりしたヒット」のしきい値 (曲全体の分布から)
    thr = {k: max(0.15, float(np.percentile(act[k][act[k] > 0], 60))) if np.any(act[k] > 0) else 1.0 for k in act}
    out = []
    for bi, c0 in enumerate(bar_starts):
        seg = {k: act[k][c0:c0 + 16] for k in act}
        if any(len(v) < 16 for v in seg.values()):
            out.append(None); continue
        # 小節内で正規化し、しきい値以上を「ヒットあり」とみなす
        hit = {k: (seg[k] >= thr[k]).astype(float) for k in seg}
        if hit["kick"].sum() + hit["snare"].sum() + 0.5 * hit["hat"].sum() < 1.5:
            out.append(None); continue
        best, best_s = None, -1e9
        for name in names:
            p = PATTERNS[name]
            s = 0.0
            for k in ("kick", "snare", "hat"):
                pv = np.array(p[k], float); av = hit[k]
                w = 1.0 if k != "hat" else 0.4
                match = np.dot(pv, av)             # パターンにもあり実際にもある
                miss = np.dot(pv, 1 - av)          # パターンにあるが実際には無い
                extra = np.dot(1 - pv, av)         # 実際にあるがパターンに無い
                s += w * (match - 0.5 * miss - 0.4 * extra)
            if best_s < s:
                best, best_s = name, s
        out.append(best)
    return out


def render_hits(pattern_names: list[str | None], bar_starts: list[int], cells: np.ndarray, act: dict[str, np.ndarray],
                intensity: np.ndarray) -> list[Hit]:
    """パターンを格子に打ち込む. 強さは小節の強度に応じて、ハイハットは静かな小節で間引く."""
    hits: list[Hit] = []
    for bi, (name, c0) in enumerate(zip(pattern_names, bar_starts)):
        if name is None:
            continue
        p = PATTERNS[name]
        inten = float(intensity[bi]) if bi < len(intensity) else 1.0
        for step in range(16):
            ci = c0 + step
            if ci >= len(cells) - 1:
                break
            t = float(cells[ci])
            if p["kick"][step]:
                hits.append(Hit(t, "kick", 0.9 + 0.1 * inten))
            if p["snare"][step]:
                hits.append(Hit(t, "snare", 0.85 + 0.15 * inten))
            if p["hat"][step] and (inten >= 0.35 or step % 4 == 0):
                hits.append(Hit(t, "hat", 0.5 + 0.4 * inten * (1.0 if step % 4 == 0 else 0.75)))
    # 小節の最後にフィル (8 小節ごと、強い小節だけ)
    for bi, (name, c0) in enumerate(zip(pattern_names, bar_starts)):
        if name is None or (bi + 1) % 8 != 0 or (bi < len(intensity) and intensity[bi] < 0.6):
            continue
        for step in (12, 13, 14, 15):
            ci = c0 + step
            if ci < len(cells) - 1:
                hits.append(Hit(float(cells[ci]), "snare", 0.6 + 0.1 * (step - 12)))
    hits.sort(key=lambda h: h.time)
    return hits
