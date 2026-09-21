"""編曲: 小節ごとの強度 (盛り上がり) と、それに応じたレイヤーの出し入れ、ベースラインの生成、アルペジオの音域。"""
from __future__ import annotations

import librosa
import numpy as np

from .analysis import Note


def bar_intensity(y: np.ndarray, voc: np.ndarray, sr: int, cells: np.ndarray, bar_starts: list[int]) -> np.ndarray:
    """小節ごとの強度 0..1: 混合音の音量 (対数) を 10〜95 パーセンタイルで正規化し、歌の有無で少し補正."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    vr = librosa.feature.rms(y=voc, frame_length=2048, hop_length=512)[0]
    fr = np.clip(np.floor(cells * sr / 512).astype(int), 0, len(rms) - 1)
    vals = []
    for c0 in bar_starts:
        a = fr[c0]; b = fr[min(c0 + 16, len(cells) - 1)]
        seg = rms[a:max(b, a + 1)]; vseg = vr[a:max(b, a + 1)]
        vals.append((20 * np.log10(np.mean(seg) + 1e-6), np.mean(vseg)))
    db = np.array([v[0] for v in vals]); vv = np.array([v[1] for v in vals])
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    inten = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    inten = np.clip(inten + 0.1 * (vv > np.percentile(vv, 50)), 0, 1)
    # 前後 1 小節で少し平滑化 (急な出し入れを防ぐ)
    if len(inten) >= 3:
        inten = np.convolve(np.pad(inten, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return inten


def arpeggio_octave(melody: list[Note]) -> int:
    """アルペジオの基準オクターブ: メロディの下 10 パーセンタイルより 1 オクターブ下に置く (音域の衝突回避)."""
    if not melody:
        return 4
    p10 = np.percentile([n.midi for n in melody], 10)
    # 基準オクターブ o の音域は 12*o .. 12*o+11。その上端が p10 - 5 以下になる最大の o
    o = int((p10 - 5 - 11) // 12)
    return int(np.clip(o, 3, 5))


def bass_line(chords_by_beat, beat_times: np.ndarray, cells: np.ndarray, detected_bass: list[int | None],
              intensity_by_beat: np.ndarray, tempo: float) -> list[Note]:
    """コードの根音を土台にしたベース. 8 分で刻み、強い小節はオクターブ上を混ぜる。
    検出したベース音がコードの構成音なら、その音 (実際の動き) を優先する."""
    notes: list[Note] = []
    for i, ch in enumerate(chords_by_beat):
        if ch is None or i + 1 >= len(beat_times):
            continue
        t0, t1 = beat_times[i], beat_times[i + 1]
        root = ch[1]
        base = 36 + root if root <= 7 else 24 + root  # E1〜B2 付近に収める
        # 検出ベースがコード音なら採用 (音域は C2 付近に畳む)
        det = detected_bass[i] if i < len(detected_bass) else None
        if det is not None and det % 12 in ch[3]:
            base = 36 + (det % 12) if det % 12 <= 7 else 24 + (det % 12)
        inten = float(intensity_by_beat[i]) if i < len(intensity_by_beat) else 0.7
        half = (t1 - t0) / 2
        # 8 分音符 2 つ。強い小節は 2 つ目をオクターブ上、弱い小節は根音を伸ばす
        if inten >= 0.45 or tempo >= 150:
            notes.append(Note(t0, t0 + half * 0.95, base, 1.0))
            second = base + 12 if inten >= 0.7 else base
            notes.append(Note(t0 + half, t1 - 0.01, second, 0.9))
        else:
            notes.append(Note(t0, t1 - 0.01, base, 0.9))
    return notes


def layer_plan(intensity: np.ndarray) -> list[dict]:
    """小節ごとのレイヤー: arp (アルペジオ), arp_speed, hats は render 側で参照."""
    plan = []
    for v in intensity:
        plan.append({
            "arp": bool(v >= 0.25),
            "arp_speed": 2 if v >= 0.8 else 1,
            "arp_gain": float(0.6 + 0.4 * v),
            "drums": bool(v >= 0.15),
        })
    return plan
