"""元曲を解析して、チップチューン用の譜面 (Score) を作る.

流れ:
  1. ビート検出 → 16分音符グリッド
  2. HPSS で「音程成分」と「打楽器成分」に分離
  3. 音程成分から メロディ (pyin) / ベース (pyin, 低域) / コード (chroma) を抽出
  4. 打楽器成分から キック / スネア / ハイハット のヒットを抽出
  5. すべてを 16分音符のセルに量子化
"""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

SR = 22050
HOP = 512
DRUM_DB_FLOOR = -70.0  # ドラム検出で無音をこの dB で頭打ちにする (-45 だとハイハットが消えるので緩めに)


@dataclass
class Note:
    start: float  # 秒
    end: float    # 秒
    midi: int
    velocity: float = 1.0
    legato: bool = False  # 次の音と無音を挟まずにつなぐ (同じフレーズ内)


@dataclass
class Hit:
    time: float
    kind: str        # "kick" | "snare" | "hat"
    velocity: float = 1.0


@dataclass
class Score:
    duration: float
    tempo: float
    cells: np.ndarray                     # shape (n_cells+1,) 各16分音符の境界時刻
    melody: list[Note] = field(default_factory=list)
    bass: list[Note] = field(default_factory=list)
    chords: list[tuple[float, float, list[int]]] = field(default_factory=list)  # (start, end, pitch classes)
    drums: list[Hit] = field(default_factory=list)
    bar_starts: list[int] = field(default_factory=list)   # 小節頭のセル番号 (16 分)
    intensity: list[float] = field(default_factory=list)  # 小節ごとの強度 0..1
    arp_octave: int = 5                                   # アルペジオの基準オクターブ
    key: tuple[int, str] | None = None
    verify: dict | None = None                            # 自己検証レポート (analysis_sep)


# --------------------------------------------------------------------------- utils

def _bandpass(y: np.ndarray, sr: int, lo: float | None, hi: float | None, order: int = 4) -> np.ndarray:
    nyq = sr / 2
    if lo and hi:
        sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    elif lo:
        sos = signal.butter(order, lo / nyq, btype="high", output="sos")
    else:
        sos = signal.butter(order, hi / nyq, btype="low", output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def _make_grid(y: np.ndarray, sr: int, duration: float, onset_src: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """ビート検出をして 16 分音符のグリッド (境界時刻の配列) を返す."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="time")
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beats = np.asarray(beats, dtype=float)

    if len(beats) < 4 or tempo <= 0:
        tempo = tempo if tempo > 0 else 120.0
        period = 60.0 / tempo
        beats = np.arange(0.0, duration + period, period)
    else:
        period = float(np.median(np.diff(beats)))
        # 曲頭 / 曲末までビートを外挿
        head = np.arange(beats[0] - period, -period / 2, -period)[::-1]
        tail = np.arange(beats[-1] + period, duration + period, period)
        beats = np.concatenate([head, beats, tail])
        beats = beats[beats >= -1e-6]

    # 各ビートを 4 分割
    cells = []
    for a, b in zip(beats[:-1], beats[1:]):
        cells.extend(np.linspace(a, b, 4, endpoint=False))
    cells.append(beats[-1])
    cells = np.asarray(cells, dtype=float)
    cells = _align_grid(cells, y if onset_src is None else onset_src, sr)
    cells = cells[(cells >= 0) & (cells <= duration + 1e-6)]
    if cells[-1] < duration:
        cells = np.append(cells, duration)
    if cells[0] > 1e-3:
        cells = np.insert(cells, 0, 0.0)
    return tempo, cells


def _align_grid(cells: np.ndarray, y: np.ndarray, sr: int) -> np.ndarray:
    """ビート検出は数十 ms 遅れることが多いので、オンセット (音の立ち上がり) のピーク時刻が
    セル境界に一番よく乗る位置へ格子全体をずらす."""
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    if len(cells) < 3 or onset.max() <= 0:
        return cells
    peaks = librosa.util.peak_pick(onset, pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=0.3, wait=2)
    if len(peaks) < 4:
        return cells
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP)
    weights = onset[peaks]
    cell_len = float(np.median(np.diff(cells)))
    sigma = 0.012  # 12 ms
    best_shift, best_score = 0.0, -1.0
    for shift in np.linspace(-cell_len / 2, cell_len / 2, 65):
        c = cells + shift
        idx = np.clip(np.searchsorted(c, times), 1, len(c) - 1)
        dist = np.minimum(np.abs(times - c[idx - 1]), np.abs(times - c[idx]))
        score = float(np.sum(weights * np.exp(-(dist / sigma) ** 2)))
        if score > best_score:
            best_score, best_shift = score, float(shift)
    return cells + best_shift


def _frame_of(t: np.ndarray | float, sr: int) -> np.ndarray:
    return np.floor(np.asarray(t) * sr / HOP).astype(int)


def _cell_slices(cells: np.ndarray, n_frames: int, sr: int):
    f = _frame_of(cells, sr)
    f = np.clip(f, 0, n_frames)
    for i in range(len(cells) - 1):
        a, b = f[i], max(f[i + 1], f[i] + 1)
        yield i, slice(a, min(b, n_frames))


# --------------------------------------------------------------------------- pitch tracks

def _pitch_track(y: np.ndarray, sr: int, fmin: float, fmax: float, frame_length: int):
    f0, voiced, prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=HOP,
        fill_na=np.nan,
    )
    return f0, voiced, prob


def _quantize_pitch(
    f0: np.ndarray, voiced: np.ndarray, energy: np.ndarray, cells: np.ndarray, sr: int,
    min_voiced_ratio: float, energy_gate: float,
) -> list[int | None]:
    """フレーム単位の f0 を 16分音符セルごとの MIDI ノート (無ければ None) に落とす."""
    n = len(f0)
    out: list[int | None] = []
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), 1.0, f0))
    for i, sl in _cell_slices(cells, n, sr):
        v = voiced[sl]
        e = energy[sl]
        if v.size == 0 or e.size == 0:
            out.append(None)
            continue
        if v.mean() < min_voiced_ratio or e.max() < energy_gate:
            out.append(None)
            continue
        out.append(int(np.round(np.median(midi[sl][v]))))
    return out


def _fix_octave_glitches(seq: list[int | None]) -> list[int | None]:
    """1セルだけ突然オクターブ飛ぶような誤検出を前後に合わせる."""
    s = list(seq)
    for i in range(1, len(s) - 1):
        a, b, c = s[i - 1], s[i], s[i + 1]
        if b is None or a is None or c is None:
            continue
        if abs(a - c) <= 4 and abs(b - a) >= 10:
            # オクターブ違いなら寄せる
            cand = b + 12 * int(np.round((a - b) / 12))
            s[i] = cand if abs(cand - a) <= 6 else a
    return s


def _cells_to_notes(seq: list[int | None], cells: np.ndarray, vel: list[float] | None = None,
                    max_hold_cells: int | None = None) -> list[Note]:
    notes: list[Note] = []
    i = 0
    n = len(seq)
    while i < n:
        m = seq[i]
        if m is None:
            i += 1
            continue
        j = i + 1
        while j < n and seq[j] == m and (max_hold_cells is None or j - i < max_hold_cells):
            j += 1
        v = float(np.mean(vel[i:j])) if vel is not None else 1.0
        notes.append(Note(cells[i], cells[j], m, v))
        i = j
    return notes


# --------------------------------------------------------------------------- drums

def _drum_hits(y_perc: np.ndarray, sr: int, cells: np.ndarray) -> list[Hit]:
    S = librosa.feature.melspectrogram(y=y_perc, sr=sr, hop_length=HOP, n_mels=64, fmax=sr / 2)
    # 無音からの立ち上がりが極端に大きく評価されると、音の密な区間 (サビ) のヒットが相対的に消えるので床を設ける
    S_db = librosa.power_to_db(S, ref=np.max)
    if DRUM_DB_FLOOR is not None:
        S_db = np.maximum(S_db, DRUM_DB_FLOOR)
    mel_f = librosa.mel_frequencies(n_mels=64, fmax=sr / 2)
    bands = {
        "kick": (mel_f < 160),
        "snare": (mel_f >= 160) & (mel_f < 3000),
        "hat": (mel_f >= 5000),
    }
    thresholds = {"kick": 0.40, "snare": 0.50, "hat": 0.40}
    hits: list[Hit] = []
    for kind, mask in bands.items():
        band = S_db[mask].mean(axis=0)
        # 立ち上がりだけを取る (スペクトラルフラックス)
        flux = np.maximum(0, np.diff(band, prepend=band[0]))
        if flux.max() <= 0:
            continue
        flux = flux / (np.percentile(flux, 99) + 1e-9)
        peaks = librosa.util.peak_pick(
            flux, pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=0.3, wait=2,
        )
        peaks = [p for p in peaks if flux[p] >= thresholds[kind]]
        # 最寄りのセル境界へスナップし、同じセルでは最強のピークだけ採用
        times = librosa.frames_to_time(np.asarray(peaks, dtype=int), sr=sr, hop_length=HOP)
        best: dict[int, float] = {}
        for p, t in zip(peaks, times):
            ci = int(np.argmin(np.abs(cells - t)))
            if ci >= len(cells) - 1:
                continue
            best[ci] = max(best.get(ci, 0.0), float(flux[p]))
        for ci, v in best.items():
            hits.append(Hit(float(cells[ci]), kind, min(1.0, 0.4 + 0.6 * v)))
    hits.sort(key=lambda h: h.time)
    return hits


# --------------------------------------------------------------------------- main

def analyze(y: np.ndarray, sr: int = SR) -> Score:
    duration = len(y) / sr
    y = y / (np.abs(y).max() + 1e-9)

    harm, perc = librosa.effects.hpss(y)
    # 格子の位置合わせは打楽器成分の方が正確
    tempo, cells = _make_grid(y, sr, duration, onset_src=perc)

    # ---- メロディ: ボーカル/主旋律の帯域を強調して pyin
    mel_src = _bandpass(harm, sr, 180, 2500)
    f0_m, v_m, _ = _pitch_track(mel_src, sr, fmin=110.0, fmax=1100.0, frame_length=2048)
    rms_m = librosa.feature.rms(y=mel_src, frame_length=2048, hop_length=HOP)[0]
    n = min(len(f0_m), len(rms_m))
    f0_m, v_m, rms_m = f0_m[:n], v_m[:n], rms_m[:n]
    gate_m = np.percentile(rms_m, 90) * 0.15
    mel_seq = _quantize_pitch(f0_m, v_m, rms_m, cells, sr, min_voiced_ratio=0.4, energy_gate=gate_m)
    mel_seq = _fix_octave_glitches(mel_seq)
    mel_vel = []
    for i, sl in _cell_slices(cells, n, sr):
        e = rms_m[sl].max() if rms_m[sl].size else 0.0
        mel_vel.append(float(np.clip(0.5 + 0.5 * e / (np.percentile(rms_m, 95) + 1e-9), 0.4, 1.0)))
    melody = _cells_to_notes(mel_seq, cells, mel_vel)

    # ---- ベース: 低域だけを pyin
    bass_src = _bandpass(harm, sr, None, 260)
    f0_b, v_b, _ = _pitch_track(bass_src, sr, fmin=36.0, fmax=250.0, frame_length=4096)
    rms_b = librosa.feature.rms(y=bass_src, frame_length=4096, hop_length=HOP)[0]
    nb = min(len(f0_b), len(rms_b))
    f0_b, v_b, rms_b = f0_b[:nb], v_b[:nb], rms_b[:nb]
    gate_b = np.percentile(rms_b, 90) * 0.2
    bass_seq = _quantize_pitch(f0_b, v_b, rms_b, cells, sr, min_voiced_ratio=0.3, energy_gate=gate_b)
    bass_seq = _fix_octave_glitches(bass_seq)
    # ベースは E1〜E3 (40〜52 くらい) に収める
    bass_seq = [None if m is None else (m - 12 if m > 52 else (m + 12 if m < 33 else m)) for m in bass_seq]

    # ---- コード: chroma を 1 拍ごとに集計して上位 3 音
    chroma = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=HOP, n_chroma=12)
    n_c = chroma.shape[1]
    chords: list[tuple[float, float, list[int]]] = []
    beat_bounds = cells[::4]
    if beat_bounds[-1] < cells[-1]:
        beat_bounds = np.append(beat_bounds, cells[-1])
    fb = np.clip(_frame_of(beat_bounds, sr), 0, n_c)
    for i in range(len(beat_bounds) - 1):
        a, b = fb[i], max(fb[i + 1], fb[i] + 1)
        c = np.median(chroma[:, a:b], axis=1) if b > a else np.zeros(12)
        if c.max() <= 0.2:
            chords.append((beat_bounds[i], beat_bounds[i + 1], []))
            continue
        order = np.argsort(c)[::-1]
        pcs = [int(p) for p in order[:3] if c[p] >= 0.5 * c[order[0]]]
        chords.append((beat_bounds[i], beat_bounds[i + 1], pcs))

    # ベースが取れないところはコードの最強音で補う
    for i in range(len(bass_seq)):
        if bass_seq[i] is None:
            t = cells[i]
            k = int(np.searchsorted(beat_bounds, t, side="right") - 1)
            if 0 <= k < len(chords) and chords[k][2] and rms_b[min(_frame_of(t, sr), nb - 1)] > gate_b:
                pc = chords[k][2][0]
                bass_seq[i] = 36 + pc  # C2 基準
    bass = _cells_to_notes(bass_seq, cells, max_hold_cells=2)  # 8分で刻む

    # ---- ドラム
    drums = _drum_hits(perc, sr, cells)

    return Score(duration=duration, tempo=tempo, cells=cells,
                 melody=melody, bass=bass, chords=chords, drums=drums)
