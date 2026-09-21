"""basic-pitch (Spotify の採譜モデル) を使った高精度モード.

多声のノートイベントを取り出し、
  - 各16分音符セルで一番高い音 → メロディ (スカイライン法)
  - 各セルで一番低い音 (低域) → ベース
  - 各拍でアクティブなノートのピッチクラス → コード
に振り分ける。ドラムは DSP 版と同じ処理を使う。
"""
from __future__ import annotations

import logging
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from . import analysis as A
from .analysis import Note, Score
from .audio_io import write_wav

log = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import basic_pitch  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _predict_notes(y: np.ndarray, sr: int) -> list[tuple[float, float, int, float]]:
    import scipy.signal
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    # basic-pitch は scipy 1.13 で削除された scipy.signal.gaussian を使うので補う
    if not hasattr(scipy.signal, "gaussian"):
        scipy.signal.gaussian = scipy.signal.windows.gaussian

    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "src.wav"
        write_wav(p, y, sr)
        _, _, events = predict(
            str(p), model_path,
            onset_threshold=0.5, frame_threshold=0.3,
            minimum_note_length=60, minimum_frequency=35.0, maximum_frequency=2100.0,
            melodia_trick=True,
        )
    # events: (start_s, end_s, midi, amplitude, pitch_bends)
    return [(float(s), float(e), int(m), float(a)) for s, e, m, a, *_ in events]


def active_notes_per_cell(y: np.ndarray, sr: int, cells: np.ndarray):
    """basic-pitch で採譜し、各セルでアクティブな (midi, amplitude) のリストと振幅の基準値を返す."""
    events = _predict_notes(y, sr)
    log.info("basic-pitch: %d note events", len(events))
    n_cells = len(cells) - 1
    active: list[list[tuple[int, float]]] = [[] for _ in range(n_cells)]
    starts = cells[:-1]
    ends = cells[1:]
    for s, e, m, a in events:
        i0 = max(0, int(np.searchsorted(cells, s, side="right") - 1))
        i1 = min(n_cells - 1, int(np.searchsorted(cells, e, side="left") - 1))
        for i in range(i0, i1 + 1):
            ov = min(e, ends[i]) - max(s, starts[i])
            if ov >= 0.4 * (ends[i] - starts[i]):
                active[i].append((m, a))
    amps = np.array([a for *_, a in events]) if events else np.array([1.0])
    amp_ref = float(np.percentile(amps, 90) + 1e-9)
    return active, amp_ref


def skyline_melody(active, amp_ref: float, lo: int = 52, hi: int = 96):
    """各セルで一番高い (かつ十分強い) 音を主旋律として選ぶ. 戻り値 (midi 列, ベロシティ列)."""
    mel_seq: list[int | None] = []
    mel_vel: list[float] = []
    prev: int | None = None
    for cell in active:
        cand = [(m, a) for m, a in cell if lo <= m <= hi and a >= 0.25 * amp_ref]
        if not cand:
            mel_seq.append(None)
            mel_vel.append(0.0)
            prev = None
            continue
        # 音量で重み付けした上で高い音を優先
        cand.sort(key=lambda x: (x[0] + 6 * min(1.0, x[1] / amp_ref)), reverse=True)
        m, a = cand[0]
        # 直前の音がまだ鳴っていて、選ばれた音がそれより少し高いだけなら直前の音を続ける (レガート)
        if prev is not None and prev != m and any(pm == prev for pm, _ in cand) and m - prev <= 3:
            m = prev
            a = next(pa for pm, pa in cand if pm == prev)
        mel_seq.append(m)
        mel_vel.append(float(np.clip(0.55 + 0.45 * a / amp_ref, 0.5, 1.0)))
        prev = m
    return A._fix_octave_glitches(mel_seq), mel_vel


def analyze_ml(y: np.ndarray, sr: int = A.SR) -> Score:
    duration = len(y) / sr
    y = y / (np.abs(y).max() + 1e-9)
    _, perc = A.librosa.effects.hpss(y)
    tempo, cells = A._make_grid(y, sr, duration, onset_src=perc)
    drums = A._drum_hits(perc, sr, cells)

    n_cells = len(cells) - 1
    active, amp_ref = active_notes_per_cell(y, sr, cells)

    # ---- メロディ: スカイライン (中高域で一番高い音)
    mel_seq, mel_vel = skyline_melody(active, amp_ref)
    melody = A._cells_to_notes(mel_seq, cells, mel_vel)

    # ---- ベース: 低域で一番低い音
    bass_seq: list[int | None] = []
    for i in range(n_cells):
        cand = [m for m, a in active[i] if m <= 55 and a >= 0.2 * amp_ref]
        if not cand:
            bass_seq.append(None)
            continue
        # ベースは C2〜B2 の 1 オクターブに畳み込んで安定させる
        m = 36 + (min(cand) % 12)
        bass_seq.append(m)

    # ---- コード: 1 拍ごとに出現ピッチクラス上位 3 つ
    chords: list[tuple[float, float, list[int]]] = []
    beat_bounds = cells[::4]
    if beat_bounds[-1] < cells[-1]:
        beat_bounds = np.append(beat_bounds, cells[-1])
    for k in range(len(beat_bounds) - 1):
        i0 = int(np.searchsorted(cells, beat_bounds[k], side="left"))
        i1 = int(np.searchsorted(cells, beat_bounds[k + 1], side="left"))
        cnt: Counter[int] = Counter()
        for i in range(i0, min(i1, n_cells)):
            for m, a in active[i]:
                cnt[m % 12] += a
        if not cnt:
            chords.append((beat_bounds[k], beat_bounds[k + 1], []))
            continue
        top = cnt.most_common(3)
        pcs = [pc for pc, w in top if w >= 0.35 * top[0][1]]
        chords.append((beat_bounds[k], beat_bounds[k + 1], pcs))

    # ベースが無いセルはコードの最強音で補う (音があるところだけ)
    for i in range(n_cells):
        if bass_seq[i] is None and active[i]:
            k = int(np.searchsorted(beat_bounds, cells[i], side="right") - 1)
            if 0 <= k < len(chords) and chords[k][2]:
                bass_seq[i] = 36 + chords[k][2][0]
    bass = A._cells_to_notes(bass_seq, cells, max_hold_cells=2)

    return Score(duration=duration, tempo=tempo, cells=cells,
                 melody=melody, bass=bass, chords=chords, drums=drums)
