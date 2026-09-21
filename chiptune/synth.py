"""NES (ファミコン) 風の 4 チャンネル・シンセサイザ.

  - pulse1  : メロディ (矩形波, デューティ比可変, ディレイ・ビブラート, レガート・ポルタメント)
  - pulse2  : メロディのエコー (細いデューティ比の矩形波を 8 分音符遅らせて重ねる, ゲーム音楽の定番)
              + コードのアルペジオ (矩形波)
  - triangle: ベース (4bit 階段状の三角波)
  - noise   : ドラム (15bit LFSR ノイズ)
"""
from __future__ import annotations

import numpy as np

from .analysis import Hit, Note, Score

OUT_SR = 44100

# メロディ表現のパラメータ
VIBRATO_ONSET = 0.20      # ビブラートが掛かり始めるまでの秒数
VIBRATO_FADE = 0.10       # 深さが最大になるまでの秒数
VIBRATO_DEPTH_ST = 0.25   # 半音単位の深さ
VIBRATO_RATE = 6.0        # Hz
LEGATO_GAP = 0.020        # これより短い隙間で続く音はレガート扱い (秒)
LEGATO_MAX_INTERVAL = 0   # ピッチスライドは既定で使わない (上ずって聞こえるため)。>0 で有効
SLIDE_TIME = 0.025        # スライドに掛ける秒数
ECHO_GAIN = 0.35
ECHO_DUTY = 0.125

# ミックスの各チャンネルの重み (正規化前)
MIX_MELODY = 0.34
MIX_ARP = 0.12
MIX_BASS = 0.30
MIX_DRUMS = 0.24
PEAK_TARGET = 0.9


# --------------------------------------------------------------------------- 波形生成

def _midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def pulse(freq: np.ndarray, sr: int, duty: float, phase0: float = 0.0) -> np.ndarray:
    phase = (phase0 + np.cumsum(freq / sr)) % 1.0
    return np.where(phase < duty, 1.0, -1.0).astype(np.float32)


def _end_phase(freq: np.ndarray, sr: int, phase0: float = 0.0) -> float:
    """pulse() と同じ位相計算で最後のサンプルの位相を返す (レガート時の位相連続用)."""
    return float((phase0 + np.sum(freq / sr, dtype=np.float64)) % 1.0)


def triangle_nes(freq: np.ndarray, sr: int) -> np.ndarray:
    """NES の三角波チャンネルは 4bit (32 ステップ) の階段状."""
    phase = np.cumsum(freq / sr) % 1.0
    step = np.floor(phase * 32).astype(int) % 32
    v = np.where(step < 16, 15 - step, step - 16).astype(np.float32)
    return (v / 7.5) - 1.0


_LFSR_CACHE: dict[bool, np.ndarray] = {}


def _lfsr_sequence(short_mode: bool) -> np.ndarray:
    """NES ノイズチャンネルの 15bit LFSR を 1 周期分生成 (キャッシュ)."""
    if short_mode in _LFSR_CACHE:
        return _LFSR_CACHE[short_mode]
    reg = 1
    n = 93 if short_mode else 32767
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = 1.0 if (reg & 1) == 0 else -1.0
        tap = (reg >> 6) & 1 if short_mode else (reg >> 1) & 1
        fb = (reg & 1) ^ tap
        reg = (reg >> 1) | (fb << 14)
    _LFSR_CACHE[short_mode] = out
    return out


def noise_nes(n_samples: int, sr: int, rate_hz: float, short_mode: bool = False, seed: int = 0) -> np.ndarray:
    """rate_hz でクロックされる LFSR ノイズ."""
    seq = _lfsr_sequence(short_mode)
    idx = (np.arange(n_samples) * rate_hz / sr + seed).astype(np.int64) % len(seq)
    return seq[idx]


def _quantize_amp(x: np.ndarray, levels: int = 16) -> np.ndarray:
    """振幅を 4bit 相当に量子化して "ピコピコ感" を出す."""
    return np.round(x * (levels - 1)) / (levels - 1)


def _envelope(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    env = np.full(n, sustain, dtype=np.float32)
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    if a > 0:
        env[:a] = np.linspace(0, 1, min(a, n))[: min(a, n)]
    if d > 0 and a < n:
        seg = np.linspace(1, sustain, d)[: max(0, n - a)]
        env[a:a + len(seg)] = seg
    if r > 0 and n > r:
        env[-r:] *= np.linspace(1, 0, r)
    return env


# --------------------------------------------------------------------------- チャンネル描画

def _legato_flags(notes: list[Note]) -> list[bool]:
    """notes[i] が直前の音からレガート (隙間ほぼ無し & 音程差が小さい) で続くかどうか."""
    flags = [False] * len(notes)
    for i in range(1, len(notes)):
        prev, cur = notes[i - 1], notes[i]
        gap = cur.start - prev.end
        interval = abs(cur.midi - prev.midi)
        flags[i] = (-1e-3 <= gap < LEGATO_GAP) and (0 < interval <= LEGATO_MAX_INTERVAL)
    return flags


def render_pulse_channel(
    notes: list[Note], total: int, sr: int, duty: float,
    vibrato: bool = True, decay_to: float = 0.80, gap: float = 0.025, octave_shift: int = 0,
    legato: bool = False,
) -> np.ndarray:
    """メロディ用パルス波.

    - ディレイ・ビブラート: 音の出だしは真っ直ぐ, VIBRATO_ONSET 秒後から掛かる
    - レガート: 隙間なく続く近い音程 (<= LEGATO_MAX_INTERVAL 半音) は前の音から
      SLIDE_TIME 秒でピッチをスライドし, リトリガーの隙間 (gap) と位相の切れ目を作らない
    - octave_shift は半音単位 (12 で 1 オクターブ)
    """
    buf = np.zeros(total, dtype=np.float32)
    notes = sorted(notes, key=lambda x: x.start)
    # 接続の判定: 譜面の legato フラグ (同じフレーズ内) を優先し、無ければ従来の隙間ベース
    auto = _legato_flags(notes) if legato else [False] * len(notes)
    connected = [False] * len(notes)   # notes[i] が直前の音と無音を挟まずに続く
    for i in range(1, len(notes)):
        prev, cur = notes[i - 1], notes[i]
        connected[i] = (getattr(prev, "legato", False) and cur.start - prev.end < 0.03) or auto[i]
    min_len = int(0.01 * sr)
    gap_n = int(gap * sr)
    slide_n = int(SLIDE_TIME * sr)
    dip_n = int(0.025 * sr)
    prev_hz: float | None = None
    prev_phase = 0.0
    prev_end = -1  # 直前に描画した音の終端サンプル
    for i, nt in enumerate(notes):
        s = int(nt.start * sr)
        conn_in = connected[i] and prev_hz is not None and prev_end >= s - int(0.03 * sr)
        conn_out = i + 1 < len(notes) and connected[i + 1]
        # 切れ目 (gap) は、直後に別の音が続くときだけ入れる。休符の前では入れない (フレーズの尻切れ防止)
        needs_gap = (not conn_out) and i + 1 < len(notes) and (notes[i + 1].start - nt.end) < 0.06
        e = int(nt.end * sr) - (gap_n if needs_gap else 0)
        e = min(e, total)
        if e - s < min_len:
            prev_hz = None
            continue
        if conn_in:
            s = prev_end  # 前の音と隙間なく繋ぐ
        n = e - s
        t = np.arange(n, dtype=np.float32) / sr
        hz = _midi_to_hz(nt.midi + octave_shift)
        f = np.full(n, hz, dtype=np.float32)
        interval = abs(nt.midi - notes[i - 1].midi) if i > 0 else 99
        slide_in = conn_in and 0 < interval <= LEGATO_MAX_INTERVAL and slide_n > 0
        if slide_in:
            # しゃくり: 半音単位で直線的に (周波数は指数的に) 前の音程から滑らせる
            k = min(slide_n, n)
            ratio = np.clip(np.arange(k, dtype=np.float32) / slide_n, 0.0, 1.0)
            f[:k] = prev_hz * (hz / prev_hz) ** ratio
        if vibrato and n > int(VIBRATO_ONSET * sr):
            depth = 2 ** (VIBRATO_DEPTH_ST / 12) - 1
            ramp = np.clip((t - VIBRATO_ONSET) / VIBRATO_FADE, 0, 1)
            f = f * (1 + depth * np.sin(2 * np.pi * VIBRATO_RATE * t) * ramp)
        phase0 = prev_phase if slide_in else 0.0
        w = pulse(f, sr, duty, phase0)
        if slide_in:
            # 音量はリトリガーせず, 前の音の持続レベルからそのまま続ける
            env = _envelope(n, sr, 0.0, 0.0, decay_to, 0.0 if conn_out else 0.01)
        elif conn_in:
            # 同じフレーズ内の音の切り替え (同音連打や大きな跳躍): 無音は作らず、音量のくぼみだけで区切る
            env = _envelope(n, sr, 0.0, 0.08, decay_to, 0.0 if conn_out else 0.01)
            k = min(dip_n, n)
            env[:k] *= np.linspace(0.25, 1.0, k, dtype=np.float32)
        else:
            env = _envelope(n, sr, 0.0, 0.08, decay_to, 0.0 if conn_out else 0.01)
        buf[s:e] += _quantize_amp(w * env * nt.velocity)
        prev_hz = float(f[-1])
        prev_phase = _end_phase(f, sr, phase0)
        prev_end = e
    return buf


def render_arpeggio(
    chords: list[tuple[float, float, list[int]]], cells: np.ndarray, total: int, sr: int,
    duty: float = 0.25, base_octave: int = 5, speed: int = 1,
    bar_starts: list[int] | None = None, intensity: list[float] | None = None,
    melody: list[Note] | None = None,
) -> np.ndarray:
    """1 拍ごとのコードを 16 分音符でぐるぐる回すアルペジオ. 小節の強度が低いところは休む/弱くする."""
    buf = np.zeros(total, dtype=np.float32)
    if not chords:
        return buf
    starts = np.array([c[0] for c in chords])
    # セル → 小節強度
    cell_gain = np.ones(len(cells))
    if bar_starts and intensity:
        bs = np.array(bar_starts)
        for i in range(len(cells) - 1):
            bi = int(np.searchsorted(bs, i, side="right") - 1)
            v = float(intensity[bi]) if 0 <= bi < len(intensity) else 1.0
            cell_gain[i] = 0.0 if v < 0.2 else 0.5 + 0.5 * v
    # 各セルで鳴っているメロディのピッチクラス (衝突回避用)
    mel_pc = [None] * (len(cells) - 1)
    if melody:
        for nt in melody:
            i0 = max(0, int(np.searchsorted(cells, nt.start, side="right") - 1)); i1 = min(len(cells) - 2, int(np.searchsorted(cells, nt.end, side="left") - 1))
            for i in range(i0, i1 + 1):
                mel_pc[i] = nt.midi % 12
    counter = 0
    for i in range(len(cells) - 1):
        t0, t1 = cells[i], cells[i + 1]
        k = int(np.searchsorted(starts, t0, side="right") - 1)
        if k < 0 or not chords[k][2] or cell_gain[i] <= 0:
            continue
        pcs = sorted(chords[k][2])
        if mel_pc[i] is not None:
            safe = [pc for pc in pcs if pc == mel_pc[i] or (abs(pc - mel_pc[i]) % 12) not in (1, 11)]
            if safe:
                pcs = safe
        # 1 セルを speed 分割 (speed=2 なら 32 分音符)
        sub = np.linspace(t0, t1, speed + 1)
        for j in range(speed):
            pc = pcs[counter % len(pcs)]
            counter += 1
            midi = 12 * base_octave + pc
            s, e = int(sub[j] * sr), int(sub[j + 1] * sr) - int(0.004 * sr)
            n = e - s
            if n <= 0:
                continue
            f = np.full(n, _midi_to_hz(midi), dtype=np.float32)
            env = _envelope(n, sr, 0.0, 0.05, 0.6, 0.005)
            buf[s:e] += _quantize_amp(pulse(f, sr, duty) * env * 0.8 * cell_gain[i])
    return buf


def render_triangle_bass(notes: list[Note], total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for nt in notes:
        s = int(nt.start * sr)
        e = int(nt.end * sr) - int(0.01 * sr)
        n = e - s
        if n <= 0:
            continue
        f = np.full(n, _midi_to_hz(nt.midi), dtype=np.float32)
        env = _envelope(n, sr, 0.0, 0.0, 1.0, 0.008)
        buf[s:e] += triangle_nes(f, sr) * env
    return buf


def render_drums(hits: list[Hit], total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, h in enumerate(hits):
        s = int(h.time * sr)
        if h.kind == "kick":
            n = int(0.16 * sr)
            t = np.arange(n) / sr
            f = 160 * np.exp(-t * 28) + 42
            body = triangle_nes(f.astype(np.float32), sr) * np.exp(-t * 18)
            click = noise_nes(int(0.015 * sr), sr, 44100 / 4, seed=i * 7) * np.linspace(1, 0, int(0.015 * sr))
            w = body
            w[: len(click)] += click * 0.6
            v = 1.0
        elif h.kind == "snare":
            n = int(0.14 * sr)
            t = np.arange(n) / sr
            nz = noise_nes(n, sr, 44100 / 6, seed=i * 13)
            tone = triangle_nes(np.full(n, 180.0, dtype=np.float32), sr) * np.exp(-t * 60)
            w = nz * np.exp(-t * 22) * 0.9 + tone * 0.5
            v = 0.85
        else:  # hat
            n = int(0.045 * sr)
            t = np.arange(n) / sr
            nz = noise_nes(n, sr, 44100 / 2, short_mode=False, seed=i * 17)
            w = nz * np.exp(-t * 80)
            v = 0.45
        e = min(total, s + len(w))
        if e <= s:
            continue
        buf[s:e] += _quantize_amp(w[: e - s] * v * h.velocity)
    return buf


# --------------------------------------------------------------------------- ミックス

def echo_delay_seconds(tempo: float) -> float:
    """エコーの遅延 = 8 分音符 1 個分 (テンポ不明なら 0.25 秒)."""
    return 60.0 / tempo / 2.0 if tempo > 0 else 0.25


def _delay(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """x を delay_samples だけ遅らせる (長さは変えない)."""
    out = np.zeros_like(x)
    d = max(0, int(delay_samples))
    if d < len(x):
        out[d:] = x[: len(x) - d]
    return out


def _soft_limit(x: np.ndarray, knee: float = 0.75) -> np.ndarray:
    """knee 以上のピークだけを tanh で丸めて |x| <= 1 に収める (波形の連続性を保つ)."""
    a = np.abs(x)
    over = a > knee
    if not np.any(over):
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (1.0 - knee) * np.tanh((a[over] - knee) / (1.0 - knee)))
    return y


def render_melody_with_echo(
    notes: list[Note], total: int, sr: int, tempo: float, *,
    duty: float = 0.5, echo: bool = False, octave_shift: int = 0, gain: float = 1.0,
) -> np.ndarray:
    """メロディ + (8 分音符遅れの細いパルスによる) エコーをまとめて描画する."""
    mel = render_pulse_channel(notes, total, sr, duty=duty, octave_shift=octave_shift)
    if echo:
        eco = render_pulse_channel(notes, total, sr, duty=ECHO_DUTY, octave_shift=octave_shift)
        mel = mel + ECHO_GAIN * _delay(eco, int(round(echo_delay_seconds(tempo) * sr)))
    return mel * gain


def render(score: Score, sr: int = OUT_SR, *, melody_duty: float = 0.5, arpeggio: bool = True,
           drums: bool = True, bass: bool = True, arp_speed: int = 1, crunch: bool = True,
           echo: bool = False, melody_octave: int = 0, melody_gain: float = 1.0) -> np.ndarray:
    total = int(np.ceil(score.duration * sr)) + sr // 10
    arp_oct = getattr(score, "arp_octave", 5)
    bar_starts = getattr(score, "bar_starts", None)
    intensity = getattr(score, "intensity", None)
    mel = render_melody_with_echo(
        score.melody, total, sr, score.tempo,
        duty=melody_duty, echo=echo, octave_shift=12 * int(melody_octave), gain=melody_gain,
    )
    arp = render_arpeggio(score.chords, score.cells, total, sr, base_octave=arp_oct, speed=arp_speed,
                          bar_starts=bar_starts, intensity=intensity, melody=score.melody) if arpeggio else 0.0
    bas = render_triangle_bass(score.bass, total, sr) if bass else 0.0
    drm = render_drums(score.drums, total, sr) if drums else 0.0

    mix = MIX_MELODY * mel + MIX_ARP * arp + MIX_BASS * bas + MIX_DRUMS * drm
    from .master import finalize
    return finalize(mix, sr, crunch=crunch)
