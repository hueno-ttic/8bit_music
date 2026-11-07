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
# メロディを主役に: 伴奏 (特に低い三角波ベース) がメロディと同じくらい鳴っていて、聴くとメロディより低音が耳に付いた
MIX_MELODY = 0.42
MIX_ARP = 0.10
MIX_BASS = 0.22
MIX_DRUMS = 0.22
PEAK_TARGET = 0.9


# --------------------------------------------------------------------------- 波形生成

def _midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """PolyBLEP: 位相 t (0..1) の不連続点の前後 1 サンプルに掛ける補正 (帯域制限ステップの近似)."""
    out = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a]
    out[a] = x + x - x * x - 1.0
    b = t > 1.0 - dt
    x = (t[b] - 1.0) / dt[b]
    out[b] = x * x + x + x + 1.0
    return out


def pulse(freq: np.ndarray, sr: int, duty: float, phase0: float = 0.0) -> np.ndarray:
    """矩形波 (パルス波). 素朴に ±1 を切り替えるとナイキストを超える倍音が折り返し、元の音と無関係な高さの成分
    (倍音比 -20 dB 程度) が混ざって濁った不協和音に聞こえたので、PolyBLEP で不連続点を帯域制限する."""
    freq = np.asarray(freq, dtype=np.float64)
    dt = np.clip(freq / sr, 1e-9, 0.5)
    phase = (phase0 + np.cumsum(freq / sr)) % 1.0
    y = np.where(phase < duty, 1.0, -1.0)
    y += _polyblep(phase, dt)                         # 立ち上がり (位相 0)
    y -= _polyblep((phase - duty) % 1.0, dt)          # 立ち下がり (位相 duty)
    y[freq <= 0] = 0.0
    return y.astype(np.float32)


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


NES_FRAME = 0.01           # 歌に沿わせるときの音高・音量の更新間隔 (実機の 1/60 秒だと音の切り替えが最大 17 ms 遅れ、楽譜との一致が下がった)
FOLLOW_SNAP_HYST = 0.5     # 半音に揃えるときのヒステリシス (半音): これ以上離れたら次の半音へ
FOLLOW_MAX_DEV = 2.5       # 歌の音高が音符からこれ以上離れていたら音符の高さを使う (分離の残渣・ハモリ)
FOLLOW_VOL_FLOOR = 9       # 音符の中で歌の音量がくぼんでも残す最小の音量 (0..15)。低すぎると小さい音節が聞こえなくなった
FOLLOW_VOL_DIV = 3.0       # 歌の音量 (dB) を何 dB ごとに 1 段階下げるか
FOLLOW_ARTIC_GAP = 0.02    # 音節の頭で言い直すときの切れ目 (秒)
FOLLOW_ARTIC_MIN = 0.06    # 前の音がこれより短いときは切れ目を入れない
FOLLOW_ARTIC_TOL = 0.04    # 音の頭と音節の頭のずれの許容 (秒)
FOLLOW_GAP_FILL = 0.15     # この秒数以内の音符のすき間は、歌が続いていれば (FOLLOW_GAP_DB 以上) 埋める
FOLLOW_GAP_DB = -25.0
FOLLOW_PITCH = "notes"     # 高さの決め方: "notes" = 採譜した音符 (+ 歌が留まった別の半音), "voice" = 歌の f0 をなぞる
FOLLOW_STAY = 0.1          # "notes" で歌に従うのは、別の半音に留まる時間がこれ以上のとき
FOLLOW_DWELL = 5           # 音節の途中で別の半音へ移るのは、それが 5 フレーム (50 ms) 以上続くときだけ


def follow_frames(notes: list[Note], contour: dict, duration: float, *, snap: bool = True):
    """歌に沿わせたメロディの 1/60 秒ごとの (MIDI 音高 [無音=nan], 音量 0..15).

    鳴らす区間は採譜した音符 (メロディと判定した所) に限り、高さは歌の f0 (外れていれば音符の高さ),
    音量は歌の音量の動き (子音のくぼみ・抑揚) を 16 段階にしたもの。音符の区切りではなく歌そのものの動きで鳴るので、
    歌と同じようにつながり、同じ所で区切れる。snap=True なら半音に揃える (ヒステリシス付きでビブラートでは揺れない)。"""
    from scipy.ndimage import median_filter
    hop = float(contour["hop"])
    cm = np.asarray(contour["midi"], dtype=np.float64)
    lv = np.asarray(contour["level_db"], dtype=np.float64)
    ok = np.isfinite(cm)
    sm = cm.copy()
    if ok.any():
        # 約 70 ms の中央値でならす (無声フレームは近くの有声値で埋めてから)
        idx = np.where(ok, np.arange(len(cm)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = np.where(ok, cm, cm[idx])
        sm = np.where(ok, median_filter(filled, size=7, mode="nearest"), np.nan)
    nf = int(np.ceil(duration / NES_FRAME)) + 1
    tf = np.arange(nf) * NES_FRAME
    npitch = np.full(nf, np.nan)
    nstart = np.zeros(nf, bool)
    for nt in notes:
        a, b = int(np.round(nt.start / NES_FRAME)), int(np.round(nt.end / NES_FRAME))
        if b > a:
            npitch[a:b] = nt.midi
            nstart[a] = True
    k = np.clip(np.round(tf / hop).astype(int), 0, len(sm) - 1)
    c, l = sm[k], lv[k]
    # 音符と音符の短いすき間 (歌は続いている) は埋めて、無音で途切れないようにする
    gap_max = int(round(FOLLOW_GAP_FILL / NES_FRAME))
    i = 0
    while i < nf:
        if np.isfinite(npitch[i]):
            i += 1
            continue
        j = i
        while j < nf and not np.isfinite(npitch[j]):
            j += 1
        if 0 < i and j < nf and j - i <= gap_max and np.min(l[i:j]) > FOLLOW_GAP_DB:
            npitch[i:j] = npitch[i - 1]
        i = j
    # オクターブ違いに飛んだ f0 は音符の高さの側へ戻す
    d = c - npitch
    c = np.where(np.isfinite(d) & (np.abs(np.abs(d) - 12) < 1.0), c - 12 * np.sign(d), c)
    use_c = np.isfinite(c) & (np.abs(c - npitch) <= FOLLOW_MAX_DEV)
    if FOLLOW_PITCH == "notes":
        # 高さは採譜した音符 (意図された高さ)。歌がはっきり別の半音に FOLLOW_STAY 秒以上留まった所だけ歌に従う
        # (楽譜の分かっている評価曲で、歌の高さをそのままなぞるとしゃくり・ビブラートの谷まで鳴らし、楽譜から離れた)
        stay = int(round(FOLLOW_STAY / NES_FRAME))
        cr = np.where(use_c, np.round(c), np.nan)
        alt = np.full(nf, np.nan)
        i = 0
        while i < nf:
            if not (np.isfinite(cr[i]) and cr[i] != npitch[i] and abs(c[i] - cr[i]) <= 0.35):
                i += 1
                continue
            j = i
            while j < nf and cr[j] == cr[i] and np.isfinite(npitch[j]) and abs(c[j] - cr[i]) <= 0.45:
                j += 1
            if j - i >= stay:
                alt[i:j] = cr[i]
            i = max(j, i + 1)
        target = np.where(np.isfinite(alt), alt, npitch)
    else:
        target = np.where(use_c, c, npitch)
    pitch = np.full(nf, np.nan)
    cur = None
    dwell = FOLLOW_DWELL
    for i in range(nf):
        if not np.isfinite(npitch[i]):
            cur = None
            continue
        x = target[i]
        if not snap:
            pitch[i] = x
            continue
        if cur is None or (nstart[i] and np.round(x) != cur):
            cur = float(np.round(x))          # 新しい音節の頭はすぐに移る
        elif abs(x - cur) > FOLLOW_SNAP_HYST:
            # 音節の途中では、新しい半音が dwell フレーム以上続くときだけ移る (ビブラート・装飾で細かく往復しない)
            new = float(np.round(x))
            seg = target[i:i + dwell]
            if len(seg) == dwell and np.all(np.isfinite(seg)) and np.all(np.abs(seg - new) <= 0.5 + 1e-6):
                cur = new
        pitch[i] = cur
    # 音量: 歌の抑揚を残しつつ幅を狭める (小さく歌った音節が聞こえなくならないように 9..15)
    vol = np.clip(np.round(15 + l / FOLLOW_VOL_DIV), FOLLOW_VOL_FLOOR, 15)
    vol = np.where(np.isfinite(pitch), vol, 0)
    # 言い直し: 音節の頭に当たる音の頭では、直前を少しだけ切って区切る (同じ音節の中の音高の変化はつなげたまま)。
    # 前の音が短い (FOLLOW_ARTIC_MIN 未満) ときは削らない (音が飛ばないように)
    syl = np.asarray(contour.get("syllables", []), dtype=np.float64)
    gap = max(1, int(round(FOLLOW_ARTIC_GAP / NES_FRAME)))
    keep = int(round(FOLLOW_ARTIC_MIN / NES_FRAME))
    if len(syl):
        starts = [nt.start for nt in notes]
        for st in starts:
            if np.min(np.abs(syl - st)) > FOLLOW_ARTIC_TOL:
                continue
            a = int(np.round(st / NES_FRAME))
            if a - gap - keep < 0 or not np.all(np.isfinite(pitch[a - gap - keep:a])):
                continue                     # 休符の後や前の音が短いときは言い直しの切れ目は不要/入れない
            vol[a - gap:a] = 0
    return pitch, vol


def render_follow_melody(notes: list[Note], contour: dict, total: int, sr: int, *, duty: float = 0.5,
                         snap: bool = True, octave_shift: int = 0, vibrato: bool = True) -> np.ndarray:
    """歌に沿わせたメロディを矩形波で描く (音高・音量は 1/60 秒ごとに段階的に変わる)."""
    pitch, vol = follow_frames(notes, contour, total / sr, snap=snap)
    spf = sr * NES_FRAME
    idx = np.minimum((np.arange(total) / spf).astype(int), len(pitch) - 1)
    p = pitch[idx] + octave_shift
    on = np.isfinite(p)
    if vibrato and snap:
        # 同じ半音を長く保つ所にだけ控えめなビブラート (半音に揃えると歌のビブラートが消えるため)
        held = np.zeros(len(pitch))
        for i in range(1, len(pitch)):
            held[i] = held[i - 1] + NES_FRAME if (np.isfinite(pitch[i]) and pitch[i] == pitch[i - 1]) else 0.0
        h = held[idx]
        t = np.arange(total) / sr
        ramp = np.clip((h - VIBRATO_ONSET) / VIBRATO_FADE, 0, 1)
        p = p + VIBRATO_DEPTH_ST * ramp * np.sin(2 * np.pi * VIBRATO_RATE * t)
    hz = np.where(on, 440.0 * 2 ** ((np.nan_to_num(p, nan=69.0) - 69) / 12), 0.0)
    w = pulse(hz, sr, duty)
    amp = (vol[idx] / 15.0).astype(np.float32)
    k = max(1, int(0.002 * sr))                                   # 2 ms でならしてクリックを抑える
    amp = np.convolve(amp, np.ones(k, dtype=np.float32) / k, mode="same")
    return w * amp * on


VOICE_GAP_FILL = 0.03     # これより短い無声 (歌の f0 の取りこぼし) は前後の高さでつなぐ。子音の途切れ (それ以上) は区切りとして残す
VOICE_DB_FLOOR = -18.0    # 音量の下限 (dB): 小さく歌った音節も聞こえるように
VOICE_DB_COMP = 0.5       # 歌の音量の動き (dB) をこの割合に圧縮する


def voice_frames(contour: dict, duration: float, *, snap: bool = True, hop: float = 0.01):
    """歌の声から直接作るメロディ: 10 ms ごとの (MIDI 音高 [無声=nan], 振幅 0..1).

    採譜した音符を経由しない。耳で確かめると、歌の音高をそのまま単純な音で鳴らしたもの (段階 4) は元のメロディに聞こえ、
    採譜した音符の高さ・区切りで鳴らすと「別の曲」に聞こえた (音符化で高さ・区切り・余計な音が入る)。
    鳴らすのは歌が声を出している所だけで、子音などの途切れはそのまま区切りになる。snap=True なら半音に揃える
    (約 70 ms の中央値でならし、ヒステリシスと最短持続でビブラート・しゃくりでは揺れないようにする)。"""
    from scipy.ndimage import median_filter
    cm = np.asarray(contour["midi"], dtype=np.float64)
    lv = np.asarray(contour["level_db"], dtype=np.float64)
    n = int(np.ceil(duration / hop)) + 1
    m = np.full(n, np.nan); m[:min(n, len(cm))] = cm[:n]
    l = np.full(n, -60.0); l[:min(n, len(lv))] = lv[:n]
    ok = np.isfinite(m)
    # ごく短い無声 (f0 の取りこぼし) をつなぐ
    gmax = int(round(VOICE_GAP_FILL / hop))
    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        if 0 < i and j < n and j - i <= gmax:
            m[i:j] = np.linspace(m[i - 1], m[j], j - i + 2)[1:-1]
            ok[i:j] = True
        i = j
    # オクターブ飛び (短い区間だけ前後から 1 オクターブずれる) を戻す
    from .analysis_sep import fix_octave_jumps
    if ok.any():
        m = np.where(ok, fix_octave_jumps(np.where(ok, m, 0.0), ok, max_len=0.3, ctx=1.0), np.nan)
    if snap and ok.any():
        idx = np.where(ok, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        sm = np.where(ok, median_filter(np.where(ok, m, m[idx]), size=7, mode="nearest"), np.nan)
        out = np.full(n, np.nan)
        cur = None
        for k in range(n):
            if not ok[k]:
                cur = None
                continue
            x = sm[k]
            if cur is None:
                cur = float(np.round(x))
            elif abs(x - cur) > FOLLOW_SNAP_HYST:
                new = float(np.round(x))
                seg = sm[k:k + FOLLOW_DWELL]
                if len(seg) == FOLLOW_DWELL and np.all(np.isfinite(seg)) and np.all(np.abs(seg - new) <= 0.5 + 1e-6):
                    cur = new
            out[k] = cur
        m = out
    db = np.maximum(l * VOICE_DB_COMP, VOICE_DB_FLOOR)
    amp = np.where(np.isfinite(m), 10 ** (db / 20.0), 0.0)
    return m, amp


def render_voice_melody(contour: dict, total: int, sr: int, *, duty: float = 0.5, snap: bool = True,
                        octave_shift: int = 0) -> np.ndarray:
    """歌の声から直接作るメロディを、折り返しを抑えた矩形波で描く."""
    hop = 0.01
    m, amp = voice_frames(contour, total / sr, snap=snap, hop=hop)
    t = np.arange(total) / sr
    fi = t / hop
    k0 = np.clip(fi.astype(int), 0, len(m) - 1)
    if snap:
        p = m[k0]                                        # 半音は段階的に切り替える
    else:
        k1 = np.clip(k0 + 1, 0, len(m) - 1)
        w = fi - k0
        p = np.where(np.isfinite(m[k0]) & np.isfinite(m[k1]), m[k0] * (1 - w) + m[k1] * w, m[k0])
    on = np.isfinite(p)
    hz = np.where(on, 440.0 * 2 ** ((np.nan_to_num(p, nan=69.0) + octave_shift - 69) / 12), 0.0)
    a = np.interp(fi, np.arange(len(amp)), amp)
    k = max(1, int(0.003 * sr))
    a = np.convolve(a * on, np.ones(k) / k, mode="same").astype(np.float32)   # 3 ms でならしてクリックを抑える
    return pulse(hz, sr, duty) * a


CRISP_GAP = 0.018        # 次の音の直前に入れる切れ目 (秒): 1 音ずつはっきり区切る
CRISP_DECAY_TO = 0.72    # 立ち上がり後に落ち着く音量 (ファミコンの音量エンベロープ風)
CRISP_DECAY = 0.10


def quantize_notes(notes: list[Note], cells: np.ndarray, div: int = 2, syllables=None) -> list[Note]:
    """音符の頭と終わりを拍の格子 (16 分を div 分割) に揃える. 歌の音節の頭は拍の前後に ±20 ms ほどばらつき、
    そのまま鳴らすと「震えながら弾いている」ように聞こえた. 同じ格子点にぶつかった音は次の点へ送り、入らなければ短い方を落とす."""
    if not notes or cells is None or len(cells) < 2:
        return list(notes)
    c = np.asarray(cells, dtype=float)
    g = np.concatenate([np.linspace(a, b, div, endpoint=False) for a, b in zip(c[:-1], c[1:])] + [c[-1:]])
    ns = sorted(notes, key=lambda n: n.start)
    syl = np.asarray(syllables if syllables is not None else [], dtype=float)
    if len(syl):
        # 音節の頭の音だけ格子に揃え、音節の中の音 (こぶし・速い経過音) は歌の位置のまま
        out = []
        for n in ns:
            if np.min(np.abs(syl - n.start)) <= 0.04:
                k = int(np.argmin(np.abs(g - n.start)))
                st = float(g[k]) if abs(g[k] - n.start) <= 0.035 else n.start
            else:
                st = n.start
            out.append(Note(st, n.end, n.midi, n.velocity, n.legato))
        for i in range(len(out) - 1):                       # ずらしで重なった所は前の音を縮める
            if out[i].end > out[i + 1].start:
                out[i] = Note(out[i].start, out[i + 1].start, out[i].midi, out[i].velocity, out[i].legato)
            if i > 0 and out[i].start < out[i - 1].end:
                out[i] = Note(out[i - 1].end, out[i].end, out[i].midi, out[i].velocity, out[i].legato)
        return [n for n in out if n.end - n.start > 0.02]
    idx = [int(np.argmin(np.abs(g - n.start))) for n in ns]
    out: list[list] = []
    for i, n in enumerate(ns):
        k = idx[i]
        if out and k <= out[-1][0]:
            k = out[-1][0] + 1
        end_k = int(np.argmin(np.abs(g - n.end)))
        nxt = idx[i + 1] if i + 1 < len(ns) else len(g) - 1
        end_k = max(k + 1, min(end_k, max(nxt, k + 1)))
        if k >= len(g) - 1 or (i + 1 < len(ns) and k >= nxt and n.end - n.start < ns[i + 1].end - ns[i + 1].start):
            continue
        if out and out[-1][1] > k:
            out[-1][1] = k
        out.append([k, min(end_k, len(g) - 1), n])
    return [Note(float(g[a]), float(g[b]), n.midi, n.velocity, n.legato) for a, b, n in out if b > a]


def render_crisp_melody(notes: list[Note], total: int, sr: int, *, duty: float = 0.25, octave_shift: int = 0,
                        syllables=None) -> np.ndarray:
    """1 音ずつはっきり鳴らすメロディ: 音節の頭の音は立ち上げ直し、その直前で短く切る。
    同じ音節の中で高さだけ変わる音 (こぶし・経過音) は切らずにつなぐ (短い音が切れ目で消えないように)。
    揺れ (ビブラート・音量の揺らぎ) は入れない."""
    buf = np.zeros(total, dtype=np.float32)
    ns = sorted(notes, key=lambda n: n.start)
    syl = np.asarray(syllables if syllables is not None else [], dtype=float)

    def is_onset(i):
        if not len(syl):
            return True
        return bool(np.min(np.abs(syl - ns[i].start)) <= 0.04) or ns[i].midi == ns[i - 1].midi

    phase = 0.0
    for i, nt in enumerate(ns):
        s = int(nt.start * sr)
        e = int(nt.end * sr)
        nxt_onset = i + 1 < len(ns) and is_onset(i + 1)
        joined_in = i > 0 and not is_onset(i) and nt.start - ns[i - 1].end < 0.01
        if i + 1 < len(ns) and ns[i + 1].start - nt.end < CRISP_GAP and nxt_onset:
            e = min(e, int(ns[i + 1].start * sr) - int(CRISP_GAP * sr))
        elif i + 1 < len(ns) and not nxt_onset and ns[i + 1].start - nt.end < 0.01:
            e = int(ns[i + 1].start * sr)
        e = min(e, total)
        n = e - s
        if n < int(0.03 * sr):
            n = min(int(0.03 * sr), total - s)          # 短い音も最低 30 ms は鳴らす (音が飛ばないように)
            e = s + n
        if n <= 0:
            continue
        hz = _midi_to_hz(nt.midi + octave_shift)
        if joined_in:
            # 同じ音節の中の高さの変化: 立ち上げ直さず、位相も音量も続ける
            w = pulse(np.full(n, hz), sr, duty, phase)
            env = np.full(n, CRISP_DECAY_TO, dtype=np.float32)
        else:
            w = pulse(np.full(n, hz), sr, duty)
            env = _envelope(n, sr, 0.002, CRISP_DECAY, CRISP_DECAY_TO, 0.0)
        nxt_join = i + 1 < len(ns) and not nxt_onset and ns[i + 1].start - nt.end < 0.01
        if not nxt_join:
            r = min(n, int(0.008 * sr))
            env[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
        phase = (phase if joined_in else 0.0) + float(np.sum(np.full(n, hz) / sr)) % 1.0
        buf[s:e] += w * env * (0.85 + 0.15 * float(np.clip(nt.velocity, 0, 1)))
    return buf


def render(score: Score, sr: int = OUT_SR, *, melody_duty: float = 0.5, arpeggio: bool = True,
           drums: bool = True, bass: bool = True, arp_speed: int = 1, crunch: bool = True,
           echo: bool = False, melody_octave: int = 0, melody_gain: float = 1.0,
           melody_style: str = "follow") -> np.ndarray:
    """melody_style: "follow" = 歌の音高・音量の動きに沿わせる (歌の情報があるとき), "notes" = 採譜した音符を鳴らす."""
    total = int(np.ceil(score.duration * sr)) + sr // 10
    arp_oct = getattr(score, "arp_octave", 5)
    bar_starts = getattr(score, "bar_starts", None)
    intensity = getattr(score, "intensity", None)
    contour = getattr(score, "contour", None)
    if melody_style == "crisp":
        syl = (contour or {}).get("syllables")
        qn = quantize_notes(score.melody, score.cells, syllables=syl)
        mel = melody_gain * render_crisp_melody(qn, total, sr, duty=melody_duty, octave_shift=12 * int(melody_octave),
                                                syllables=syl)
        if echo:
            eco = render_crisp_melody(qn, total, sr, duty=ECHO_DUTY, octave_shift=12 * int(melody_octave), syllables=syl)
            mel = mel + melody_gain * ECHO_GAIN * _delay(eco, int(round(echo_delay_seconds(score.tempo) * sr)))
    elif melody_style in ("voice", "voice_cont") and contour is not None:
        mel = melody_gain * render_voice_melody(contour, total, sr, duty=melody_duty, snap=melody_style == "voice",
                                                octave_shift=12 * int(melody_octave))
        if echo:
            eco = render_voice_melody(contour, total, sr, duty=ECHO_DUTY, snap=melody_style == "voice",
                                      octave_shift=12 * int(melody_octave))
            mel = mel + melody_gain * ECHO_GAIN * _delay(eco, int(round(echo_delay_seconds(score.tempo) * sr)))
    elif melody_style == "follow" and contour is not None:
        mel = melody_gain * render_follow_melody(score.melody, contour, total, sr, duty=melody_duty,
                                                 octave_shift=12 * int(melody_octave))
        if echo:
            eco = render_follow_melody(score.melody, contour, total, sr, duty=ECHO_DUTY, octave_shift=12 * int(melody_octave))
            mel = mel + melody_gain * ECHO_GAIN * _delay(eco, int(round(echo_delay_seconds(score.tempo) * sr)))
    else:
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
