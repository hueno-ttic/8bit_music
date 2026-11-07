"""ボーカル分離モード: Demucs で分けたステムごとに採譜する.

  vocals → メロディ (pyin + 平滑化)。歌が無い区間は other ステムの主旋律 (basic-pitch スカイライン) で補う
  bass   → ベース (pyin, 低域)
  other  → コード (chroma)
  drums  → ドラム
"""
from __future__ import annotations

import logging

import librosa
import numpy as np
from scipy.ndimage import median_filter

from . import analysis as A
from .analysis import HOP, SR, Score
from .separation import separate

log = logging.getLogger(__name__)

BEAT_CELLS = 4          # 1 拍 = 16 分音符 4 つ
BAR_CELLS = 16
MEL_DIV = 2             # メロディは 16 分セルをさらに 2 分割 (32 分) で追跡
MEL_BAR = BAR_CELLS * MEL_DIV


def _rms(y: np.ndarray, frame_length: int) -> np.ndarray:
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=HOP)[0]


def _cell_max(x: np.ndarray, cells: np.ndarray, sr: int) -> np.ndarray:
    out = np.zeros(len(cells) - 1, dtype=np.float32)
    for i, sl in A._cell_slices(cells, len(x), sr):
        out[i] = x[sl].max() if x[sl].size else 0.0
    return out


# --------------------------------------------------------------------------- 平滑化

def _drop_isolated(seq: list[int | None]) -> list[int | None]:
    """前後どちらとも 7 半音以上離れた 1 セルだけの音を消す (ノイズ由来の誤検出)."""
    s = list(seq)
    for i in range(len(s)):
        if s[i] is None:
            continue
        if i > 0 and s[i - 1] == s[i]:
            continue
        if i + 1 < len(s) and s[i + 1] == s[i]:
            continue
        prev = next((s[j] for j in range(i - 1, max(-1, i - 4), -1) if s[j] is not None), None)
        nxt = next((s[j] for j in range(i + 1, min(len(s), i + 4)) if s[j] is not None), None)
        far = lambda m: m is None or abs(m - s[i]) > 7  # noqa: E731
        if far(prev) and far(nxt):
            s[i] = None
    return s


def _drop_register_outliers(seq: list[int | None], window: int = 2 * MEL_BAR, max_dev: int = 10,
                            max_len: int = 2 * MEL_DIV) -> list[int | None]:
    """前後 2 小節の中央値から大きく外れた短い音を消す (歌ステムに混ざった伴奏やオクターブ誤り)."""
    s = list(seq)
    n = len(s)
    i = 0
    while i < n:
        if s[i] is None:
            i += 1
            continue
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if j - i <= max_len:
            ctx = [s[k] for k in range(max(0, i - window), min(n, j + window)) if s[k] is not None and not (i <= k < j)]
            if len(ctx) >= 8 and abs(s[i] - float(np.median(ctx))) > max_dev:
                for k in range(i, j):
                    s[k] = None
        i = j
    return s


def _smooth_melody(seq: list[int | None]) -> list[int | None]:
    """細切れ対策:
       1. 同じ音に挟まれた 1 セルの空白を埋める
       2. 直前の音から 2 半音以内の 1 セルの音 (ビブラート/しゃくり) は直前の音に吸収する
    """
    s = list(seq)
    n = len(s)
    for i in range(1, n - 1):
        if s[i] is None and s[i - 1] is not None and s[i - 1] == s[i + 1]:
            s[i] = s[i - 1]
    for i in range(1, n):
        if s[i] is None or s[i - 1] is None or s[i] == s[i - 1]:
            continue
        single = (i + 1 >= n) or (s[i + 1] != s[i])
        if single and abs(s[i] - s[i - 1]) <= 2:
            s[i] = s[i - 1]
    return s


MEL_HOP = 256  # メロディ追跡は細かい hop (11.6 ms) で


def _f0_pyin(y: np.ndarray, sr: int, fmin: float, fmax: float):
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=MEL_HOP, fill_na=np.nan)
    return f0, np.where(voiced, np.maximum(prob, 0.5), prob)  # (f0, 確信度 0-1)


def _f0_crepe(y: np.ndarray, sr: int, fmin: float, fmax: float):
    """CREPE (torchcrepe, tiny) で f0 と確信度 (周期性) を MEL_HOP 間隔で返す. 使えなければ None."""
    try:
        import torch
        import torchcrepe
    except Exception:
        return None
    from .separation import _device

    csr = 16000
    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=csr)
    hop16 = int(round(MEL_HOP * csr / sr))  # 11.6 ms 相当
    x = torch.from_numpy(y16)[None]
    device = _device()
    if device == "cuda":
        device = "cuda"
    try:
        f0, per = torchcrepe.predict(x, csr, hop_length=hop16, fmin=fmin, fmax=fmax, model="tiny", device=device,
                                     return_periodicity=True, batch_size=1024, decoder=torchcrepe.decode.viterbi)
    except Exception as e:  # noqa: BLE001
        log.warning("CREPE (%s) に失敗したので pyin を使います: %s", device, e)
        return None
    per = torchcrepe.filter.median(per, 5)
    f0 = f0[0].cpu().numpy().astype(np.float64)
    per = per[0].cpu().numpy().astype(np.float64)
    # 目標フレーム数 (MEL_HOP @ sr) に合わせて長さを揃える
    n_target = int(np.ceil(len(y) / MEL_HOP)) + 1
    idx = np.clip(np.round(np.arange(n_target) * MEL_HOP / sr * csr / hop16).astype(int), 0, len(f0) - 1)
    return f0[idx], per[idx]


def _vocal_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """歌ステムの発音の立ち上がり (シラブル境界) の時刻."""
    on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=MEL_HOP, aggregate=np.median)
    if on.max() <= 0:
        return np.zeros(0)
    on = on / (np.percentile(on, 98) + 1e-9)
    pk = librosa.util.peak_pick(on, pre_max=4, post_max=4, pre_avg=8, post_avg=8, delta=0.12, wait=4)
    pk = [p for p in pk if on[p] >= 0.25]
    return librosa.frames_to_time(np.asarray(pk, dtype=int), sr=sr, hop_length=MEL_HOP)


def _track_melody(y: np.ndarray, sr: int, cells: np.ndarray, fmin: float, fmax: float,
                  min_voiced_ratio: float, gate_ratio: float, sensitivity: float = 0.5,
                  use_crepe: bool = True):
    """モノフォニックに f0 を追跡し、cells (32 分音符) 単位の MIDI 列・ベロシティ・区切りフラグを返す.
    sensitivity: 0.3 (厳しめ) 〜 0.7 (拾いやすい)。確信度のしきい値と音量ゲートに効く。"""
    # CREPE と pyin の両方で追跡し、どちらかが自信を持って有声と言えばその f0 を使う
    f0p, confp = _f0_pyin(y, sr, fmin, fmax)
    voiced_p = (confp >= 0.5) & np.isfinite(f0p)
    res = _f0_crepe(y, sr, fmin, fmax) if use_crepe else None
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=MEL_HOP)[0]
    if res is None:
        n = min(len(f0p), len(rms))
        f0, voiced, rms = f0p[:n], voiced_p[:n], rms[:n]
    else:
        f0c, confc = res
        thr = 0.70 - 0.5 * sensitivity  # 0.5 → 0.45, 0.7 → 0.35, 0.3 → 0.55
        n = min(len(f0c), len(f0p), len(rms))
        f0c, confc, f0p, voiced_p, rms = f0c[:n], confc[:n], f0p[:n], voiced_p[:n], rms[:n]
        voiced_c = (confc >= thr) & np.isfinite(f0c) & (f0c > 0)
        f0 = np.where(voiced_c, f0c, np.where(voiced_p, f0p, np.nan))
        voiced = voiced_c | voiced_p
    # ビブラート / しゃくりをならす (約 60 ms の中央値)
    midi = librosa.hz_to_midi(np.where(voiced, f0, 1.0))
    midi_s = median_filter(midi, size=5, mode="nearest")
    ref = np.percentile(rms, 95) + 1e-9
    gate = ref * gate_ratio * (1.5 - sensitivity)  # 感度が高いほどゲートを下げる
    fr = np.clip(np.floor(cells * sr / MEL_HOP).astype(int), 0, n)
    seq: list[int | None] = []
    vel: list[float] = []
    for i in range(len(cells) - 1):
        a, b = fr[i], max(fr[i + 1], fr[i] + 1)
        v = voiced[a:b]
        e = rms[a:b]
        if v.size == 0 or v.mean() < min_voiced_ratio or e.max() < gate:
            seq.append(None)
            vel.append(0.0)
            continue
        seq.append(int(np.round(np.median(midi_s[a:b][v]))))
        vel.append(float(np.clip(0.5 + 0.5 * e.max() / ref, 0.4, 1.0)))
    seq = A._fix_octave_glitches(seq)
    seq = _drop_isolated(seq)
    seq = _drop_register_outliers(seq)
    seq = _smooth_melody(seq)
    # 発音の立ち上がりがあるセルには区切りフラグを立てる (同じ高さで歌い直す部分を分ける)
    onsets = _vocal_onsets(y, sr)
    split = [False] * (len(cells) - 1)
    if len(onsets):
        idx = np.clip(np.searchsorted(cells, onsets, side="right") - 1, 0, len(cells) - 2)
        for ci, t in zip(idx, onsets):
            # セル境界に近い方へ寄せる
            if t - cells[ci] > (cells[ci + 1] - cells[ci]) / 2 and ci + 1 < len(split):
                ci += 1
            split[ci] = True
    return seq, vel, split


def _track_melody_hmm(y: np.ndarray, sr: int, mcells: np.ndarray, sensitivity: float = 0.5, quantize: bool = True, progress=None):
    """RMVPE で f0 を取り、HMM で音符に分割する (格子に丸めない)。戻り値: (音符 [(s,e,midi,vel)], セル列)."""
    from . import notes as N
    from .pitch import rmvpe_f0

    t, f0 = rmvpe_f0(y, sr, thred=0.03)
    # 歌ステムの音量で無声を判定 (RMVPE は伴奏の混入にも反応するため)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=int(round(N.HOP_S * sr)))[0]
    n = min(len(f0), len(rms))
    t, f0, rms = t[:n], f0[:n], rms[:n]
    ref = np.percentile(rms, 95) + 1e-9
    gate = ref * 0.04 * (1.5 - sensitivity)
    f0 = np.where(rms >= gate, f0, 0.0)
    energy = np.clip(0.5 + 0.5 * rms / ref, 0.4, 1.0)
    raw = N.hmm_segment(t, f0, stay=HMM_STAY, sigma=HMM_SIGMA, min_dur=HMM_MIN_DUR, unv_stay=HMM_UNV_STAY, energy=energy)
    raw = N.drop_register_outliers(raw)
    if quantize:
        raw = N.musicalize(raw, mcells, beat_cells=BEAT_CELLS * MEL_DIV)
    else:
        raw = N.snap_onsets(raw, mcells, tol=0.03)
    # 元の歌のフレーズに従って接続: フレーズ内は無音を挟まず、フレーズの切れ目だけ休符
    phrases = N.phrases_from_voicing(t, f0 > 0)
    raw = N.connect_by_phrases(raw, phrases, mcells, t=t, f0=f0)
    raw = N.fill_phrase_holes(raw, phrases, mcells, t, f0)
    raw = N.merge_passing_notes(raw)
    raw = N.split_at_onsets(raw, N.syllable_valleys(y, sr), mcells)
    from . import verify as V
    midi_all = librosa.hz_to_midi(np.maximum(f0, 1e-3))
    raw, report = V.verify_loop(raw, midi_all, f0 > 0, max_iter=VERIFY_MAX_ITER, target=VERIFY_TARGET, progress=progress, log=log)
    LAST_VERIFY.clear(); LAST_VERIFY.update(report)
    seq = N.seq_from_notes([(a, b, m, v) for a, b, m, v, _ in raw], mcells)
    return raw, seq


HMM_STAY, HMM_SIGMA, HMM_MIN_DUR, HMM_UNV_STAY = 0.998, 0.8, 0.08, 0.9  # vocadito で探索 (min_dur は穴を減らすため 0.12→0.08)


def _note_pitch(midi_frames: np.ndarray, voiced: np.ndarray, energy: np.ndarray | None = None) -> float | None:
    """音符区間の音高 (連続値)。頭 (しゃくり) と尻を除いた安定区間の、エネルギー重み付き中央値."""
    n = len(midi_frames)
    if n == 0:
        return None
    head = min(6, n // 3)          # 最大 60 ms
    tail = min(3, max(0, (n - head) // 4))
    sel = np.zeros(n, bool); sel[head:n - tail] = True
    sel &= voiced
    if sel.sum() < 3:
        sel = voiced.copy()
        if sel.sum() < 3:
            return None
    vals = midi_frames[sel]
    if energy is not None:
        w = energy[sel]; order = np.argsort(vals); c = np.cumsum(w[order]) / w[order].sum()
        return float(vals[order][int(np.searchsorted(c, 0.5))])
    return float(np.median(vals))


def _round_in_key(p: float, scale: set[int] | None, amb: float = 0.3) -> int:
    """連続値の音高を整数に。半音の境目 (小数部が amb 以上) なら、調に合う方の隣を選ぶ."""
    lo, hi = int(np.floor(p)), int(np.ceil(p))
    if lo == hi:
        return lo
    frac = p - lo
    if scale is not None and min(frac, 1 - frac) >= amb:
        lo_in, hi_in = lo % 12 in scale, hi % 12 in scale
        if lo_in and not hi_in:
            return lo
        if hi_in and not lo_in:
            return hi
    return int(np.round(p))


F0_THRED = 0.03     # RMVPE の有声判定しきい値
F0_GATE = 0.04      # 歌ステムの音量がこの割合 (95 パーセンタイル比) 未満のフレームは無声にする
F0_OCTAVE_FIX = False


def fix_octave_jumps(midi: np.ndarray, voiced: np.ndarray, max_len: float = 0.6, ctx: float = 1.5) -> np.ndarray:
    """f0 のオクターブ飛び (分離の残渣・倍音への誤追従) を直す: 前後と 6 半音以上離れて切れた短い有声区間
    (max_len 秒未満) が、周囲 ±ctx 秒の音域から 9 半音以上外れ、1 オクターブずらすとずっと近くなるなら、ずらす."""
    m = midi.copy()
    n = len(m)
    hop = 0.01
    seg_start = None
    segs = []
    for i in range(n + 1):
        brk = i == n or not voiced[i] or (seg_start is not None and abs(m[i] - m[i - 1]) > 6)
        if brk:
            if seg_start is not None:
                segs.append((seg_start, i))
            seg_start = i if i < n and voiced[i] else None
        elif seg_start is None:
            seg_start = i
    for a, b in segs:
        if (b - a) * hop >= max_len:
            continue
        lo, hi = max(0, a - int(ctx / hop)), min(n, b + int(ctx / hop))
        sel = voiced[lo:hi].copy()
        sel[a - lo:b - lo] = False
        if sel.sum() < 20:
            continue
        c = float(np.median(m[lo:hi][sel]))
        med = float(np.median(m[a:b]))
        if abs(med - c) < 9:
            continue
        for sh in (12, -12):
            if abs(med + sh - c) < abs(med - c) - 6:
                m[a:b] += sh
                break
    return m


TOP_VOICE = True      # 歌ステムに同時に複数の声 (ハモリ) があるとき、一番上の声を主旋律として追う
TOP_VOICE_MIN = 0.15  # 上の声がこの秒数以上続くときだけ切り替える
TOP_VOICE_AMP = 0.6   # 上の声の強さが、いま追っている声のこの割合以上のとき


def prefer_top_voice(f0: np.ndarray, y: np.ndarray, sr: int, hop: float = 0.01) -> np.ndarray:
    """f0 (RMVPE: いちばん強い声を追う) が下のハモリを追っている所を、上の声 (主旋律) に置き換える.
    J-POP / ボカロのハモリは主旋律の下に付けることが多く、サビでは歌ステムに 2 声以上が重なる。RMVPE は強い方の声を
    追うので、サビで主旋律とハモリの間を行き来し、旋律としてつながらなくなっていた。
    歌ステムを basic-pitch で多声採譜し、いま追っている声よりオクターブ以外の音程で上に、十分強い声が続いていればそちらにする."""
    from . import analysis_ml
    if not analysis_ml.is_available():
        return f0
    try:
        events = analysis_ml._predict_notes(y / (np.abs(y).max() + 1e-9), sr)
    except Exception as e:  # noqa: BLE001
        log.warning("多声採譜に失敗したので上の声の選び直しをしません: %s", e)
        return f0
    n = len(f0)
    top = np.full(n, np.nan)       # 各フレームの、追っている声より上の最も高い強い声
    top_amp = np.zeros(n)
    cur_amp = np.zeros(n)
    m = np.where(f0 > 0, librosa.hz_to_midi(np.maximum(f0, 1e-3)), np.nan)
    for s_, e_, mi, a in events:
        if not 48 <= mi <= 92:
            continue
        a0, a1 = int(s_ / hop), min(n, int(e_ / hop))
        for k in range(a0, a1):
            if not np.isfinite(m[k]):
                continue
            d = mi - m[k]
            if abs(d) < 0.7:
                cur_amp[k] = max(cur_amp[k], a)
            elif d >= 1.5 and int(round(d)) % 12 != 0 and (not np.isfinite(top[k]) or mi > top[k]):
                top[k], top_amp[k] = mi, a
    use = np.isfinite(top) & (top_amp >= TOP_VOICE_AMP * np.maximum(cur_amp, 1e-3))
    out = f0.copy()
    k = 0
    min_len = int(TOP_VOICE_MIN / hop)
    changed = 0
    while k < n:
        if not use[k]:
            k += 1
            continue
        j = k
        while j < n and use[j]:
            j += 1
        if j - k >= min_len:
            # 上の声の高さ (整数 MIDI) に、元の f0 の細かい揺れ (半音以下) を足す
            frac = m[k:j] - np.round(m[k:j])
            out[k:j] = librosa.midi_to_hz(top[k:j] + np.clip(frac, -0.5, 0.5))
            changed += j - k
        k = j
    if changed:
        log.info("ハモリの下の声を追っていた %.1f 秒を、上の声 (主旋律) に置き換えました", changed * hop)
    return out


_VF0_CACHE: dict = {}


def vocal_f0(y: np.ndarray, sr: int):
    """歌ステムの f0 (Hz, 10 ms, 無声=0)・音量・エネルギー. 楽譜モデルの経路で共通に使う.
    同じ歌ステムに対しては 1 回の解析の中で 1 度だけ計算する (RMVPE と多声採譜で 15〜25 秒かかる)."""
    key = (id(y), y.shape, float(np.sum(y[::997])), sr, F0_THRED, F0_GATE, F0_OCTAVE_FIX, TOP_VOICE)
    if key in _VF0_CACHE:
        return _VF0_CACHE[key]
    from . import budget
    from . import notes as N
    from .pitch import rmvpe_f0
    t, f0 = rmvpe_f0(y, sr, thred=F0_THRED)
    if TOP_VOICE:
        if budget.enough(70):
            f0 = prefer_top_voice(f0, y, sr)
        else:
            log.info("時間が足りないので、ハモリの上の声の選び直しを省きます")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=int(round(N.HOP_S * sr)))[0]
    n = min(len(f0), len(rms)); t, f0, rms = t[:n], f0[:n], rms[:n]
    ref = np.percentile(rms, 95) + 1e-9
    f0 = np.where(rms >= ref * F0_GATE, f0, 0.0)
    if F0_OCTAVE_FIX:
        v = f0 > 0
        mm = fix_octave_jumps(librosa.hz_to_midi(np.maximum(f0, 1e-3)), v)
        f0 = np.where(v, librosa.midi_to_hz(mm), 0.0)
    energy = np.clip(0.5 + 0.5 * rms / ref, 0.4, 1.0)
    _VF0_CACHE.clear()
    _VF0_CACHE[key] = (t, f0, rms, energy)
    return t, f0, rms, energy


def _melody_from_lyrics(y: np.ndarray, sr: int, chars: list[tuple[float, float]], mcells: np.ndarray,
                        key: tuple[int, str] | None = None):
    """歌詞の文字 (モーラ) ごとの発声区間を音符にし、音高は歌ステムの RMVPE f0 で決める.
    - 音高は頭・尻を除いた安定区間の中央値。半音の境目なら調に合う方へ
    - 1 文字の中で 2 半音以上の段階的な音程変化があれば分割 (メリスマ)
    - 40 ms 未満の文字は隣に併合、歌ステムに音程が無い文字は落とす
    - 歌詞に無い歌の区間は解析 (HMM + 音量の谷) で補う
    - 隣接する文字は legato"""
    from . import notes as N
    from .pitch import rmvpe_f0
    from .harmony import scale_pcs

    scale = scale_pcs(*key) if key else None
    t, f0, rms, energy = vocal_f0(y, sr)
    midi_all = librosa.hz_to_midi(np.maximum(f0, 1e-3))
    valleys = N.syllable_valleys(y, sr, prominence_db=3.0, min_width_ms=20.0)  # 文字内分割の裏付けに使う
    # 短い文字を隣に併合
    merged: list[list[float]] = []
    for s_, e_ in chars:
        if merged and s_ - merged[-1][1] < 0.03 and (e_ - s_ < 0.04 or merged[-1][1] - merged[-1][0] < 0.04):
            merged[-1][1] = max(merged[-1][1], e_)
        else:
            merged.append([s_, e_])
    raw = []   # (start, end, 連続音高, vel)
    for s_, e_ in merged:
        a, b = int(s_ / N.HOP_S), max(int(e_ / N.HOP_S), int(s_ / N.HOP_S) + 1)
        seg = f0[a:b]; v = seg > 0
        if v.size == 0 or v.mean() < 0.3:
            # 音程が取れない文字 (子音主体・弱い発声): 直前の音があれば同じ高さで歌い直しとして立てる
            if raw and s_ - raw[-1][1] < 0.05 and (e_ - s_) >= 0.06 and energy[a:b].size and energy[a:b].max() >= 0.45:
                raw.append((s_, e_, raw[-1][2], float(np.clip(energy[a:b].max(), 0.4, 1.0))))
            continue
        vel = float(np.clip(energy[a:b].max(), 0.4, 1.0)) if energy[a:b].size else 0.9
        # 文字内のメリスマ: HMM で分割し、2 半音以上違う区間だけ別の音にする
        sub = N.hmm_segment(t[a:b], seg, stay=HMM_STAY, sigma=HMM_SIGMA, min_dur=0.07, unv_stay=HMM_UNV_STAY, energy=energy[a:b]) if (b - a) * N.HOP_S >= 0.20 else []
        sub = [x for x in sub if x[1] - x[0] >= 0.07]
        parts = []
        for x in sub:
            if parts and abs(x[2] - parts[-1][2]) < 2:
                parts[-1] = (parts[-1][0], x[1], parts[-1][2])
            else:
                parts.append((x[0], x[1], x[2]))
        # 分割点には発音の証拠 (音量の谷) か大きな跳躍 (4 半音以上) を要求する。単なるベンドは 1 音のまま
        if len(parts) >= 2:
            kept = [parts[0]]
            for x in parts[1:]:
                jump = abs(x[2] - kept[-1][2])
                evidence = len(valleys) and np.min(np.abs(valleys - x[0])) <= 0.05
                if jump >= 4 or evidence:
                    kept.append(x)
                else:
                    kept[-1] = (kept[-1][0], x[1], kept[-1][2])
            parts = kept
        if len(parts) >= 2:
            pts = [s_] + [float(x[0]) for x in parts[1:]] + [e_]
            for k in range(len(parts)):
                pa, pb = int(pts[k] / N.HOP_S), max(int(pts[k + 1] / N.HOP_S), int(pts[k] / N.HOP_S) + 1)
                p_ = _note_pitch(midi_all[pa:pb], f0[pa:pb] > 0, energy[pa:pb])
                if p_ is not None:
                    raw.append((pts[k], pts[k + 1], p_, vel))
        else:
            p_ = _note_pitch(midi_all[a:b], v, energy[a:b])
            if p_ is not None:
                raw.append((s_, e_, p_, vel))
    # 整数化: 半音の境目は調に合う方へ。隣接する文字が半音差で片方が境目付近なら隣に揃える
    ints = [_round_in_key(p_, scale) for _, _, p_, _ in raw]
    for i in range(1, len(raw)):
        if abs(ints[i] - ints[i - 1]) == 1 and raw[i][0] - raw[i - 1][1] < 0.03:
            fi = abs(raw[i][2] - np.round(raw[i][2])); fp = abs(raw[i - 1][2] - np.round(raw[i - 1][2]))
            if fi >= 0.3 and fp < 0.2 and abs(raw[i][2] - ints[i - 1]) <= 0.7:
                ints[i] = ints[i - 1]
            elif fp >= 0.3 and fi < 0.2 and abs(raw[i - 1][2] - ints[i]) <= 0.7:
                ints[i - 1] = ints[i]
    raw = [(s_, e_, m, v) for (s_, e_, _, v), m in zip(raw, ints)]
    raw = N.drop_register_outliers(raw)
    # 歌詞データに無い歌 (コーラス、ラララ、繰り返し) の区間は、解析による区切り (HMM + 音量の谷) で補う
    phrases = N.phrases_from_voicing(t, f0 > 0)
    covered = np.zeros(len(t), bool)
    for s_, e_ in chars:
        covered[int(s_ / N.HOP_S):int(e_ / N.HOP_S)] = True
    extra = []
    for ps, pe in phrases:
        a, b = int(ps / N.HOP_S), int(pe / N.HOP_S)
        if b - a < 10 or covered[a:b].mean() >= 0.3:
            continue
        seg = N.hmm_segment(t[a:b], f0[a:b], stay=HMM_STAY, sigma=HMM_SIGMA, min_dur=HMM_MIN_DUR, unv_stay=HMM_UNV_STAY, energy=energy[a:b])
        seg = N.drop_register_outliers(seg)
        seg = N.split_at_onsets([(x[0], x[1], x[2], x[3], False) for x in seg], N.syllable_valleys(y[int(ps * sr):int(pe * sr)], sr) + ps, mcells)
        extra.extend((x[0], x[1], x[2], x[3]) for x in seg)
    if extra:
        log.info("歌詞に無い区間を解析で補完: %d 音", len(extra))
        raw = sorted(raw + extra, key=lambda x: x[0])
    # 文字の終わりは実際の発声より早めに付いていることが多いので、歌のフレーズに従って
    # 次の文字までつなぎ、フレーズ末尾は歌の終わりまで伸ばす (HMM 経路と同じ処理)
    out = N.connect_by_phrases(raw, phrases, mcells, t=t, f0=f0)
    out = N.fill_phrase_holes(out, phrases, mcells, t, f0)
    # 最終パス: 補完・接続で増えた音も含め、全ての音の音高を安定区間 + 調で決め直す
    fixed = []
    for s_, e_, m, v, leg in out:
        a, b = int(s_ / N.HOP_S), max(int(e_ / N.HOP_S), int(s_ / N.HOP_S) + 1)
        p_ = _note_pitch(midi_all[a:b], f0[a:b] > 0, energy[a:b])
        if p_ is not None and abs(p_ - m) < 2.5:
            m = _round_in_key(p_, scale)
        fixed.append((s_, e_, m, v, leg))
    out = fixed
    seq = N.seq_from_notes([(a, b, m, v) for a, b, m, v, _ in out], mcells)
    return out, seq


# マジカルミライ 6 曲 + vocadito で調整。歌詞の文字の頭はそのまま音の頭にする (refine / refine_bnd = False):
# 楽譜の分かっている評価曲 (tools/eval_synth.py) で音符 F 値 81→85、実曲でも文字の頭は f0 の変わり目と同じくらい拍の格子に揃う
SM_PARAMS = dict(w_score=3.0, gamma=0.8, on=2.0, inside=-3.0, cross=-6.0, rest_voiced=-5.0, plateau=True,
                 refine=False, refine_bnd=False)


def _melody_from_score_model(y: np.ndarray, sr: int, chars: list[tuple[float, float]], beats: list[dict],
                             mcells: np.ndarray, key: tuple[int, str], chords: list | None = None, progress=None):
    """楽譜モデル + F0 モデル (Nishikimi らの半マルコフ Viterbi) で音符列を推定する.
    - 拍節格子は Songle の拍から (32 分)、歌詞の文字境界で格子点を発声時刻に寄せる
    - 調 (音階) の事前、音程の事前、音価とリズムの事前、歌詞境界の事前 (文字の頭で音を切り直す)
    - f0 は Cauchy 分布で観測 (ビブラート・しゃくりに頑健、オクターブ飛びは弱い証拠として扱う)
    - 同じ高さで根拠なく割れた音は併合し、ごみ音を落とし、フレーズに従って legato 接続"""
    from . import notes as N
    from .pitch import rmvpe_f0

    t, f0, rms, energy = vocal_f0(y, sr)
    out = score_model_notes(t, f0, energy, chars, beats, key, chords=chords, progress=progress)
    seq = N.seq_from_notes([(a, b, m, v) for a, b, m, v, _ in out], mcells)
    return out, seq


USE_ONSET_MODEL = True      # 歌詞タイミングが無い曲で、検出した音節 + 楽譜モデルを使う (False なら従来の HMM)
# 検出した音節で楽譜モデルを使うときの SM_PARAMS の上書き (開発 6 曲を交差検証した検出器で調整: tools/eval_8bit.py --folds)
#   soft_floor: 検出の山が無くても音高が変われば音を始められる下限 / soft_same_floor: 同じ高さの歌い直しには山を要求
ONSET_SM_PARAMS: dict = {"soft_on": 3.0, "soft_cross": 10.0, "soft_base": 0.02, "soft_floor": 0.06, "soft_same_floor": 0.001,
                         "refine": True, "refine_bnd": True}   # 検出した音節の頭は不確かなので f0 の変わり目に合わせる
ONSET_THR: float | None = None  # 音節検出のしきい値 (None ならモデルに付属の値)
ONSET_SOFT = True               # 検出確率を連続的な事前として使う (False なら山を歌詞の文字と同じ扱いにする)


def _beats_from_cells(cells: np.ndarray) -> list[dict]:
    """16 分セル境界 → Songle 形式の拍 [{start (ms), position 1..4}] (小節の位相は不明なので仮に置く)."""
    bt = np.asarray(cells[::BEAT_CELLS], dtype=float)
    if len(bt) > 3:
        d = np.diff(bt)
        med = float(np.median(d))
        # 曲頭に挿入された 0 や曲末の端数など、拍間隔から大きく外れた境界は除く
        keep = [0] + [i + 1 for i in range(len(d)) if 0.7 * med <= d[i] <= 1.3 * med]
        if len(bt) > 1 and not (0.7 * med <= d[0] <= 1.3 * med):
            keep = keep[1:]
        bt = bt[keep]
    return [{"start": float(b * 1000.0), "position": int(i % 4) + 1} for i, b in enumerate(bt)]


def _melody_from_detected_onsets(y: np.ndarray, sr: int, cells: np.ndarray, mcells: np.ndarray,
                                 key: tuple[int, str], progress=None):
    """歌詞タイミングが無い曲: 歌ステムから音節の始まりを検出し (chiptune.onset_model)、
    それを歌詞の文字境界の代わりにして楽譜モデルで音符列を推定する. 拍節格子はビート検出の結果から作る."""
    from . import notes as N, onset_model as OM
    from .pitch import rmvpe_f0

    t, f0, rms, energy = vocal_f0(y, sr)
    onsets, post = OM.detect(y, sr, t, f0, rms, thr=ONSET_THR)
    chars = OM.pseudo_chars(onsets, t, f0 > 0)
    log.info("歌ステムから音節の始まりを %d 個検出し、楽譜モデルの区切りに使います", len(chars))
    beats = _beats_from_cells(cells)
    out = score_model_notes(t, f0, energy, chars, beats, key, params=dict(ONSET_SM_PARAMS) or None, progress=progress,
                            onset_post=post if ONSET_SOFT else None)
    seq = N.seq_from_notes([(a, b, m, v) for a, b, m, v, _ in out], mcells)
    return out, seq


VERIFY_MAX_ITER = 5
VERIFY_TARGET = 0.001
LAST_VERIFY: dict = {}   # 直近の自己検証レポート (analyze_sep が Score に載せる)
LAST_SYLLABLES: list = []  # 直近の楽譜モデルで使った音節の頭 [秒] (歌詞の文字, または検出した音節)
LAST_SYLLABLE_SPANS: list = []  # 同じく音節の区間 [(開始, 終了)]


def score_model_notes(t, f0, energy, chars, beats, key, chords=None, params: dict | None = None, progress=None,
                      onset_post: np.ndarray | None = None):
    """f0 (Hz, 10 ms, 0=無声) から音符列 [(start, end, midi, vel, legato)] を作る (楽譜モデル経路の本体).
    onset_post: 歌詞の代わりに音節検出器の確率 (10 ms ごと) を使う場合に渡す。chars は検出した山から作った仮の文字区間で、
    格子を寄せるのに使い、音の始まり/またぎの事前は確率から連続的に作る (取りこぼし・誤検出に強い)."""
    from . import notes as N, score_model as SM, melody_clean as MC
    from .harmony import scale_pcs as _spcs

    midi_all = librosa.hz_to_midi(np.maximum(f0, 1e-3))
    voiced = f0 > 0
    chars = SM.merge_short_chars(chars)
    if p_refine := (params or {}).get("refine", SM_PARAMS.get("refine", True)):
        chars = SM.refine_onsets(chars, t, midi_all, voiced)
    LAST_SYLLABLES[:] = [float(c[0]) for c in chars]   # 音節の頭 (8bit で言い直す所: synth.follow_frames)
    LAST_SYLLABLE_SPANS[:] = [(float(a), float(b)) for a, b in chars]
    cells16, cpos = SM.cells_from_beats(beats, float(t[-1]) + 0.1)
    lattice = SM.warp_lattice(SM.make_lattice(cells16, MEL_DIV), [c[0] for c in chars])
    beat_pos = (np.repeat(cpos, MEL_DIV) * MEL_DIV + np.tile(np.arange(MEL_DIV), len(cpos)))[: len(lattice) - 1]
    p = dict(SM_PARAMS); p.update(params or {})
    landing = bool(p.pop("landing", False)); plateau = bool(p.pop("plateau", False)); p.pop("refine", None)
    p_defrag = bool(p.pop("defrag", False))   # 細切れの音をまとめる (検出した音節を使うときは常に行う)
    p_refine_bnd = bool(p.pop("refine_bnd", True))   # 音高が変わる境界を f0 の変わり目に合わせる
    p_merge_same = bool(p.pop("merge_same", True))
    on_w, inside_w, cross_w = p.pop("on"), p.pop("inside"), p.pop("cross")
    soft_on, soft_cross, soft_base = p.pop("soft_on", 1.0), p.pop("soft_cross", 3.0), p.pop("soft_base", 0.15)
    soft_floor, soft_same_floor = p.pop("soft_floor", 1e-3), p.pop("soft_same_floor", None)
    if onset_post is not None:
        pri = SM.soft_onset_priors(lattice, onset_post, soft_on, soft_cross, soft_base, soft_floor, soft_same_floor)
        onset_lp, cross_lp = pri[0], pri[1]
        if len(pri) > 2:
            p["same_prior"] = pri[2]
    else:
        onset_lp, cross_lp = SM.lyric_priors(lattice, chars, on_w, inside_w, cross_w)
    est = SM.estimate_notes(t, midi_all, voiced, lattice, tonic=key[0], mode=key[1], div=MEL_DIV,
                            onset_prior=onset_lp, beat_pos=beat_pos, cross_prior=cross_lp, **p)
    if p_merge_same:
        # 同じ高さで根拠なく割れた音をつなぐ (歌詞の文字境界を根拠にする)。検出器の確率で区切った場合は、
        # 楽譜モデルがすでに確率を見て歌い直しかどうかを決めているので行わない (行うと歌い直しの 1〜2 割が消えた)
        est = SM.merge_same_pitch(est, [c[0] for c in chars])
    raw = []
    for s_, e_, m in est:
        a, b = int(s_ / N.HOP_S), max(int(e_ / N.HOP_S), int(s_ / N.HOP_S) + 1)
        vel = float(np.clip(energy[a:b].max(), 0.4, 1.0)) if energy[a:b].size else 0.9
        raw.append((s_, e_, m, vel))
    log.info("楽譜モデルで %d 音 (格子 %d 点, 歌詞 %d 文字)", len(raw), len(lattice), len(chars))
    chord_at = None
    if chords:
        _parsed = [(cs, ce, set(pc[2])) for cs, ce, pc in chords if pc]
        def chord_at(t_):
            for cs, ce, pcs in _parsed:
                if cs <= t_ < ce:
                    return pcs
            return None
    raw, cstats = MC.clean(raw, midi_all, voiced, _spcs(*key), chord_at, landing=landing, plateau=plateau)
    log.info("後処理: %s", cstats)
    phrases = N.phrases_from_voicing(t, voiced)
    out = SM.link_phrases(raw, t, midi_all, voiced, phrases)
    if onset_post is not None or p_defrag:
        out = N.defragment(out)
    if p_refine_bnd:
        out = N.refine_boundaries(out, midi_all, voiced)
    # 自己検証と修正の繰り返し (外れ <= 0.1% になるまで、上限 5 回)。実際に鳴らす音符列 (接続後) を検証する
    from . import verify as V
    out, report = V.verify_loop(out, midi_all, voiced, max_iter=VERIFY_MAX_ITER, target=VERIFY_TARGET, progress=progress, log=log,
                                onsets=[c[0] for c in chars])
    LAST_VERIFY.clear(); LAST_VERIFY.update(report)
    # 修正 (取りこぼしの補い等) で残った細切れを整理する (歌詞タイミングの経路も同じ)。
    # 検証で入れた短い音 (歌が留まった音) は消さないよう、ここでは同じ高さの断片と 50 ms 未満の切れ端だけ
    out = N.defragment(out, min_len=0.05)
    return out


SUNG_SEG_MIN = 0.04   # 歌がこの秒数以上はっきり留まった高さは、すべて音符にする
INSERT_SUNG_SEGMENTS = False
MELODY_METHOD = "syllable"   # "syllable" = 音節ごとに 1 音 / "model" = 楽譜モデルの音符


def insert_sung_segments(melody: list, contour_midi: np.ndarray, hop: float = 0.01, min_len: float = SUNG_SEG_MIN) -> list:
    """楽譜モデルの音符 (骨組み) に、歌がはっきり留まったのに鳴っていない高さを音符として差し込む.
    ボカロは 1 音節に 2 つ以上の音 (こぶし・速い経過音) を打ち込むことが多いが、文字単位の楽譜モデルと細切れ対策で
    それが消え、本物の曲で「歌が 50 ms 以上留まった高さ」の 13〜20% が鳴っておらず、聴くと間の音が抜けて別の曲に聞こえた."""
    from scipy.ndimage import median_filter
    from .analysis import Note
    m = np.asarray(contour_midi, dtype=float)
    ok = np.isfinite(m)
    sm = np.where(ok, median_filter(np.nan_to_num(m), 5), np.nan)
    q = np.where(ok & (np.abs(sm - np.round(sm)) <= 0.4), np.round(sm), np.nan)
    n = len(q)
    owner = np.full(n, -1)
    notes = sorted(melody, key=lambda x: x.start)
    for i, nt in enumerate(notes):
        owner[max(0, int(round(nt.start / hop))):min(n, int(round(nt.end / hop)))] = i
    cuts: dict[int, list] = {}
    i = 0
    L = max(2, int(round(min_len / hop)))
    while i < n:
        if not np.isfinite(q[i]):
            i += 1
            continue
        j = i
        while j < n and q[j] == q[i]:
            j += 1
        if j - i >= L:
            own = owner[i:j]
            k = int(np.bincount(own[own >= 0]).argmax()) if np.any(own >= 0) else -1
            if k >= 0 and notes[k].midi != int(q[i]) and np.mean(own == k) >= 0.6:
                cuts.setdefault(k, []).append((i * hop, j * hop, int(q[i])))
        i = j
    out = []
    for i, nt in enumerate(notes):
        if i not in cuts:
            out.append(nt)
            continue
        cur = nt.start
        for a, b, p in sorted(cuts[i]):
            a, b = max(a, nt.start), min(b, nt.end)
            if a - cur >= min_len:
                out.append(Note(cur, a, nt.midi, nt.velocity, True))
            elif out and a > cur and out[-1].end == cur:
                pass
            out.append(Note(max(cur, a) if a - cur >= min_len else cur, b, p, nt.velocity, True))
            cur = b
        if nt.end - cur >= min_len:
            out.append(Note(cur, nt.end, nt.midi, nt.velocity, nt.legato))
        elif out:
            out[-1] = Note(out[-1].start, nt.end, out[-1].midi, out[-1].velocity, nt.legato)
    log.info("歌がはっきり留まったのに鳴っていなかった高さ %d 個を音符にしました", sum(len(v) for v in cuts.values()))
    return out


SYL_MELISMA_MIN = 0.12   # 音節の中で 2 つ目の高さにこれ以上はっきり留まったら音を分ける
SYL_TAIL_MAX = 0.6       # フレーズの終わりの音を、歌の声が続く限り最大この秒数まで伸ばす


def syllable_notes(chars, midi_contour: np.ndarray, voiced: np.ndarray, level_db: np.ndarray | None = None,
                   hop: float = 0.01) -> list:
    """歌詞の音節ごとに 1 音を作る (ボカロの譜面は基本 1 音節 = 1 音符で打ち込まれる).
    - 音は音節の頭から次の音節の頭まで (間が空けば、歌の声が続く所まで)。歌詞の無い所には音を置かない
    - 高さは音節の後ろ 7 割の、歌が安定している所に重みを置いた中央値 (しゃくりを除く。知覚される音高は安定部に寄る)
    - 音節の中で 2 つ目の高さに SYL_MELISMA_MIN 秒以上はっきり留まる (こぶし・メリスマ) ときだけ音を分ける
    これまでの方式は文字の中の揺れまで音符にしたり、文字単位の事前で音を消したりして、聴くと「余計な音」「抜けた音」になった."""
    from scipy.ndimage import median_filter
    from .analysis import Note
    m = np.asarray(midi_contour, dtype=float)
    v = np.asarray(voiced, bool) & np.isfinite(m)
    n = len(m)
    sm = np.where(v, median_filter(np.nan_to_num(m), 5), np.nan)
    stab = np.zeros(n)
    d = np.abs(np.diff(np.nan_to_num(sm), prepend=0.0))
    stab[v] = 1.0 / (1.0 + (d[v] / 0.15) ** 2)
    out = []
    prev_pitch = None
    for i, (cs, ce) in enumerate(chars):
        nxt = chars[i + 1][0] if i + 1 < len(chars) else None
        a, b = int(round(cs / hop)), min(n, int(round(ce / hop)))
        if b - a < 3:
            continue
        # 終わり: 次の音節が近ければそこまで、離れていれば歌の声が続く所まで
        if nxt is not None and nxt - ce < 0.25:
            end = nxt
        else:
            k = b
            lim = min(n, b + int(SYL_TAIL_MAX / hop), int(round(nxt / hop)) if nxt is not None else n)
            while k < lim and (v[k] or (k + 1 < lim and v[k + 1])):
                k += 1
            end = max(ce, k * hop)
        e = min(n, int(round(end / hop)))
        core = np.arange(a + int(0.3 * (b - a)), max(b, a + int(0.3 * (b - a)) + 1))
        core = core[(core < n) & v[np.clip(core, 0, n - 1)]]
        if len(core) < 2:
            seg = np.arange(a, e)
            core = seg[(seg < n) & v[np.clip(seg, 0, n - 1)]]
        if len(core) < 2:
            if prev_pitch is not None and out and cs - out[-1].end < 0.2 and level_db is not None \
                    and np.max(level_db[a:max(a + 1, b)]) > -25:
                out.append(Note(cs, end, prev_pitch, 1.0, True))   # 音程の取れない文字 (子音主体): 同じ高さで歌い直す
            continue
        w = stab[core] + 1e-3
        vals = sm[core]
        order = np.argsort(vals)
        cw = np.cumsum(w[order]) / w.sum()
        pitch = int(np.round(vals[order][int(np.searchsorted(cw, 0.5))]))
        # メリスマ: 音節の中に別の高さの安定区間 (SYL_MELISMA_MIN 以上) があれば分ける
        q = np.where(np.isfinite(sm[a:e]) & (np.abs(sm[a:e] - np.round(sm[a:e])) <= 0.4), np.round(sm[a:e]), np.nan)
        segs = []
        j = 0
        while j < len(q):
            if not np.isfinite(q[j]):
                j += 1
                continue
            k = j
            while k < len(q) and q[k] == q[j]:
                k += 1
            if (k - j) * hop >= SYL_MELISMA_MIN:
                segs.append([a + j, a + k, int(q[j])])
            j = k
        merged = []
        for sg in segs:
            if merged and merged[-1][2] == sg[2]:
                merged[-1][1] = sg[1]
            else:
                merged.append(sg)
        if len(merged) >= 2:
            bounds = [cs] + [merged[k][0] * hop for k in range(1, len(merged))] + [end]
            for k, sg in enumerate(merged):
                out.append(Note(float(bounds[k]), float(bounds[k + 1]), sg[2], 1.0, k + 1 < len(merged)))
            prev_pitch = merged[-1][2]
        else:
            out.append(Note(float(cs), float(end), pitch, 1.0, False))
            prev_pitch = pitch
    # 重なりの解消
    out.sort(key=lambda x: x.start)
    for k in range(len(out) - 1):
        if out[k].end > out[k + 1].start:
            out[k] = Note(out[k].start, out[k + 1].start, out[k].midi, out[k].velocity, out[k].legato)
    return [x for x in out if x.end - x.start >= 0.03]


def _cells_to_notes_split(seq, cells, vel, split):
    """A._cells_to_notes と同じだが、split が True のセルでは同じ高さでも音を切り直す (最短 2 セル)."""
    notes = []
    i, n = 0, len(seq)
    while i < n:
        m = seq[i]
        if m is None:
            i += 1
            continue
        j = i + 1
        while j < n and seq[j] == m and not (split[j] and j - i >= 2):
            j += 1
        v = float(np.mean(vel[i:j])) if vel is not None else 1.0
        notes.append(A.Note(cells[i], cells[j], m, v))
        i = j
    return notes


# --------------------------------------------------------------------------- 調とテンポ

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def estimate_key(chroma_mean: np.ndarray) -> tuple[int, str]:
    """Krumhansl プロファイルとの相関で調 (主音, major/minor) を推定."""
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for name, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(np.roll(prof, tonic), chroma_mean)[0, 1]
            if r > best[0]:
                best = (r, tonic, name)
    return best[1], best[2]


def _snap_to_key(seq: list[int | None], tonic: int, mode: str, max_len: int = 2) -> list[int | None]:
    """調に無い短い音 (max_len セル以下) を、隣の調内の音へ半音寄せる."""
    scale = {(tonic + d) % 12 for d in (_MAJOR_SCALE if mode == "major" else _MINOR_SCALE)}
    s = list(seq)
    n = len(s)
    i = 0
    while i < n:
        if s[i] is None:
            i += 1
            continue
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        m = s[i]
        if j - i <= max_len and m % 12 not in scale:
            prev = next((s[k] for k in range(i - 1, -1, -1) if s[k] is not None), None)
            nxt = s[j] if j < n else None
            cand = [c for c in (m - 1, m + 1) if c % 12 in scale]
            if cand:
                ref_note = prev if prev is not None else nxt
                pick = min(cand, key=lambda c: abs(c - ref_note)) if ref_note is not None else cand[0]
                for k in range(i, j):
                    s[k] = pick
        i = j
    return s


def _should_double_tempo(tempo: float, seq: list[int | None]) -> bool:
    """ビート検出が実際の半分のテンポを返した疑いがあるか (遅いテンポ + 32 分単位で音が細かく動く)."""
    if tempo >= 110:
        return False
    notes = [m for m in seq if m is not None]
    if len(notes) < 50:
        return False
    # 音が変わる回数 / 音のあるセル数: 16 分 1 個 (=2 セル) ごとに変わるなら 0.5
    changes = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a is not None and b is not None and a != b)
    rate = changes / len(notes)
    return rate >= 0.28


def _subdivide(cells: np.ndarray, k: int = 2) -> np.ndarray:
    """各セルを k 分割した境界列 (16 分 → 32 分)."""
    out = []
    for a, b in zip(cells[:-1], cells[1:]):
        out.extend(np.linspace(a, b, k, endpoint=False))
    out.append(cells[-1])
    return np.asarray(out)


def _coherent(seq: list[int | None]) -> bool:
    """埋め草に使えるくらい旋律としてまとまっているか (音があり、跳躍が少ない)."""
    notes = [m for m in seq if m is not None]
    if len(notes) < 8 or len(notes) < 0.5 * len(seq):
        return False
    ch = [abs(b - a) for a, b in zip(notes[:-1], notes[1:]) if a != b]
    return not ch or np.mean(np.array(ch) > 7) < 0.15


def _fill_gaps(main: list[int | None], alt: list[int | None], min_gap: int) -> tuple[list[int | None], list[bool]]:
    """main の無音が min_gap セル以上続き、かつ alt がその区間で旋律らしくまとまっている場合だけ alt で埋める.
    戻り値: (埋めた列, 各セルが埋め草かどうか)"""
    out = list(main)
    filled = [False] * len(out)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        if j - i >= min_gap and _coherent(alt[i:j]):
            for k in range(i, j):
                out[k] = alt[k]
                filled[k] = alt[k] is not None
        i = j
    return out, filled


def _lead_from_other(oth: np.ndarray, sr: int, cells: np.ndarray):
    """歌が無いときの主旋律: other ステムを basic-pitch で多声採譜してスカイライン."""
    try:
        from .analysis_ml import active_notes_per_cell, skyline_melody
        active, amp_ref = active_notes_per_cell(oth / (np.abs(oth).max() + 1e-9), sr, cells)
        seq, vel = skyline_melody(active, amp_ref, lo=55, hi=88)
        seq = _drop_register_outliers(seq)
        seq = _smooth_melody(seq)
        return seq, vel
    except Exception as e:  # noqa: BLE001  (basic-pitch が無い場合は pyin にフォールバック)
        log.warning("basic-pitch が使えないので other ステムを pyin で追跡します: %s", e)
        lead_src = A._bandpass(oth, sr, 200, 3000)
        return _track_melody(lead_src, sr, cells, fmin=150.0, fmax=1500.0, min_voiced_ratio=0.5, gate_ratio=0.15)


# --------------------------------------------------------------------------- main

def analyze_sep(y: np.ndarray, sr: int = SR, progress=None, stems: dict[str, np.ndarray] | None = None, *,
                sensitivity: float = 0.5, tempo_mult: str | int = "auto", fill_gaps: bool = False,
                key_snap: bool = True, use_crepe: bool = True, melody_method: str = "hmm",
                quantize: bool = True, source_url: str | None = None, songle_offset: float = 0.0) -> Score:
    """sensitivity: 歌の拾いやすさ 0.3〜0.7. tempo_mult: "auto" | 1 | 2. fill_gaps: 間奏をリード楽器で埋めるか.
    source_url: Songle の拍・歌詞タイミング・コードを使う元の動画. songle_offset: 手元の音声が元の動画より何秒遅れているか."""
    duration = len(y) / sr
    y = y / (np.abs(y).max() + 1e-9)
    LAST_SYLLABLES.clear()
    LAST_SYLLABLE_SPANS.clear()

    if stems is None:
        stems = separate(y, sr, progress=progress)
    voc, drm, bas, oth = stems["vocals"], stems["drums"], stems["bass"], stems["other"]
    if progress:
        progress("分離した各パートを採譜しています…")

    sg_beats = None
    if source_url:
        from . import songle as SG
        sg_beats = SG.beats(source_url, songle_offset)
    if sg_beats and len(sg_beats) > 8:
        # Songle の拍 (人手修正済み) をそのまま 16 分格子にする。メロディの格子と完全に一致させる
        from . import score_model as SM
        cells, cpos = SM.cells_from_beats(sg_beats, duration)
        cells = cells[(cells >= 0) & (cells <= duration + 1e-6)]
        if cells[0] > 1e-3:
            cells = np.insert(cells, 0, 0.0); cpos = np.insert(cpos, 0, (cpos[0] - 1) % 16)
        if cells[-1] < duration:
            cells = np.append(cells, duration)
        bt_ = np.array([b["start"] / 1000.0 for b in sg_beats])
        tempo = float(60.0 / np.median(np.diff(bt_)))
        log.info("Songle の拍を格子に使います (%.1f BPM, %d 拍)", tempo, len(bt_))
    else:
        tempo, cells = A._make_grid(y, sr, duration, onset_src=drm)
    n_cells = len(cells) - 1
    mcells = _subdivide(cells, MEL_DIV)  # メロディ用 32 分音符グリッド

    # ---- メロディ: ボーカル
    hmm_notes = None
    lyric_chars = None
    if source_url:
        from . import songle as SG
        lyric_chars = SG.lyrics_chars(source_url, songle_offset)
        if lyric_chars:
            log.info("Songle/TextAlive の歌詞タイミング %d 文字を音符の区切りに使います", len(lyric_chars))

    def _lyric_melody(mc):
        if sg_beats and len(sg_beats) > 8:
            log.info("楽譜モデル (調・リズム・歌詞の事前 + Cauchy F0 モデル) でメロディを推定します")
            _sgc = SG.chords(source_url, songle_offset) or []
            _parsed = [(c["start"] / 1000.0, (c["start"] + c["duration"]) / 1000.0, SG.parse_chord(c.get("name"))) for c in _sgc]
            return _melody_from_score_model(voc, sr, lyric_chars, sg_beats, mc, key=_key, chords=_parsed, progress=progress)
        return _melody_from_lyrics(voc, sr, lyric_chars, mc, key=_key)

    from . import onset_model as _OM
    use_onsets = not lyric_chars and melody_method == "hmm" and USE_ONSET_MODEL and _OM.is_available()
    if lyric_chars or use_onsets:
        from .harmony import estimate_key as _ek
        _chroma = librosa.feature.chroma_cqt(y=oth + voc, sr=sr, hop_length=HOP, n_chroma=12)
        _key = _ek(_chroma.mean(axis=1))
    if lyric_chars:
        hmm_notes, voc_seq = _lyric_melody(mcells)
        voc_vel = [1.0] * len(voc_seq)
        voc_split = [False] * len(voc_seq)
    elif use_onsets:
        hmm_notes, voc_seq = _melody_from_detected_onsets(voc, sr, cells, mcells, _key, progress=progress)
        voc_vel = [1.0] * len(voc_seq)
        voc_split = [False] * len(voc_seq)
    elif melody_method == "hmm":
        hmm_notes, voc_seq = _track_melody_hmm(voc, sr, mcells, sensitivity, quantize=quantize, progress=progress)
        voc_vel = [1.0] * len(voc_seq)
        voc_split = [False] * len(voc_seq)
    else:
        voc_seq, voc_vel, voc_split = _track_melody(voc, sr, mcells, fmin=80.0, fmax=1400.0, min_voiced_ratio=0.35,
                                                    gate_ratio=0.08, sensitivity=sensitivity, use_crepe=use_crepe)

    # ---- テンポの倍判定 (ビート検出が半分のテンポを返す速い曲向け)
    if sg_beats is None and (tempo_mult == 2 or (tempo_mult == "auto" and _should_double_tempo(tempo, voc_seq))):
        log.info("テンポを倍 (%.0f → %.0f BPM) にします", tempo, tempo * 2)
        tempo *= 2
        cells = mcells                     # 旧 32 分 = 新 16 分
        mcells = _subdivide(cells, MEL_DIV)
        n_cells = len(cells) - 1
        if lyric_chars:
            hmm_notes, voc_seq = _lyric_melody(mcells)
            voc_vel = [1.0] * len(voc_seq)
            voc_split = [False] * len(voc_seq)
        elif use_onsets:
            hmm_notes, voc_seq = _melody_from_detected_onsets(voc, sr, cells, mcells, _key, progress=progress)
            voc_vel = [1.0] * len(voc_seq)
            voc_split = [False] * len(voc_seq)
        elif melody_method == "hmm":
            hmm_notes, voc_seq = _track_melody_hmm(voc, sr, mcells, sensitivity, quantize=quantize, progress=progress)
            voc_vel = [1.0] * len(voc_seq)
            voc_split = [False] * len(voc_seq)
        else:
            voc_seq, voc_vel, voc_split = _track_melody(voc, sr, mcells, fmin=80.0, fmax=1400.0, min_voiced_ratio=0.35,
                                                        gate_ratio=0.08, sensitivity=sensitivity, use_crepe=use_crepe)

    # ---- 調の推定 (コードとメロディの補正に使う)
    chroma_all = librosa.feature.chroma_cqt(y=oth + voc, sr=sr, hop_length=HOP, n_chroma=12)
    tonic, mode = estimate_key(chroma_all.mean(axis=1))
    log.info("調: %s %s", "C C# D D# E F F# G G# A A# B".split()[tonic], mode)
    if key_snap:
        voc_seq = _snap_to_key(voc_seq, tonic, mode)
    vocal_cov = float(np.mean([m is not None for m in voc_seq]))
    # 歌ステムの音量が伴奏に比べてどれくらいあるか (インスト曲の判定に使う)
    voc_level = float(np.percentile(_rms(voc, 2048), 90))
    oth_level = float(np.percentile(_rms(oth, 2048), 90)) + 1e-9
    log.info("ボーカル検出率 %.0f%%, 歌/伴奏 音量比 %.2f", vocal_cov * 100, voc_level / oth_level)

    # ---- リード楽器 (other ステム): 歌が無いところの補填 / インスト曲用
    if vocal_cov < 0.15 or voc_level / oth_level < 0.25:
        log.info("歌がほとんど無いのでリード楽器をメロディに使います")
        lead_seq, lead_vel = _lead_from_other(oth, sr, mcells)
        if key_snap:
            lead_seq = _snap_to_key(lead_seq, tonic, mode)
        mel_seq, mel_vel = lead_seq, lead_vel
        mel_split = [False] * len(mel_seq)
    else:
        mel_split = voc_split
        # 2 小節以上歌が無いところだけ、リードが旋律らしくまとまっていれば控えめに埋める
        gaps = [j - i for i, j in _gap_spans(voc_seq) if j - i >= 2 * MEL_BAR] if fill_gaps else []
        if gaps:
            lead_seq, lead_vel = _lead_from_other(oth, sr, mcells)
            # 埋め草は歌の音域 (中央値 -7〜+12 半音) に収まる音だけ使う
            voc_notes = [m for m in voc_seq if m is not None]
            center = float(np.median(voc_notes)) if voc_notes else 70.0
            lead_seq = [m if m is not None and center - 7 <= m <= center + 12 else None for m in lead_seq]
            if key_snap:
                lead_seq = _snap_to_key(lead_seq, tonic, mode)
            mel_seq, filled = _fill_gaps(voc_seq, lead_seq, min_gap=2 * MEL_BAR)
            mel_vel = [voc_vel[i] if not filled[i] else lead_vel[i] * 0.7 for i in range(len(mel_seq))]
            log.info("間奏の埋め草: %d セル", sum(filled))
        else:
            mel_seq, mel_vel = voc_seq, voc_vel
    if hmm_notes is not None and not (vocal_cov < 0.15 or voc_level / oth_level < 0.25):
        raw_notes = hmm_notes
        if key_snap and not lyric_chars and not use_onsets:  # 楽譜モデルは調の事前を持つので後から寄せない
            snapped = _snap_to_key([m for _, _, m, _, _ in raw_notes], tonic, mode, max_len=1)
            raw_notes = [(s_, e_, m2, v_, l_) for (s_, e_, _, v_, l_), m2 in zip(raw_notes, snapped)]
        melody = [A.Note(s_, e_, m_, v_, l_) for s_, e_, m_, v_, l_ in raw_notes]
        # 歌が 2 小節以上無い区間の埋め草 (従来ロジックの結果を流用)
        if fill_gaps and gaps:
            occupied = np.zeros(len(mcells) - 1, dtype=bool)
            for s_, e_, *_ in raw_notes:
                i0 = max(0, int(np.searchsorted(mcells, s_, side="right") - 1)); i1 = min(len(occupied) - 1, int(np.searchsorted(mcells, e_, side="left") - 1))
                occupied[i0:i1 + 1] = True
            fill_seq = [None if occupied[i] or voc_seq[i] is not None else mel_seq[i] for i in range(len(occupied))]
            melody += A._cells_to_notes(fill_seq, mcells, [v * 0.7 for v in mel_vel])
            melody.sort(key=lambda n: n.start)
        log.info("メロディ: RMVPE + HMM 分割で %d 音", len(melody))
    else:
        melody = _cells_to_notes_split(mel_seq, mcells, mel_vel, mel_split)

    # ---- 伴奏: 小節・強度・コード (Viterbi)・ベース・パターンドラム
    from . import harmony as H, rhythm as R, arrange as AR
    act = R.band_activity(drm, sr, cells)
    mult = R.fix_tempo_octave(tempo, cells, act) if sg_beats is None else 1
    if mult == 2 and tempo_mult == "auto":
        log.info("ドラムの周期からテンポを倍 (%.0f → %.0f BPM) にします", tempo, tempo * 2)
        tempo *= 2
        cells = mcells
        mcells = _subdivide(cells, MEL_DIV)
        n_cells = len(cells) - 1
        act = R.band_activity(drm, sr, cells)
    # 格子の位相補正: キック/スネアが拍の頭 (4 セルごと) に乗るように、格子を 0〜3 セルずらす
    prof = np.array([act["kick"][k::BEAT_CELLS].mean() + 0.5 * act["snare"][k::BEAT_CELLS].mean() for k in range(BEAT_CELLS)])
    off = int(np.argmax(prof))
    if sg_beats is None and off != 0 and prof[off] > 1.2 * prof[0]:
        log.info("格子の位相を %d セルずらします (キックが拍頭に乗るように)", off)
        cells = cells[off:]
        n_cells = len(cells) - 1
        act = {k: v[off:] for k, v in act.items()}
    beat_cells_n = BEAT_CELLS
    n_beats = n_cells // beat_cells_n
    beat_times = cells[::beat_cells_n][: n_beats + 1]
    phase = R.estimate_downbeat_phase(act)
    if sg_beats is not None:
        # Songle の小節頭 (position == 1) を使う
        bar_beats = [i for i in range(n_beats) if 4 * i < len(cpos) and cpos[4 * i] == 0]
        if bar_beats:
            phase = int(bar_beats[0] % 4)
    bar_starts = [(phase + 4 * k) * beat_cells_n for k in range(-1, n_beats // 4 + 1)]
    bar_starts = [b for b in bar_starts if 0 <= b < n_cells]
    if not bar_starts or bar_starts[0] > 0:
        bar_starts = [0] + bar_starts if (not bar_starts or bar_starts[0] >= beat_cells_n) else bar_starts
    beat_in_bar = np.array([((i - phase) % 4) for i in range(n_beats)])
    # 拍ごとの chroma (伴奏 = other + bass) と、ベース音のピッチクラス
    acc = oth + bas
    chroma = librosa.feature.chroma_cqt(y=acc, sr=sr, hop_length=HOP, n_chroma=12)
    fb = np.clip(A._frame_of(beat_times, sr), 0, chroma.shape[1])
    beat_chroma = np.zeros((n_beats, 12))
    acc_rms = _rms(acc, 2048); acc_gate = np.percentile(acc_rms, 95) * 0.05
    has_energy = np.zeros(n_beats, bool)
    f0_b, v_b, _ = librosa.pyin(bas, fmin=30.0, fmax=260.0, sr=sr, frame_length=4096, hop_length=HOP, fill_na=np.nan)
    bass_pc: list[int | None] = []
    det_bass: list[int | None] = []
    for i in range(n_beats):
        a, b = fb[i], max(fb[i + 1], fb[i] + 1)
        beat_chroma[i] = np.median(chroma[:, a:b], axis=1) if b > a else 0
        seg = acc_rms[a:min(b, len(acc_rms))]
        has_energy[i] = seg.size > 0 and seg.max() >= acc_gate
        fb_seg = f0_b[a:min(b, len(f0_b))]; vb_seg = v_b[a:min(b, len(v_b))]
        if fb_seg.size and vb_seg.mean() >= 0.3:
            m = int(np.round(np.median(librosa.hz_to_midi(fb_seg[vb_seg]))))
            det_bass.append(m); bass_pc.append(m % 12)
        else:
            det_bass.append(None); bass_pc.append(None)
    chroma_all = librosa.feature.chroma_cqt(y=oth + voc, sr=sr, hop_length=HOP, n_chroma=12)
    tonic, mode = H.estimate_key(chroma_all.mean(axis=1))
    # メロディは最も確かな和声の手掛かりなので、拍ごとのメロディ音 (長さ重み) を chroma に加える
    mel_hist = np.zeros((n_beats, 12))
    for nt in melody:
        i0 = max(0, int(np.searchsorted(beat_times, nt.start, side="right") - 1)); i1 = min(n_beats - 1, int(np.searchsorted(beat_times, nt.end, side="left") - 1))
        for i in range(i0, i1 + 1):
            ov = min(nt.end, beat_times[i + 1]) - max(nt.start, beat_times[i])
            if ov > 0:
                mel_hist[i, nt.midi % 12] += ov
    for i in range(n_beats):
        if mel_hist[i].sum() > 0 and beat_chroma[i].max() > 0:
            beat_chroma[i] = beat_chroma[i] / beat_chroma[i].max() + 0.6 * mel_hist[i] / mel_hist[i].sum()
    chords_by_beat = H.estimate_chords(beat_chroma, bass_pc, tonic, mode, beat_in_bar=beat_in_bar, min_energy=has_energy)
    if source_url:
        from . import songle as SG
        sg_chords = SG.chords(source_url, songle_offset)
        if sg_chords:
            parsed = [(c["start"] / 1000.0, (c["start"] + c["duration"]) / 1000.0, SG.parse_chord(c.get("name"))) for c in sg_chords]
            used = 0
            for i in range(n_beats):
                mid = 0.5 * (beat_times[i] + beat_times[i + 1])
                for cs, ce, pc in parsed:
                    if cs <= mid < ce:
                        if pc is not None:
                            chords_by_beat[i] = (H.NAMES[pc[0]] + ("m" if pc[1] == "min" else ""), pc[0], pc[1], list(pc[2])); used += 1
                        break
            log.info("Songle のコード進行を %d/%d 拍に適用", used, n_beats)
    chords = [(float(beat_times[i]), float(beat_times[i + 1]), (list(ch[3]) if ch else [])) for i, ch in enumerate(chords_by_beat)]
    n_changes = sum(1 for a_, b_ in zip(chords_by_beat[:-1], chords_by_beat[1:]) if (a_ and a_[0]) != (b_ and b_[0]))
    log.info("コード: %s %s, %d 拍, 変化 %.2f 回/小節", H.NAMES[tonic], mode, n_beats, n_changes / max(1, n_beats / 4))

    # ---- 強度と編曲
    intensity = AR.bar_intensity(y, voc, sr, cells, bar_starts)
    inten_beat = np.zeros(n_beats)
    for bi, c0 in enumerate(bar_starts):
        for k in range(4):
            bb = c0 // beat_cells_n + k
            if 0 <= bb < n_beats:
                inten_beat[bb] = intensity[bi]
    bass = AR.bass_line(chords_by_beat, beat_times, cells, det_bass, inten_beat, tempo)
    # ---- ドラム: パターン化
    pats = R.choose_patterns(act, bar_starts, intensity)
    drums = R.render_hits(pats, bar_starts, cells, act, intensity)
    log.info("ドラムパターン: %s", {k: pats.count(k) for k in set(pats)})

    # 歌の音高と音量の動き: メロディを「歌に沿わせて」鳴らすのに使う (synth.render_follow_melody)
    contour = None
    if hmm_notes is not None and not (vocal_cov < 0.15 or voc_level / oth_level < 0.25):
        _, c_f0, c_rms, _ = vocal_f0(voc, sr)
        c_midi = np.where(c_f0 > 0, librosa.hz_to_midi(np.maximum(c_f0, 1e-3)), np.nan)
        c_lv = 20 * np.log10(c_rms / (np.percentile(c_rms, 95) + 1e-9) + 1e-6)
        if MELODY_METHOD == "syllable" and LAST_SYLLABLE_SPANS:
            # 音節ごとに 1 音 (歌詞の文字, 無ければ検出した音節)
            melody = syllable_notes(LAST_SYLLABLE_SPANS, c_midi, c_f0 > 0, c_lv)
            log.info("メロディ: 音節ごとに 1 音で %d 音 (音節 %d)", len(melody), len(LAST_SYLLABLE_SPANS))
        elif INSERT_SUNG_SEGMENTS:
            melody = insert_sung_segments(melody, c_midi)
        c_ref = np.percentile(c_rms, 95) + 1e-9
        contour = dict(hop=0.01, midi=np.where(c_f0 > 0, librosa.hz_to_midi(np.maximum(c_f0, 1e-3)), np.nan).astype(np.float32),
                       level_db=(20 * np.log10(c_rms / c_ref + 1e-6)).astype(np.float32),
                       syllables=np.asarray(LAST_SYLLABLES, dtype=np.float64))

    return Score(duration=duration, tempo=tempo, cells=cells,
                 melody=melody, bass=bass, chords=chords, drums=drums,
                 bar_starts=list(map(int, bar_starts)), intensity=[float(v) for v in intensity],
                 arp_octave=AR.arpeggio_octave(melody), key=(int(tonic), mode),
                 verify=dict(LAST_VERIFY) if LAST_VERIFY else None, contour=contour)


def _gap_spans(seq: list[int | None]):
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] is not None:
            i += 1
            continue
        j = i
        while j < n and seq[j] is None:
            j += 1
        yield i, j
        i = j
