"""f0 トラック → 音符列 (HMM / Viterbi による音符分割).

Tony (Mauch et al.) / Ryynänen の方式に倣い、状態 = 各半音 + 無声、観測 = f0 (半音) と有声/無声とし、
「音高を変える」遷移に罰則を掛ける。ビブラートやしゃくりでは音が切れず、本当に音が動いたときだけ切れる。
格子に丸めないので、速い歌でも音が潰れない。
"""
from __future__ import annotations

import numpy as np
import librosa

from .analysis import Note

HOP_S = 0.01


def hmm_segment(t: np.ndarray, f0: np.ndarray, *, lo: int = 36, hi: int = 96, stay: float = 0.985,
                sigma: float = 0.8, min_dur: float = 0.08, unv_stay: float = 0.97,
                energy: np.ndarray | None = None) -> list[tuple[float, float, int, float]]:
    """(start, end, midi, velocity) のリストを返す. f0 は 10 ms 間隔、無声 = 0."""
    m = np.where(f0 > 0, librosa.hz_to_midi(np.maximum(f0, 1e-3)), np.nan)
    pitches = np.arange(lo, hi + 1)
    P = len(pitches)
    U = P
    T = len(m)
    if T == 0:
        return []
    voiced = ~np.isnan(m)
    logB = np.full((T, P + 1), -50.0)
    d = (m[voiced, None] - pitches[None, :]) / sigma
    logB[voiced, :P] = -0.5 * d ** 2 - np.log(sigma)
    logB[voiced, U] = -6.0
    logB[~voiced, :P] = -6.0
    logB[~voiced, U] = 0.0

    logA = np.empty((P + 1, P + 1))
    idx = np.arange(P)
    dist = np.abs(idx[:, None] - idx[None, :])
    logA[:P, :P] = np.log((1 - stay) * 0.5) - 0.15 * dist
    logA[idx, idx] = np.log(stay)
    logA[:P, U] = np.log((1 - stay) * 0.5)
    logA[U, U] = np.log(unv_stay)
    logA[U, :P] = np.log((1 - unv_stay) / P)

    delta = logB[0] + np.log(1.0 / (P + 1))
    back = np.zeros((T, P + 1), dtype=np.int32)
    for k in range(1, T):
        cand = delta[:, None] + logA
        back[k] = np.argmax(cand, axis=0)
        delta = cand[back[k], np.arange(P + 1)] + logB[k]
    st = np.zeros(T, dtype=np.int32)
    st[-1] = int(np.argmax(delta))
    for k in range(T - 1, 0, -1):
        st[k - 1] = back[k, st[k]]

    notes: list[tuple[float, float, int, float]] = []
    i = 0
    while i < T:
        if st[i] == U:
            i += 1
            continue
        j = i
        while j < T and st[j] == st[i]:
            j += 1
        if (j - i) * HOP_S >= min_dur:
            vel = float(np.clip(energy[i:j].max(), 0.0, 1.0)) if energy is not None and energy[i:j].size else 1.0
            notes.append((float(t[i]), float(t[j - 1] + HOP_S), int(pitches[st[i]]), vel))
        i = j
    return notes


def snap_onsets(notes, grid: np.ndarray, tol: float = 0.03):
    """音の始まり/終わりを近い格子 (32 分) に寄せる。tol 秒より離れていればそのまま (リズム感だけ整える)."""
    out = []
    for s, e, m, v in notes:
        k = int(np.argmin(np.abs(grid - s)))
        if abs(grid[k] - s) <= tol:
            s = float(grid[k])
        k = int(np.argmin(np.abs(grid - e)))
        if abs(grid[k] - e) <= tol and grid[k] > s:
            e = float(grid[k])
        if e - s >= 0.03:
            out.append((s, e, m, v))
    # 重なりを解消
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][1]:
            out[i - 1] = (out[i - 1][0], out[i][0], out[i - 1][2], out[i - 1][3])
    return [n for n in out if n[1] - n[0] >= 0.03]


def to_notes(notes) -> list[Note]:
    return [Note(s, e, m, v) for s, e, m, v in notes]


def seq_from_notes(notes, cells: np.ndarray) -> list[int | None]:
    """セル列表現 (既存の調補正・埋め草ロジック用) に変換: 各セルで一番長く鳴っている音."""
    n = len(cells) - 1
    seq: list[int | None] = [None] * n
    for s, e, m, _ in notes:
        i0 = max(0, int(np.searchsorted(cells, s, side="right") - 1))
        i1 = min(n - 1, int(np.searchsorted(cells, e, side="left") - 1))
        for i in range(i0, i1 + 1):
            ov = min(e, cells[i + 1]) - max(s, cells[i])
            if ov >= 0.5 * (cells[i + 1] - cells[i]) or seq[i] is None:
                seq[i] = m
    return seq


def drop_register_outliers(notes, window: float = 6.0, max_dev: float = 9.0, max_len: float = 0.5):
    """局所的な音域 (前後 window 秒の音符の、長さで重み付けした中央値) から max_dev 半音以上外れた
    max_len 秒未満の音を捨てる。歌ステムに混ざったハモリ・伴奏・歓声などの短い割り込みを除く."""
    if len(notes) < 5:
        return notes
    starts = np.array([n[0] for n in notes]); ends = np.array([n[1] for n in notes])
    mids = np.array([n[2] for n in notes], dtype=float); durs = ends - starts
    outlier = np.zeros(len(notes), bool)
    for i, (s, e, m, v) in enumerate(notes):
        sel = (starts < e + window) & (ends > s - window)
        sel[i] = False
        if sel.sum() < 4:
            continue
        # 長さ重み付き中央値
        order = np.argsort(mids[sel]); w = durs[sel][order]; c = np.cumsum(w) / w.sum()
        med = mids[sel][order][int(np.searchsorted(c, 0.5))]
        outlier[i] = durs[i] < max_len and abs(m - med) > max_dev
    # 外れ音が連続して 0.4 秒以上続くなら、それは本物のフレーズ (低い/高い節回し) なので残す
    keep = []
    i = 0
    while i < len(notes):
        if not outlier[i]:
            keep.append(notes[i]); i += 1; continue
        j = i
        while j < len(notes) and outlier[j] and (j == i or notes[j][0] - notes[j - 1][1] < 0.15):
            j += 1
        run_dur = notes[j - 1][1] - notes[i][0]
        if run_dur >= 0.4 or (j - i) >= 3:
            keep.extend(notes[i:j])
        i = j
    return keep


def legato_fill(notes, max_gap: float = 0.15, tail: float = 0.12, breath: float = 0.4):
    """ぶつ切り対策: 音と音の短い隙間を埋める.
       - 次の音まで max_gap 秒未満 → 前の音を次の音の直前まで伸ばす (レガート。シンセ側で滑らかにつながる)
       - max_gap〜breath 秒 → 前の音を tail 秒だけ伸ばす (ブレスは残す)
    """
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: n[0])
    out = []
    for i, (s, e, m, v) in enumerate(notes):
        if i + 1 < len(notes):
            ns = notes[i + 1][0]
            gap = ns - e
            if 0 < gap < max_gap:
                e = ns - 0.005
            elif max_gap <= gap < breath:
                e = min(ns - 0.02, e + tail)
        out.append((s, e, m, v))
    return out


def musicalize(notes, grid: np.ndarray, *, beat_cells: int = 8, min_cells: int = 1):
    """楽曲として滑らかに聞こえるように音符列を整える (grid = 32 分音符の境界時刻).
       1. 始まりを最寄りの格子に量子化、長さは格子の整数倍 (最短 min_cells)
       2. 同じ格子位置に重なった音は長い方を残す
       3. 1 セルだけの揺れ (前後どちらかと 2 半音以内で、前後が同じ音) は隣に吸収
       4. 局所音域から 1 オクターブ以上外れた音は、オクターブ移動で音域に入るなら移す
       5. 2 セル未満の隙間 (子音) だけ前の音を伸ばす
       6. 音量は軽く均して 0.6〜1.0、拍頭と休符明けにアクセント
    """
    if not notes:
        return notes
    g = np.asarray(grid)
    q = []
    for s, e, m, v in sorted(notes, key=lambda n: n[0]):
        i = int(np.argmin(np.abs(g - s)))
        j = int(np.argmin(np.abs(g - e)))
        j = max(j, i + min_cells)
        if j >= len(g):
            continue
        q.append([i, j, int(m), float(v)])
    # 2. 重なり解消: 同じ開始セルは長い方、前の音が次の開始を跨いだら切る
    q.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    dedup = []
    for n in q:
        if dedup and dedup[-1][0] == n[0]:
            continue
        if dedup and dedup[-1][1] > n[0]:
            dedup[-1][1] = n[0]
        if n[1] > n[0]:
            dedup.append(n)
    q = [n for n in dedup if n[1] > n[0]]
    # 3. 1 セルの揺れを吸収
    out = []
    for k, n in enumerate(q):
        if n[1] - n[0] == 1 and out and k + 1 < len(q):
            prev, nxt = out[-1], q[k + 1]
            if prev[2] == nxt[2] and abs(n[2] - prev[2]) <= 2 and prev[1] == n[0]:
                prev[1] = n[1]
                continue
        out.append(n)
    q = out
    # 4. 音域外の音はオクターブ移動
    if len(q) >= 5:
        st = np.array([n[0] for n in q]); en = np.array([n[1] for n in q]); mid = np.array([n[2] for n in q], float); dur = en - st
        win = 8 * beat_cells
        for k, n in enumerate(q):
            sel = (st < en[k] + win) & (en > st[k] - win); sel[k] = False
            if sel.sum() < 4:
                continue
            order = np.argsort(mid[sel]); w = dur[sel][order]; c = np.cumsum(w) / w.sum(); med = mid[sel][order][int(np.searchsorted(c, 0.5))]
            if abs(n[2] - med) >= 12:
                prev_far = k > 0 and abs(q[k - 1][2] - med) >= 9 and q[k - 1][1] >= n[0] - 2
                next_far = k + 1 < len(q) and abs(q[k + 1][2] - med) >= 9 and q[k + 1][0] <= n[1] + 2
                if prev_far or next_far:
                    continue  # 連続する外れ音は本物のフレーズ
                cand = n[2] + 12 * int(np.round((med - n[2]) / 12))
                if abs(cand - med) <= 6:
                    n[2] = int(cand)
    # 5. 子音などによる短い隙間 (2 セル未満) だけ前の音を伸ばして埋める。それ以上は休符として残す
    for k in range(len(q) - 1):
        gap = q[k + 1][0] - q[k][1]
        if 0 < gap < 2:
            q[k][1] = q[k + 1][0]
    # 6. 音量: 抑揚は残しつつ極端なばらつきだけ均し、拍頭の音と休符明けの音にアクセント
    vel = np.array([n[3] for n in q])
    if len(vel) >= 3:
        sm = np.convolve(np.pad(vel, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        vel = 0.7 * vel + 0.3 * sm
    vel = np.clip(vel, 0.6, 1.0)
    for k, n in enumerate(q):
        on_beat = (n[0] % beat_cells) == 0
        after_rest = k == 0 or q[k - 1][1] < n[0]
        if on_beat or after_rest:
            vel[k] = min(1.0, vel[k] + 0.1)
    return [(float(g[n[0]]), float(g[n[1]]), n[2], float(vv)) for n, vv in zip(q, vel)]


def phrases_from_voicing(t: np.ndarray, voiced: np.ndarray, max_gap: float = 0.10, min_len: float = 0.15):
    """元の歌の有声区間から「フレーズ」を作る (max_gap 未満の途切れは同じフレーズ)."""
    out = []
    n = len(voiced)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        last = i
        while j < n:
            if voiced[j]:
                last = j
            elif (j - last) * HOP_S >= max_gap:
                break
            j += 1
        s_, e_ = float(t[i]), float(t[last] + HOP_S)
        if e_ - s_ >= min_len:
            out.append((s_, e_))
        i = j
    return out


def connect_by_phrases(notes, phrases, grid: np.ndarray, t=None, f0=None, min_gap_fill: float = 0.06):
    """フレーズ内の隣接音は無音を挟まずにつなぎ (legato)、フレーズ末尾の音はフレーズ終端まで伸ばす.
    フレーズをまたぐ隙間は休符として残す. 隙間が長く、その間の f0 が別の音程なら、その音程の音を補って埋める
    (HMM が落とした短い音の復元). 戻り値: (start, end, midi, vel, legato)"""
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n[0])
    # 隙間の f0 から音を補う
    if t is not None and f0 is not None and len(notes) > 1:
        filled = []
        for i, n in enumerate(notes):
            filled.append(list(n))
            if i + 1 < len(notes):
                gs, ge = n[1], notes[i + 1][0]
                if ge - gs >= min_gap_fill:
                    sel = (t >= gs) & (t < ge) & (f0 > 0)
                    if sel.sum() >= 0.5 * max(1, int((ge - gs) / HOP_S)):
                        m_gap = int(np.round(np.median(librosa.hz_to_midi(f0[sel]))))
                        if m_gap != n[2]:
                            k0 = int(np.argmin(np.abs(grid - gs))); k1 = int(np.argmin(np.abs(grid - ge)))
                            if k1 > k0:
                                filled.append([float(grid[k0]), float(grid[k1]), m_gap, float(n[3])])
        notes = [tuple(x) for x in sorted(filled, key=lambda n: n[0])]
    ph = np.array(phrases) if phrases else np.zeros((0, 2))
    def phrase_of(t_):
        if len(ph) == 0:
            return -1
        k = np.where((ph[:, 0] - 0.05 <= t_) & (t_ < ph[:, 1] + 0.05))[0]
        return int(k[0]) if len(k) else -1
    out = []
    for i, (s_, e_, m, v) in enumerate(notes):
        pi = phrase_of(s_)
        leg = False
        if i + 1 < len(notes):
            ns = notes[i + 1][0]
            pj = phrase_of(ns)
            if pi >= 0 and pi == pj:
                e_ = ns          # 次の音まで伸ばして接続
                leg = True
            elif pi >= 0:
                # フレーズ末尾: フレーズ終端 (格子に丸め) まで伸ばす
                pe = ph[pi, 1]
                k = int(np.argmin(np.abs(grid - pe)))
                e_ = max(e_, min(float(grid[k]), ns - 0.02))
        elif pi >= 0:
            pe = ph[pi, 1]
            k = int(np.argmin(np.abs(grid - pe)))
            e_ = max(e_, float(grid[k]))
        if e_ > s_:
            out.append((s_, e_, m, v, leg))
    return out


def fill_phrase_holes(notes, phrases, grid: np.ndarray, t, f0, min_hole: float = 0.06, new_pitches: bool = True):
    """フレーズの中で音符に覆われていない穴を埋める最終パス.
       穴の f0 が十分にあればその中央値の音を入れ、無ければ隣の音を伸ばす (フレーズ先頭は最初の音を前に伸ばす)."""
    if not notes or not phrases:
        return notes
    notes = [list(n) for n in sorted(notes, key=lambda n: n[0])]
    g = np.asarray(grid)
    def q(x):
        return float(g[int(np.argmin(np.abs(g - x)))])
    added = []
    for a, b in phrases:
        a, b = q(a), q(b)
        inside = [n for n in notes if n[1] > a and n[0] < b]
        # 穴を列挙
        holes = []
        cur = a
        for n in sorted(inside, key=lambda n: n[0]):
            if n[0] - cur >= min_hole:
                holes.append((cur, n[0], n))
            cur = max(cur, n[1])
        if b - cur >= min_hole:
            holes.append((cur, b, None))
        prev = None
        for hs, he, nxt in holes:
            sel = (t >= hs) & (t < he) & (f0 > 0)
            frames = max(1, int((he - hs) / HOP_S))
            m_gap = int(np.round(np.median(librosa.hz_to_midi(f0[sel])))) if (new_pitches and sel.sum() >= 0.5 * frames) else None
            prev = max([n for n in inside if n[1] <= hs + 1e-6], key=lambda n: n[1], default=None)
            if m_gap is not None and (prev is None or m_gap != prev[2]) and (nxt is None or m_gap != nxt[2]):
                vel = prev[3] if prev is not None else (nxt[3] if nxt is not None else 0.8)
                added.append([hs, he, m_gap, vel, nxt is not None])
                if prev is not None:
                    prev[4] = True
            elif prev is not None:
                prev[1] = he
                prev[4] = nxt is not None
            elif nxt is not None:
                nxt[0] = hs
    out = sorted(notes + added, key=lambda n: n[0])
    # 重なりの解消
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][1]:
            out[i - 1][1] = out[i][0]
    return [tuple(n) for n in out if n[1] - n[0] >= 0.03]


def split_at_onsets(notes, onsets: np.ndarray, grid: np.ndarray, min_part: float = 0.08):
    """歌のシラブルの立ち上がり (onsets) で音を切り直す。同じ高さで歌い直している箇所を別の音にする
    (つながりすぎ対策)。切った前半は legato=True (無音を挟まず音量のくぼみだけで区切る)."""
    if not notes or len(onsets) == 0:
        return notes
    g = np.asarray(grid)
    out = []
    for n in sorted(notes, key=lambda x: x[0]):
        s_, e_, m, v, leg = n
        cuts = [float(g[int(np.argmin(np.abs(g - o)))]) for o in onsets if s_ + min_part <= o <= e_ - min_part]
        cuts = sorted({c for c in cuts if s_ + min_part <= c <= e_ - min_part})
        cur = s_
        for c in cuts:
            if c - cur >= min_part:
                out.append((cur, c, m, v, True))
                cur = c
        out.append((cur, e_, m, v, leg))
    return out


def syllable_valleys(y: np.ndarray, sr: int, prominence_db: float = 4.0, min_width_ms: float = 30.0) -> np.ndarray:
    """歌ステムの音量の谷 (シラブルの切れ目) の時刻。
    スペクトル変化のピーク (オンセット) はビブラートや子音ノイズでも反応し過分割を招いたので、
    深さ prominence_db 以上・幅 min_width_ms 以上の音量の谷だけを境界の手掛かりにする."""
    from scipy.signal import find_peaks
    hop = int(round(HOP_S * sr))
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0] + 1e-6)
    pk, _ = find_peaks(-db, prominence=prominence_db, width=max(1, min_width_ms / (HOP_S * 1000)), rel_height=0.5)
    return pk * HOP_S


def merge_passing_notes(notes, max_len: float = 0.09):
    """短い経過音 (max_len 未満で、前後どちらとも違う高さ、次の音と隙間なし) を次の音に吸収する
    (しゃくりの途中が別の音として出るのを防ぐ)."""
    notes = [tuple(n) for n in sorted(notes, key=lambda n: n[0])]
    out = []
    i = 0
    while i < len(notes):
        n = notes[i]
        if (n[1] - n[0] < max_len and i + 1 < len(notes) and notes[i + 1][0] - n[1] < 0.02
                and out and notes[i + 1][2] != n[2] and out[-1][2] != n[2]):
            nx = notes[i + 1]
            notes[i + 1] = (n[0], nx[1], nx[2], nx[3]) + tuple(nx[4:])
            i += 1
            continue
        out.append(n)
        i += 1
    return out
