"""和声: 調の推定と、拍ごとのコード推定 (Viterbi)。

コードは調内の三和音 (メジャー/マイナー) を中心に 24 種 + 無音を候補にし、
  観測 = 伴奏 (other + bass ステム) の chroma とテンプレートの類似度
  遷移 = コードを変えるとペナルティ (拍の頭ほど変わりやすい)
  事前 = 調に含まれる (ダイアトニック) コードを優遇、ベース音が根音ならボーナス
で最尤の進行を求める。
"""
from __future__ import annotations

import numpy as np

NAMES = "C C# D D# E F F# G G# A A# B".split()
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]


def estimate_key(chroma_mean: np.ndarray) -> tuple[int, str]:
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for name, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(np.roll(prof, tonic), chroma_mean)[0, 1]
            if r > best[0]:
                best = (r, tonic, name)
    return best[1], best[2]


def scale_pcs(tonic: int, mode: str) -> set[int]:
    return {(tonic + d) % 12 for d in (MAJOR_SCALE if mode == "major" else MINOR_SCALE)}


def chord_templates():
    """(名前, 根音, 種類, 構成音) のリスト. 種類: maj / min."""
    out = []
    for root in range(12):
        out.append((f"{NAMES[root]}", root, "maj", [root, (root + 4) % 12, (root + 7) % 12]))
        out.append((f"{NAMES[root]}m", root, "min", [root, (root + 3) % 12, (root + 7) % 12]))
    return out


def diatonic_chords(tonic: int, mode: str) -> set[int]:
    """調内の三和音のインデックス (chord_templates の順)."""
    sc = scale_pcs(tonic, mode)
    idx = set()
    for k, (_, root, kind, pcs) in enumerate(chord_templates()):
        if all(pc in sc for pc in pcs):
            idx.add(k)
    return idx


def estimate_chords(beat_chroma: np.ndarray, beat_bass_pc: list[int | None], tonic: int, mode: str,
                    beat_in_bar: np.ndarray | None = None, change_penalty: float = 1.2,
                    min_energy: np.ndarray | None = None):
    """拍ごとの chroma (n_beats, 12) からコード列を求める. 戻り値: 各拍の (名前, 根音, 種類, 構成音) または None."""
    temps = chord_templates()
    T = np.zeros((len(temps), 12))
    for k, (_, root, kind, pcs) in enumerate(temps):
        T[k, pcs] = 1.0
        T[k, root] += 0.5  # 根音を少し重く
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    n = len(beat_chroma)
    K = len(temps) + 1  # 最後 = 無音 (N)
    dia = diatonic_chords(tonic, mode)
    logB = np.full((n, K), -8.0)
    for i in range(n):
        c = beat_chroma[i]
        if min_energy is not None and not min_energy[i]:
            logB[i, K - 1] = 0.0
            continue
        cn = c / (np.linalg.norm(c) + 1e-9)
        sim = T @ cn  # 0..1
        for k in range(len(temps)):
            score = 4.0 * sim[k]
            if k in dia:
                score += 0.8
            bpc = beat_bass_pc[i]
            if bpc is not None:
                if bpc == temps[k][1]:
                    score += 1.0
                elif bpc in temps[k][3]:
                    score += 0.3
            logB[i, k] = score
        logB[i, K - 1] = 1.5  # 無音候補は弱め
    # 遷移: 同じコードは 0、変えるとペナルティ (小節頭ならペナルティを軽く)
    delta = logB[0].copy()
    back = np.zeros((n, K), dtype=np.int32)
    for i in range(1, n):
        pen = change_penalty * (0.4 if (beat_in_bar is not None and beat_in_bar[i] == 0) else (0.8 if (beat_in_bar is not None and beat_in_bar[i] == 2) else 1.0))
        stay = delta
        switch = delta.max() - pen
        cand = np.where(stay >= switch, stay, switch)
        back[i] = np.where(stay >= switch, np.arange(K), int(np.argmax(delta)))
        delta = cand + logB[i]
    st = np.zeros(n, dtype=np.int32)
    st[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        st[i - 1] = back[i, st[i]]
    return [None if k == K - 1 else temps[k] for k in st]


def melody_clash_ratio(melody, chords_by_beat, beat_times):
    """メロディの音が同時に鳴るコード音と半音でぶつかっている時間の割合 (評価用)."""
    clash = tot = 0.0
    for n in melody:
        i0 = max(0, int(np.searchsorted(beat_times, n.start, side="right") - 1))
        i1 = min(len(chords_by_beat) - 1, int(np.searchsorted(beat_times, n.end, side="left") - 1))
        for i in range(i0, i1 + 1):
            ch = chords_by_beat[i]
            if ch is None:
                continue
            ov = min(n.end, beat_times[i + 1]) - max(n.start, beat_times[i])
            if ov <= 0:
                continue
            tot += ov
            pc = n.midi % 12
            if pc not in ch[3] and any((abs(pc - q) % 12) in (1, 11) for q in ch[3]):
                clash += ov
    return clash / tot if tot else 0.0
