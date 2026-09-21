"""楽譜モデル + F0 モデルによる音符推定 (Nishikimi, Nakamura, Goto ら ISMIR 2017 の考え方を Viterbi で実装).

「歌の f0 を規則で刻む」のではなく、
  楽譜モデル: 調 (音階) に沿った音高の遷移、拍節格子 (タタム) 上の音の始まり (リズム)、音価の分布
  F0 モデル : 音符列から歌の f0 が生成される過程 (音の頭の移行区間、ビブラート等のずれを Cauchy 分布で吸収)
を合わせた確率モデルの下で、最も確からしい音符列を動的計画法 (半マルコフ Viterbi) で求める。
歌詞の文字タイミング (TextAlive/Songle) があれば、音の始まりの事前確率として組み込む。
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01
TRANS_CELLS = 3   # 移行区間が及び得る最大の格子数
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def make_lattice(cells16: np.ndarray, div: int = 2) -> np.ndarray:
    """16 分音符の境界時刻から、音の始まり/終わりを置ける格子 (既定は 32 分) を作る."""
    out = []
    for a, b in zip(cells16[:-1], cells16[1:]):
        out.extend(np.linspace(a, b, div, endpoint=False))
    out.append(cells16[-1])
    return np.asarray(out, dtype=float)


def cells_from_beats(beats: list[dict], duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Songle の拍 (start ms, position 1..4) から 16 分セル境界と各セルの小節内位置 (0..15) を作る."""
    bt = np.array([b["start"] / 1000.0 for b in beats]); pos = np.array([b["position"] for b in beats])
    d0 = bt[1] - bt[0]
    k = 1
    while bt[0] - k * d0 > 0:
        k += 1
    pre = [bt[0] - j * d0 for j in range(k - 1, 0, -1)]
    pre_pos = [(pos[0] - j - 1) % 4 + 1 for j in range(k - 1, 0, -1)]
    allb = np.concatenate([pre, bt]); allp = np.concatenate([pre_pos, pos])
    cells, cpos = [], []
    for i in range(len(allb) - 1):
        for q in range(4):
            cells.append(allb[i] + (allb[i + 1] - allb[i]) * q / 4); cpos.append(((allp[i] - 1) % 4) * 4 + q)
    d = allb[-1] - allb[-2]
    tcur = allb[-1]; p = allp[-1]
    while tcur < duration:
        for q in range(4):
            cells.append(tcur + d * q / 4); cpos.append(((p - 1) % 4) * 4 + q)
        tcur += d; p = p % 4 + 1
    cells.append(tcur)
    return np.array(cells), np.array(cpos)


def warp_lattice(lattice: np.ndarray, onsets, max_shift: float | None = None) -> np.ndarray:
    """既知の音の始まり (歌詞の文字境界など) に最も近い格子点をその時刻に寄せる (既定: 半セルまで).
    音価は格子単位のまま (整数タタム) で、時刻だけ実際の発声に合わせる (論文の onset 時間ずれのモデルに相当)."""
    lat = np.array(lattice, dtype=float)
    for o in onsets:
        i = int(np.argmin(np.abs(lat - o)))
        lim = max_shift if max_shift is not None else 0.5 * float(np.median(np.diff(lattice)))
        if abs(lat[i] - o) <= lim:
            lo_ = lat[i - 1] + 0.01 if i > 0 else -1e9
            hi_ = lat[i + 1] - 0.01 if i + 1 < len(lat) else 1e9
            lat[i] = min(max(o, lo_), hi_)
    return lat


def _interval_logprior(K: int, scale_pcs: set[int], lo: int, in_key_w: float = 10.0, out_key_w: float = 1.0,
                       step_decay: float = 0.35, same_pitch_w: float = 1.0) -> np.ndarray:
    """音高遷移の対数事前確率 [K, K] (前の音 → 次の音)。調内の音は 10:1 で優遇し、音程は小さいほど確からしい."""
    pcs = np.array([(lo + i) % 12 for i in range(K)])
    key_w = np.where(np.isin(pcs, list(scale_pcs)), in_key_w, out_key_w)
    idx = np.arange(K)
    d = np.abs(idx[:, None] - idx[None, :])
    w = np.exp(-step_decay * d) * key_w[None, :]
    w[idx, idx] *= same_pitch_w
    w = w / w.sum(axis=1, keepdims=True)
    return np.log(w + 1e-12)


def _duration_logprior(L: int, div: int) -> np.ndarray:
    """音価 (格子単位) の対数事前確率。16 分の整数倍を優遇、32 分の端数は少し不利、長い音は緩やかに減衰."""
    lp = np.zeros(L + 1)
    for l in range(1, L + 1):
        tat = l / div
        base = -0.12 * tat  # 長さに対する緩い減衰
        if abs(tat - round(tat)) > 1e-6:
            base -= 1.0  # 32 分の端数
        if round(tat) in (1, 2, 4, 8):
            base += 0.4  # 16 分・8 分・4 分・2 分
        lp[l] = base
    lp[1:] -= np.log(np.exp(lp[1:]).sum())  # 正規化 (対数確率 ≤ 0)。正の値だと無音中に音を立てる動機になる
    lp[0] = -1e9
    return lp


def estimate_notes(t: np.ndarray, f0_midi: np.ndarray, voiced: np.ndarray, lattice: np.ndarray, *,
                   tonic: int, mode: str, lo: int = 45, hi: int = 90, div: int = 2,
                   onset_prior: np.ndarray | None = None, beat_pos: np.ndarray | None = None,
                   gamma: float = 0.6, w_score: float = 2.5, w_f0: float = 1.0, max_len_tatum: int = 16,
                   rest_logp: float = -0.3, trans_frames: int = 4, note_cost: float = -1.0,
                   cross_prior: np.ndarray | None = None, in_key_w: float = 10.0,
                   chord_pcs: list | None = None, w_chord: float = 0.0, octave_w: float = 0.0,
                   rest_voiced: float = -3.0):
    """f0 (MIDI 単位, 10 ms) と格子から音符列 [(start, end, midi)] を推定する.

    onset_prior: 格子点ごとの「ここで音が始まる」対数事前 (歌詞の文字境界などから)。None なら一様。
    beat_pos   : 格子点ごとの拍節位置 (0=拍頭, 1=16分裏 ...) があればリズム事前に使う。
    cross_prior: 格子点ごとの「音がここをまたいで続く」対数事前 (歌詞の文字境界では負にして、音を区切り直させる)。
    chord_pcs  : 格子点ごとのコード構成音 (pitch class の集合、無ければ None)。w_chord > 0 なら構成音の音を優遇する。
    """
    scale_pcs = {(tonic + d) % 12 for d in (MAJOR if mode == "major" else MINOR)}
    K = hi - lo + 1
    N = len(lattice) - 1
    L = max_len_tatum * div
    # --- フレーム→格子の対応と、音高ごとの累積対数尤度 (O(1) で区間和を取るため)
    fr = np.clip(np.round(lattice / HOP_S).astype(int), 0, len(t))
    pitches = np.arange(lo, hi + 1, dtype=float)
    d = (f0_midi[:, None] - pitches[None, :]) / gamma
    ll_voiced = -np.log(1 + d ** 2) - np.log(np.pi * gamma)    # Cauchy
    if octave_w > 0:
        # 分離の残渣やハモリでオクターブ違いに飛ぶフレームは、同じ音の弱い証拠として扱う (休符にしない)
        d_up = (f0_midi[:, None] - (pitches[None, :] + 12)) / gamma
        d_dn = (f0_midi[:, None] - (pitches[None, :] - 12)) / gamma
        ll_oct = np.logaddexp(-np.log(1 + d_up ** 2), -np.log(1 + d_dn ** 2)) - np.log(np.pi * gamma) + np.log(octave_w)
        ll_voiced = np.logaddexp(ll_voiced, ll_oct)
    ll = np.where(voiced[:, None], ll_voiced, rest_logp)         # 無声フレームはどの音高でも同じ (小さな罰)
    ll_rest = np.where(voiced, rest_voiced, 0.0)                # 休符状態: 有声フレームは罰、無声は 0
    cum = np.vstack([np.zeros((1, K)), np.cumsum(ll, axis=0)])
    cum_rest = np.concatenate([[0.0], np.cumsum(ll_rest)])
    # --- 事前分布
    trans = _interval_logprior(K, scale_pcs, lo, in_key_w=in_key_w)
    pcs_k = np.array([(lo + i) % 12 for i in range(K)])
    chord_bonus = None
    if chord_pcs is not None and w_chord > 0:
        chord_bonus = np.zeros((N, K))
        for i in range(N):
            c = chord_pcs[i] if i < len(chord_pcs) else None
            if c:
                chord_bonus[i] = np.where(np.isin(pcs_k, list(c)), w_chord, 0.0)
    dur_lp = _duration_logprior(L, div)
    onset_lp = np.zeros(N) if onset_prior is None else np.asarray(onset_prior, dtype=float)
    if beat_pos is not None:
        # 拍頭 > 8 分裏 > 16 分裏 > 32 分の端数
        rhythm = np.array([0.0 if p % (4 * div) == 0 else (-0.3 if p % (2 * div) == 0 else (-0.6 if p % div == 0 else -1.5)) for p in beat_pos])
        onset_lp = onset_lp + rhythm
    # 休符の後の最初の音: 調内の音を優遇する (音程の事前は無い)
    pcs = np.array([(lo + i) % 12 for i in range(K)])
    init_w = np.where(np.isin(pcs, list(scale_pcs)), in_key_w, 1.0); init = np.log(init_w / init_w.sum())
    cross = np.zeros(N + 1) if cross_prior is None else np.asarray(cross_prior, dtype=float)[: N + 1]
    cum_cross = np.concatenate([[0.0], np.cumsum(cross)])   # 区間 [s, n) がまたぐ内部格子点 s+1..n-1 の和
    # --- 音の頭の移行区間 (ポルタメント): 前の音高から新しい音高へ移る途中とみなし、
    #     両方の音高の混合で観測する (論文の直線遷移モデルの簡略版)。
    #     区間は最大 trans_frames フレーム、かつ音の始まりから最大 TRANS_CELLS 格子まで (短い音では音全体が移行区間になり得る)
    Lk = np.exp(ll_voiced)                                     # [T, K] Cauchy 尤度 (有声フレーム用)
    c_bg = np.exp(-2.0)                                        # 休符から入る場合の相手側
    T_trans = np.zeros((TRANS_CELLS, N, K, K), dtype=np.float32)   # [c-1, s, prev, new]: s から c 格子以内の移行区間
    R_trans = np.zeros((TRANS_CELLS, N, K), dtype=np.float32)      # 休符から
    tf_of = np.zeros((TRANS_CELLS, N), dtype=int)
    for s in range(N):
        a = fr[s]
        acc_T = np.zeros((K, K)); acc_R = np.zeros(K); f = a
        for c in range(1, TRANS_CELLS + 1):
            b1 = fr[min(s + c, N)]
            end = min(a + trans_frames, b1, len(t))
            while f < end:
                if voiced[f]:
                    acc_T += np.log(0.5 * Lk[f][:, None] + 0.5 * Lk[f][None, :])
                    acc_R += np.log(0.5 * Lk[f] + 0.5 * c_bg)
                else:
                    acc_T += rest_logp; acc_R += rest_logp
                f += 1
            T_trans[c - 1, s] = acc_T; R_trans[c - 1, s] = acc_R; tf_of[c - 1, s] = max(f, a) - a
    # --- 半マルコフ Viterbi: best[n, k] = 格子点 n で終わる最後の音の音高 k のときの最良スコア
    NEG = -1e18
    best = np.full((N + 1, K), NEG)
    best_rest = np.full(N + 1, NEG)
    back = np.zeros((N + 1, K, 3), dtype=np.int64)       # (start, prev_pitch or -1(rest), 0)
    back_rest = np.zeros(N + 1, dtype=np.int64)          # 休符の開始点
    best_rest[0] = 0.0
    for n in range(1, N + 1):
        # 休符で n に至る (1 格子ずつ延長)
        # 休符が歌詞の文字境界をまたいで続くのも、音がまたぐのと同じ罰 (休符で境界を素通りさせない)
        cand_rest_from_rest = best_rest[n - 1] + (cum_rest[fr[n]] - cum_rest[fr[n - 1]]) + cross[n - 1] * w_score
        cand_rest_from_note = best[n - 1].max() + (cum_rest[fr[n]] - cum_rest[fr[n - 1]])
        if cand_rest_from_rest >= cand_rest_from_note:
            best_rest[n] = cand_rest_from_rest; back_rest[n] = back_rest[n - 1] if best_rest[n - 1] > NEG / 2 else n - 1
        else:
            best_rest[n] = cand_rest_from_note; back_rest[n] = n - 1
        # 音 [s, n) で n に至る
        for l in range(1, min(L, n) + 1):
            s = n - l
            a, b = fr[s], fr[n]
            if b <= a:
                continue
            c = min(TRANS_CELLS, l) - 1
            a2 = min(a + tf_of[c, s], b)
            emis = (cum[b] - cum[a2]) * w_f0                                  # [K] 安定区間
            prior_note = (dur_lp[l] + onset_lp[s] + note_cost + (cum_cross[n] - cum_cross[s + 1])) * w_score
            # 直前が休符
            from_rest = best_rest[s] + init * w_score + R_trans[c, s] * w_f0
            # 直前が音: max_q best[s, q] + trans[q, k] + 移行区間の尤度
            prev = best[s]
            if prev.max() > NEG / 2:
                cand = prev[:, None] + trans * w_score + T_trans[c, s] * w_f0    # [K_prev, K]
                q_best = np.argmax(cand, axis=0); from_note = cand[q_best, np.arange(K)]
            else:
                from_note = np.full(K, NEG); q_best = np.full(K, -1)
            use_rest = from_rest > from_note
            start_score = np.where(use_rest, from_rest, from_note) + prior_note + emis
            if chord_bonus is not None:
                start_score = start_score + chord_bonus[s] * w_score
            better = start_score > best[n]
            if np.any(better):
                best[n, better] = start_score[better]
                back[n, better, 0] = s
                back[n, better, 1] = np.where(use_rest, -1, q_best)[better]
    # --- バックトラック
    notes = []
    n = N
    if best[N].max() >= best_rest[N]:
        k = int(np.argmax(best[N])); in_note = True
    else:
        in_note = False
    while n > 0:
        if in_note:
            s, q = int(back[n, k, 0]), int(back[n, k, 1])
            notes.append((float(lattice[s]), float(lattice[n]), int(lo + k)))
            n = s
            if q < 0:
                in_note = False
            else:
                k = q
        else:
            s = int(back_rest[n])
            n = s
            if n > 0:
                k = int(np.argmax(best[n])); in_note = True
    notes.reverse()
    return notes


def lyric_priors(lattice: np.ndarray, chars, on: float = 2.0, inside: float = -3.0, cross: float = -6.0):
    """歌詞の文字境界 → (格子点ごとの音の始まりの事前, またぎ越しの事前)."""
    lp = np.zeros(len(lattice) - 1); cr = np.zeros(len(lattice) + 1)
    lim = 0.5 * float(np.median(np.diff(lattice))) + 1e-6
    for s_, e_ in chars:
        i = int(np.argmin(np.abs(lattice[:-1] - s_)))
        j0, j1 = np.searchsorted(lattice[:-1], s_ + 0.03), np.searchsorted(lattice[:-1], e_ - 0.03)
        lp[j0:j1] = np.minimum(lp[j0:j1], inside)
        lp[i] = on
        if abs(lattice[i] - s_) <= lim:
            cr[i] = cross
    return lp, cr


def merge_same_pitch(notes, onsets, tol: float = 0.03):
    """同じ高さで隙間なく続く音のうち、境界に発音の根拠 (歌詞の文字境界など) が無いものを 1 音にまとめる."""
    on = np.asarray(sorted(onsets), dtype=float)
    out = []
    for s_, e_, m in sorted(notes):
        if out and out[-1][2] == m and s_ - out[-1][1] <= 0.011 and not (len(on) and np.min(np.abs(on - s_)) <= tol):
            out[-1] = (out[-1][0], e_, m)
        else:
            out.append((s_, e_, m))
    return out


def chord_pcs_for_lattice(lattice: np.ndarray, chords) -> list:
    """Songle のコード [(start_s, end_s, (root, kind, pcs))] を格子点ごとの構成音集合に."""
    out = []
    j = 0
    for x in lattice[:-1]:
        while j < len(chords) and chords[j][1] <= x:
            j += 1
        if j < len(chords) and chords[j][0] <= x < chords[j][1] and chords[j][2]:
            out.append(set(chords[j][2][2]))
        else:
            out.append(None)
    return out


def merge_short_chars(chars, min_len: float = 0.07, max_gap: float = 0.03):
    """歌詞データで機械的に等分された極端に短い文字 (音節の境界ではない) を隣の文字に併合する."""
    out: list[list[float]] = []
    for s_, e_ in chars:
        if out and s_ - out[-1][1] <= max_gap and (e_ - s_ < min_len or out[-1][1] - out[-1][0] < min_len):
            out[-1][1] = max(out[-1][1], e_)
        else:
            out.append([s_, e_])
    return [(a, b) for a, b in out]


def link_phrases(notes, t, midi, voiced, phrases, *, max_gap: float = 0.25, consonant_gap: float = 0.12,
                 tol: float = 1.0, tail: float = 0.3):
    """楽譜モデルの音符列をフレーズに沿ってつなぐ (旧 connect_by_phrases の f0 を見る版).
    - 隣り合う音の隙間: 隙間の f0 が前の音の高さなら前の音を伸ばし、次の音の高さなら次の音を前に出す。
      f0 がほぼ無い短い隙間 (子音) は前の音を伸ばしてつなぐ。それ以外 (別の音程のごみ) は休符のまま
    - フレーズ末尾の音は、f0 が同じ高さで続く間だけ伸ばす
    戻り値: [(start, end, midi, vel, legato)]"""
    if not notes:
        return []
    ph = np.array(phrases) if phrases else np.zeros((0, 2))

    def phrase_of(x):
        k = np.where((ph[:, 0] - 0.05 <= x) & (x < ph[:, 1] + 0.05))[0] if len(ph) else []
        return int(k[0]) if len(k) else -1

    def near(fr_idx, p):
        m = midi[fr_idx]
        return np.abs(m - p) <= tol

    ns = [list(n) + [False] for n in sorted(notes, key=lambda n: n[0])]
    for i in range(len(ns) - 1):
        cur, nxt = ns[i], ns[i + 1]
        gap = nxt[0] - cur[1]
        if gap <= 0.011:
            cur[4] = True
            continue
        if gap > max_gap or phrase_of(cur[1]) < 0 or phrase_of(cur[1]) != phrase_of(nxt[0]):
            continue
        a, b = int(cur[1] / HOP_S), int(nxt[0] / HOP_S)
        idx = np.arange(a, min(b, len(midi)))
        vidx = idx[voiced[idx]] if len(idx) else idx
        if len(vidx) < 0.3 * max(1, len(idx)):
            if gap <= consonant_gap:
                cur[1] = nxt[0]; cur[4] = True
            continue
        # 前の音の高さで続く先頭部分
        e_new = cur[1]
        for f in idx:
            if voiced[f] and not near(f, cur[2]):
                break
            e_new = (f + 1) * HOP_S
        s_new = nxt[0]
        for f in idx[::-1]:
            if voiced[f] and not near(f, nxt[2]):
                break
            s_new = f * HOP_S
        if s_new <= e_new + 0.011:
            cur[1] = nxt[0]; cur[4] = True
        else:
            cur[1] = max(cur[1], e_new); nxt[0] = min(nxt[0], s_new)
    # フレーズ末尾
    for i, cur in enumerate(ns):
        if cur[4]:
            continue
        pi = phrase_of(cur[1] - 0.01)
        if pi < 0:
            continue
        lim = min(ph[pi, 1], cur[1] + tail, ns[i + 1][0] - 0.02 if i + 1 < len(ns) else 1e9)
        e_new = cur[1]
        for f in range(int(cur[1] / HOP_S), int(lim / HOP_S)):
            if f >= len(midi) or (voiced[f] and not near(f, cur[2])):
                break
            e_new = (f + 1) * HOP_S
        cur[1] = max(cur[1], e_new)
    return [tuple(n) for n in ns if n[1] - n[0] >= 0.03]


def refine_onsets(chars, t, midi, voiced, *, before: float = 0.03, after: float = 0.15, min_change: float = 0.8):
    """歌詞の文字の始まりを、実際に音高が変わる時刻に寄せる (人が採譜するときの「音の頭」).
    文字の始まりは子音の頭なので、母音で音高が変わるのはその 50〜120 ms 後になりやすい。
    前後の安定音高が min_change 半音以上違う文字だけ、f0 が両者の中間を横切る時刻を新しい始まりにする。
    同じ高さの歌い直しや、f0 が取れない文字はそのまま。"""
    out = []
    n = len(midi)
    for i, (s_, e_) in enumerate(chars):
        c = s_
        a0, a1 = int((c - 0.12) / HOP_S), int((c - before) / HOP_S)
        b0, b1 = int((c + 0.10) / HOP_S), int(min(c + 0.25, e_) / HOP_S)
        if a0 < 0 or b1 > n or b1 <= b0 + 2:
            out.append((s_, e_)); continue
        pv, nv = voiced[a0:a1], voiced[b0:b1]
        if pv.sum() < 3 or nv.sum() < 3:
            out.append((s_, e_)); continue
        p_prev = float(np.median(midi[a0:a1][pv])); p_next = float(np.median(midi[b0:b1][nv]))
        if abs(p_next - p_prev) < min_change:
            out.append((s_, e_)); continue
        mid = 0.5 * (p_prev + p_next)
        w0, w1 = int((c - before) / HOP_S), int((c + after) / HOP_S)
        new = None
        for f in range(w0, min(w1, n)):
            if voiced[f] and ((p_next > p_prev and midi[f] >= mid) or (p_next < p_prev and midi[f] <= mid)):
                new = f * HOP_S; break
        if new is None or new >= e_ - 0.03:
            out.append((s_, e_)); continue
        if out and new < out[-1][1]:
            out[-1] = (out[-1][0], new)      # 前の文字の終わりも合わせる
        out.append((new, e_))
    return out
