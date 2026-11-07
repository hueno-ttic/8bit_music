"""採譜結果の自己検証と修正の繰り返し.

生成した音符列を歌の f0 と照合し、問題のある音を直して、目標に達するまで (上限回数まで) 繰り返す。
  外れ  : 音の中央 50% の f0 中央値が音符の高さから 1 半音超ずれ、かつその高さが音の中のどこにも歌われていない
          → 安定区間の f0 から高さを取り直す
  短音外れ: 120 ms 未満の音で f0 中央値が 1 半音超ずれ → 高さを取り直す
  半音外れ: 80 ms 以上の音で、中央 50% の歌が別の半音にはっきり留まっている (中央値が半音の中央 ±0.3、揺れ 0.8 半音以内)
          → その半音に直す (「隣のキーを押した」誤り。1 半音超の判定では見逃していた)
  併合  : 長い音の中に、音符の高さから 1 半音以上離れた別の高さの安定区間 (120 ms 以上) がある → そこで音を分ける
  取りこぼし: 歌が 60 ms 以上はっきり 1 つの半音に留まった区間で別の高さが鳴っている (切り替えが早い/遅い、
          短い音節が隣にのみ込まれる) → その区間をその半音の音にする
合否は「外れ (短音含む) の割合 <= target」。
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01
LONG = 12          # フレーム (120 ms) 以上を「長い音」とする
MERGE_RUN = 12     # 併合とみなす別音高の安定区間の最短フレーム数
SEMI_MIN = 8       # 半音外れを調べる最短の音 (80 ms)
PLATEAU_MIN = 6    # 取りこぼしとみなす、歌が 1 つの半音に留まる最短の長さ (60 ms)
PLATEAU_NEW_MIN = 10  # これより短い区間は、境界の修正か音節の始まりがある場合だけ音を立てる (100 ms)
PLATEAU_CHECK = True


def _seg(midi, voiced, s, e):
    a = max(0, int(s / HOP_S)); b = min(len(midi), max(int(e / HOP_S), a + 1))
    return a, b, midi[a:b], voiced[a:b]


def _best_pitch(vals: np.ndarray) -> int:
    """f0 (MIDI 連続値) の集合に最も合う整数音高: 各候補の ±0.6 半音内フレーム数が最大のもの (同数なら中央値に近い方)."""
    med = float(np.median(vals))
    cands = {int(np.floor(med)), int(np.ceil(med)), int(np.round(med))}
    return max(cands, key=lambda p: (np.sum(np.abs(vals - p) <= 0.6), -abs(med - p)))


def check(notes, midi, voiced):
    """問題のある音を列挙する. 戻り値: (issues, n_checked). issue = dict(idx, kind, pitch | split)."""
    issues = []
    n_checked = 0
    for i, n in enumerate(notes):
        a, b, m, v = _seg(midi, voiced, n[0], n[1])
        L = b - a
        if v.sum() < 3:
            continue
        n_checked += 1
        p = n[2]
        if L >= LONG:
            sel = np.zeros(L, bool); sel[L // 4: L - L // 4] = True; sel &= v
            if sel.sum() >= 3:
                med = float(np.median(m[sel]))
                allv = m[v]
                if abs(med - p) > 1.0 and np.mean(np.abs(allv - p) <= 0.6) < 0.3:
                    issues.append(dict(idx=i, kind="外れ", pitch=_best_pitch(m[sel])))
                    continue
            # 併合: 頭 6 フレームを除き、音符の高さから 1 半音以上離れた別の高さの安定区間を探す
            if L >= 2 * MERGE_RUN:
                dev = np.where(v, np.abs(m - p), 0.0)
                run_s = None
                for f in range(6, L + 1):
                    on = f < L and v[f] and dev[f] >= 1.0
                    if on and run_s is None:
                        run_s = f
                    elif not on and run_s is not None:
                        if f - run_s >= MERGE_RUN:
                            seg = m[run_s:f][v[run_s:f]]
                            if np.std(seg) < 0.5:
                                q = _best_pitch(seg)
                                if q != p:
                                    issues.append(dict(idx=i, kind="併合", split=((a + run_s) * HOP_S, (a + f) * HOP_S), pitch=q))
                                    break
                        run_s = None
        else:
            med = float(np.median(m[v]))
            if abs(med - p) > 1.0:
                issues.append(dict(idx=i, kind="短音外れ", pitch=_best_pitch(m[v])))
                continue
        if issues and issues[-1]["idx"] == i:
            continue
        # 半音外れ: 中央 50% で歌がはっきり別の半音に留まっている (安定・半音の中央 ±0.3) のに音符がその隣.
        # 上の判定 (1 半音超) では「隣のキーを押した」誤りを見逃し、自己検証が合格と報告してしまう
        if L >= SEMI_MIN:
            sel = np.zeros(L, bool); sel[L // 4: L - L // 4] = True; sel &= v
            if sel.sum() >= 5:
                vals = m[sel]
                med = float(np.median(vals))
                q = int(np.round(med))
                if (q != p and abs(med - q) <= 0.3 and np.percentile(vals, 90) - np.percentile(vals, 10) <= 0.8
                        and np.mean(np.abs(vals - q) <= 0.5) >= 0.6):
                    issues.append(dict(idx=i, kind="半音外れ", pitch=q))
    return issues, n_checked


def fix(notes, issues, min_len: float = 0.05):
    """issues を適用した新しい音符列を返す (音符は (start, end, midi, ...) のタプル; 余分な要素は保持)."""
    notes = [list(n) for n in notes]
    extra = []
    for it in issues:
        n = notes[it["idx"]]
        if it["kind"] in ("外れ", "短音外れ", "半音外れ"):
            n[2] = int(it["pitch"])
        else:  # 併合: 別音高の区間を切り出す
            s0, e0 = n[0], n[1]
            rs, re_ = it["split"]
            rs = max(s0, rs); re_ = min(e0, re_)
            if re_ - rs < min_len:
                continue
            parts = []
            if rs - s0 >= min_len:
                parts.append([s0, rs, n[2]] + list(n[3:]))
            parts.append([rs, re_, int(it["pitch"])] + list(n[3:]))
            if e0 - re_ >= min_len:
                parts.append([re_, e0, n[2]] + list(n[3:]))
            else:
                parts[-1][1] = e0
            if rs - s0 < min_len:
                parts[0][0] = s0
            n[0] = -1  # 削除印
            extra.extend(parts)
    out = [n for n in notes if n[0] >= 0] + extra
    out.sort(key=lambda n: n[0])
    # 同じ高さで隣接した音が分割で並んだら戻す
    merged = []
    for n in out:
        if merged and merged[-1][2] == n[2] and abs(n[0] - merged[-1][1]) < 1e-6 and len(n) > 3 and len(merged[-1]) > 3 \
                and n[3:] == merged[-1][3:] and False:
            merged[-1][1] = n[1]
        else:
            merged.append(n)
    return [tuple(n) for n in merged]


def plateaus(midi, voiced, min_len: int = PLATEAU_MIN, tol: float = 0.35, step: float = 0.2, spread: float = 0.6):
    """歌がはっきり 1 つの半音に留まっている区間 [(開始フレーム, 終了フレーム, 半音)].
    隣り合うフレームの変化が step 半音以下で、区間の揺れ (最大 - 最小) が spread 以下の有声区間のうち、
    中央値が半音の中央 ±tol 以内のもの (半音の中央から少しずれて歌われても、安定していれば取る)."""
    n = len(midi)
    out = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        lo = hi = midi[i]
        while j < n and voiced[j] and abs(midi[j] - midi[j - 1]) <= step and max(hi, midi[j]) - min(lo, midi[j]) <= spread:
            lo, hi = min(lo, midi[j]), max(hi, midi[j])
            j += 1
        if j - i >= min_len:
            med = float(np.median(midi[i:j]))
            if abs(med - np.round(med)) <= tol:
                out.append((i, j, int(np.round(med))))
            i = j
        else:
            i += 1
    return out


def plateau_issues(notes, midi, voiced, onsets=None):
    """取りこぼし: 歌が 70 ms 以上はっきり留まった半音が、その区間で鳴っていない (隣の音が早く/遅く切り替わる、
    短い音節が隣の音にのみ込まれる). 区間の中央 60% のうち 6 割以上で別の高さが鳴っていれば問題とする."""
    if not notes:
        return []
    ns = sorted(notes, key=lambda x: x[0])
    starts = np.array([x[0] for x in ns]); ends = np.array([x[1] for x in ns]); pit = np.array([x[2] for x in ns])
    frame_pitch = np.full(len(midi), -1)
    for s_, e_, m_ in zip(starts, ends, pit):
        frame_pitch[max(0, int(round(s_ / HOP_S))):min(len(midi), int(round(e_ / HOP_S)))] = m_
    issues = []
    for a, b, q in plateaus(midi, voiced):
        L = b - a
        c0, c1 = a + int(0.2 * L), b - int(0.2 * L)
        core = frame_pitch[c0:c1]
        if not len(core) or np.mean(core < 0) >= 0.5:      # 休符の所は扱わない
            continue
        # 中央で正しい高さが鳴っていて、区間全体でも (切り替えの遅れ・早まりが 30 ms 程度まで) 鳴っていれば問題なし
        whole = frame_pitch[a:b]
        if np.mean(core == q) >= 0.6 and np.sum(whole != q) <= 3:
            continue
        # 分離の残渣・オクターブ飛び (前後 2 秒の音域から 10 半音以上外れる) は扱わない
        near = pit[(ends > a * HOP_S - 2.0) & (starts < b * HOP_S + 2.0)]
        if len(near) and abs(q - float(np.median(near))) > 10:
            continue
        # 短い区間 (100 ms 未満) は装飾 (しゃくりの入り口・経過音) のことが多い。新しい音を立てるのは、
        # 隣の音がもうその高さで境界をずらすだけで済む場合か、そこに音節の始まりがある場合だけ
        if L < PLATEAU_NEW_MIN:
            adjacent_q = np.any(frame_pitch[max(0, a - 3):a] == q) or np.any(frame_pitch[b:b + 3] == q) or np.any(core == q)
            has_onset = onsets is not None and len(onsets) and np.any((onsets >= a * HOP_S - 0.06) & (onsets <= b * HOP_S))
            if not (adjacent_q or has_onset):
                continue
        issues.append(dict(kind="取りこぼし", span=(a * HOP_S, b * HOP_S), pitch=q))
    return issues


def paint(notes, span, q, min_len: float = 0.03):
    """区間 span を高さ q の音にする (重なる音は切り詰め/分割し、隣接する q の音とはつなぐ)."""
    a, b = span
    out = []
    template = None
    for n in sorted(notes, key=lambda x: x[0]):
        n = list(n)
        if n[1] <= a or n[0] >= b:
            out.append(n)
            continue
        template = template or n
        if n[0] < a and a - n[0] >= min_len:
            out.append([n[0], a] + n[2:])
        if n[1] > b and n[1] - b >= min_len:
            out.append([b, n[1]] + n[2:])
    if template is None:
        return [tuple(n) for n in notes]
    new = [a, b, int(q)] + list(template[3:])
    if len(new) > 4:
        new[4] = True
    out.append(new)
    out.sort(key=lambda x: x[0])
    merged = []
    for n in out:
        if merged and merged[-1][2] == n[2] and n[0] - merged[-1][1] < 0.02 and (a - 0.02 <= n[0] <= b + 0.02 or a - 0.02 <= merged[-1][1] <= b + 0.02):
            merged[-1][1] = max(merged[-1][1], n[1])
            if len(n) > 4:
                merged[-1][4] = n[4]
        else:
            merged.append(n)
    # 重なりの解消
    for i in range(1, len(merged)):
        if merged[i][0] < merged[i - 1][1]:
            merged[i - 1][1] = merged[i][0]
    return [tuple(n) for n in merged if n[1] - n[0] >= 0.01]


def verify_loop(notes, midi, voiced, *, max_iter: int = 5, target: float = 0.001, progress=None, log=None, onsets=None):
    """検証 → 修正 を繰り返す. 戻り値: (notes, report)
    report = dict(iterations, passed, rate, n_bad, n_checked, target, history=[{iter, n_bad, n_merge, rate}])"""
    history = []
    cur = list(notes)
    passed = False
    rate = 0.0; n_bad = 0; n_checked = 0
    for k in range(max_iter + 1):
        issues, n_checked = check(cur, midi, voiced)
        if PLATEAU_CHECK:
            issues = issues + plateau_issues(cur, midi, voiced, None if onsets is None else np.asarray(onsets, float))
        bad = [x for x in issues if x["kind"] != "併合"]
        merges = [x for x in issues if x["kind"] == "併合"]
        n_bad = len(bad)
        rate = n_bad / max(1, n_checked)
        history.append(dict(iter=k, n_bad=n_bad, n_merge=len(merges), rate=rate))
        msg = f"自己検証 {k}/{max_iter} 回目: 外れ {n_bad} 音 ({rate * 100:.2f}%), 併合 {len(merges)} 音 / 検証 {n_checked} 音"
        if log:
            log.info(msg)
        if progress:
            progress(msg)
        if rate <= target and not merges:
            passed = True
            break
        if k == max_iter:
            passed = rate <= target
            break
        cur = fix(cur, [x for x in issues if x["kind"] != "取りこぼし"])
        for it in issues:
            if it["kind"] == "取りこぼし":
                cur = paint(cur, it["span"], it["pitch"])
    return cur, dict(iterations=len(history) - 1, passed=passed, rate=rate, n_bad=n_bad, n_checked=n_checked,
                     target=target, max_iter=max_iter, history=history)
