"""採譜結果の自己検証と修正の繰り返し.

生成した音符列を歌の f0 と照合し、問題のある音を直して、目標に達するまで (上限回数まで) 繰り返す。
  外れ  : 音の中央 50% の f0 中央値が音符の高さから 1 半音超ずれ、かつその高さが音の中のどこにも歌われていない
          → 安定区間の f0 から高さを取り直す
  短音外れ: 120 ms 未満の音で f0 中央値が 1 半音超ずれ → 高さを取り直す
  併合  : 長い音の中に、音符の高さから 1 半音以上離れた別の高さの安定区間 (120 ms 以上) がある → そこで音を分ける
合否は「外れ (短音含む) の割合 <= target」。
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01
LONG = 12          # フレーム (120 ms) 以上を「長い音」とする
MERGE_RUN = 12     # 併合とみなす別音高の安定区間の最短フレーム数


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
    return issues, n_checked


def fix(notes, issues, min_len: float = 0.05):
    """issues を適用した新しい音符列を返す (音符は (start, end, midi, ...) のタプル; 余分な要素は保持)."""
    notes = [list(n) for n in notes]
    extra = []
    for it in issues:
        n = notes[it["idx"]]
        if it["kind"] in ("外れ", "短音外れ"):
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


def verify_loop(notes, midi, voiced, *, max_iter: int = 5, target: float = 0.001, progress=None, log=None):
    """検証 → 修正 を繰り返す. 戻り値: (notes, report)
    report = dict(iterations, passed, rate, n_bad, n_checked, target, history=[{iter, n_bad, n_merge, rate}])"""
    history = []
    cur = list(notes)
    passed = False
    rate = 0.0; n_bad = 0; n_checked = 0
    for k in range(max_iter + 1):
        issues, n_checked = check(cur, midi, voiced)
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
        cur = fix(cur, issues)
    return cur, dict(iterations=len(history) - 1, passed=passed, rate=rate, n_bad=n_bad, n_checked=n_checked,
                     target=target, max_iter=max_iter, history=history)
