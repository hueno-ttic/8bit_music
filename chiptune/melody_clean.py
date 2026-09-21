"""推定した音符列の音楽理論に基づく後処理 (自動編曲の前段).

楽譜モデルの出力に対して、8bit で「音が外れて聞こえる」原因になりやすいものだけを直す:
  1. しゃくり (ポルタメント) の音: 1 音の中で f0 が単調に動いている場合、人が聞き取る音高は
     着地した高さなので、後半の安定区間から音高を取り直す
  2. ごみ音: 分離の残渣などで生じた、前後から大きく跳んで戻る極端に短い音は落とし、隣の音で埋める
  3. 半音の隣: 調にもコードにも無い短い音で、f0 が半音の境目 (小数部 0.35 以上) にあり、
     隣接する音と半音差なら、その隣の音の高さに揃える (「隣のキーを押した」誤り)
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.01


def _seg(midi, voiced, s, e):
    a, b = max(0, int(s / HOP_S)), max(int(e / HOP_S), int(s / HOP_S) + 1)
    return midi[a:b], voiced[a:b]


def landing_pitch(m, v, min_len: int = 12, min_range: float = 0.8):
    """f0 が単調に動く音なら着地区間 (後半 40%) の中央値を返す。そうでなければ None."""
    if len(m) < min_len or v.sum() < min_len * 0.6:
        return None
    x = m.copy(); x[~v] = np.nan
    idx = np.where(v)[0]
    if len(idx) < 8:
        return None
    first, last = idx[: max(3, len(idx) // 4)], idx[-max(3, int(len(idx) * 0.4)):]
    a, b = np.nanmedian(x[first]), np.nanmedian(x[last])
    if abs(b - a) < min_range:
        return None
    # 単調性: 前半→後半へ一方向に動き、途中で戻っていない
    mid = idx[len(idx) // 4: -max(3, int(len(idx) * 0.4))]
    if len(mid) and ((b > a and np.nanmin(x[mid]) < a - 0.5) or (b < a and np.nanmax(x[mid]) > a + 0.5)):
        return None
    tail = x[last]
    if np.nanstd(tail) > 0.6:
        return None
    return float(b)


def plateau_pitch(m, v, energy=None, min_len: int = 12):
    """頭 (最大 60 ms) と尻 (30 ms) を除いた安定区間の中央値が整数に近い (±0.35) ならその整数、なければ None."""
    n = len(m)
    if n < min_len:
        return None
    head = min(6, n // 3); tail = min(3, max(0, (n - head) // 4))
    sel = np.zeros(n, bool); sel[head:n - tail] = True; sel &= v
    if sel.sum() < 6:
        return None
    vals = m[sel]
    med = float(np.median(vals))
    if abs(med - np.round(med)) > 0.35:
        return None
    # 安定区間の半分以上がその高さ (±0.6) にあること (2 つの高さの中間値を採らない)
    if np.mean(np.abs(vals - np.round(med)) <= 0.6) < 0.5:
        return None
    return int(np.round(med))


def clean(notes, midi, voiced, scale_pcs: set[int] | None, chord_at=None, *, in_scale_snap: bool = False,
          landing: bool = False, plateau: bool = False):
    """notes: [(start, end, midi, vel)] → 同じ形式. chord_at(t) は構成音 pc の集合か None を返す."""
    notes = [list(n) for n in sorted(notes, key=lambda n: n[0])]
    stats = {"landing": 0, "garbage": 0, "neighbor": 0}
    # 1. しゃくりの着地音高 (既定は無効: 楽譜モデルの移行区間モデルで扱う。着地先が調外になる例が多かった)
    for n in notes if landing else []:
        m, v = _seg(midi, voiced, n[0], n[1])
        lp = landing_pitch(m, v)
        if lp is not None:
            cand = int(np.round(lp))
            if cand != n[2] and abs(cand - lp) <= 0.35:
                n[2] = cand; stats["landing"] += 1
    # 1b. 安定区間の音高が明確で、モデルの音高と違うなら f0 の証拠を優先する (しゃくりが長い音)
    if plateau:
        for n in notes:
            m, v = _seg(midi, voiced, n[0], n[1])
            pp = plateau_pitch(m, v)
            if pp is not None and pp != n[2]:
                n[2] = pp; stats["plateau"] = stats.get("plateau", 0) + 1
    # 2. ごみ音
    out = []
    for i, n in enumerate(notes):
        dur = n[1] - n[0]
        prev = out[-1] if out else None
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        if dur < 0.07 and prev is not None and nxt is not None and n[0] - prev[1] < 0.03 and nxt[0] - n[1] < 0.03 \
                and abs(n[2] - prev[2]) >= 7 and abs(n[2] - nxt[2]) >= 7:
            prev[1] = n[1]; stats["garbage"] += 1
            continue
        # f0 がまばら (有声 50% 未満) な短い音で、前後から 5 半音以上跳ぶものも残渣とみなす
        m_, v_ = _seg(midi, voiced, n[0], n[1])
        if dur < 0.15 and v_.mean() < 0.5 and (prev is None or abs(n[2] - prev[2]) >= 5) and (nxt is None or abs(n[2] - nxt[2]) >= 5):
            stats["garbage"] += 1
            continue
        out.append(n)
    notes = out
    # 2b. 音域の外れ音: 前後 3 秒の音域 (長さ重み中央値) から 9 半音以上離れた短い音 (ハモリ・ブレスの残渣)
    out = []
    for i, n in enumerate(notes):
        dur = n[1] - n[0]
        if dur < 0.3:
            near = [(x[1] - x[0], x[2]) for x in notes if x is not n and x[1] > n[0] - 3.0 and x[0] < n[1] + 3.0]
            if near:
                w = np.array([d for d, _ in near]); ps = np.array([p for _, p in near])
                order = np.argsort(ps); c = np.cumsum(w[order]) / w.sum()
                center = ps[order][int(np.searchsorted(c, 0.5))]
                if abs(n[2] - center) >= 9:
                    stats["register"] = stats.get("register", 0) + 1
                    continue
        out.append(n)
    notes = out
    # 3. 半音の隣
    if in_scale_snap and scale_pcs:
        for i, n in enumerate(notes):
            if n[2] % 12 in scale_pcs or n[1] - n[0] > 0.3:
                continue
            ch = chord_at(0.5 * (n[0] + n[1])) if chord_at else None
            if ch and n[2] % 12 in ch:
                continue
            m, v = _seg(midi, voiced, n[0], n[1])
            if v.sum() < 3:
                continue
            med = float(np.median(m[v])); frac = abs(med - np.round(med))
            if frac < 0.35:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(notes) and abs(notes[j][2] - n[2]) == 1 and notes[j][2] % 12 in scale_pcs \
                        and abs(notes[j][2] - med) <= 0.75:
                    n[2] = notes[j][2]; stats["neighbor"] += 1
                    break
    return [tuple(n) for n in notes], stats
