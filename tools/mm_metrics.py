"""音符列の評価 (歌詞タイミングを基準にした 過分割/過連結/音高誤り など)."""
from __future__ import annotations
import numpy as np
HOP_S = 0.01


def evaluate(notes, chars, midi, voiced, scale_pcs=None, tol=0.05):
    """notes: [(s, e, midi)], chars: [(s, e)]. 戻り値 dict."""
    notes = sorted(notes)
    starts = np.array([n[0] for n in notes]) if notes else np.zeros(0)
    # 過分割: 文字の内側 (境界から tol 以上離れた所) で始まる音
    over_split = 0
    for s_, e_ in chars:
        if e_ - s_ < 2 * tol:
            continue
        over_split += int(np.sum((starts > s_ + tol) & (starts < e_ - tol)))
    # 過連結: 文字の始まりに音の境界が無い (前の文字と連続していて、かつ音高も違う文字)
    over_conn = 0; missed = 0
    for i, (s_, e_) in enumerate(chars):
        if i == 0:
            continue
        ps, pe = chars[i - 1]
        if s_ - pe > 0.05 or e_ - s_ < 0.07 or pe - ps < 0.07:
            continue  # 機械的に等分された極端に短い文字は音節境界として信用しない
        a, b = int(s_ / HOP_S), int(e_ / HOP_S); pa, pb = int(ps / HOP_S), int(pe / HOP_S)
        va, vb = voiced[a:b], voiced[pa:pb]
        if va.sum() < 3 or vb.sum() < 3:
            continue
        has_b = len(starts) and np.min(np.abs(starts - s_)) <= tol
        if not has_b:
            missed += 1   # 文字の始まりに区切りが無い (同じ高さの歌い直しも含む)
        if abs(np.median(midi[a:b][va]) - np.median(midi[pa:pb][vb])) < 1.0:
            continue  # 同じ高さの連続文字は境界が無くても許容
        if has_b:
            continue
        over_conn += 1
    # 音高誤り: 音符の安定区間の f0 中央値と音符の音高が 1 半音以上違う
    pitch_err = 0; short_sus = 0; oct_err = 0; n_long = 0; merged_mov = 0
    for s_, e_, m in notes:
        a, b = max(0, int(s_ / HOP_S)), min(len(midi), int(e_ / HOP_S))
        n = b - a
        if n <= 0:
            continue
        if n >= 12:
            # 長めの音: 中央 50% の有声フレームの中央値と比べる (しゃくり・語尾を除く)
            sel = np.zeros(n, bool); sel[n // 4: n - n // 4] = True; sel &= voiced[a:b]
            if sel.sum() < 3:
                continue
            n_long += 1
            med = np.median(midi[a:b][sel])
            if abs(med - m) > 1.0:
                # 音符の高さが音の中のどこかで実際に歌われている (30% 以上のフレームが 0.6 半音以内) なら
                # 「音が外れた」のではなく「音の動きを 1 音にまとめた」(併合)
                allv = midi[a:b][voiced[a:b]]
                if len(allv) and np.mean(np.abs(allv - m) <= 0.6) >= 0.3:
                    merged_mov += 1
                else:
                    pitch_err += 1
            if abs(abs(med - m) - 12) < 1.0:
                oct_err += 1
        else:
            sel = voiced[a:b]
            if sel.sum() >= 3 and abs(np.median(midi[a:b][sel]) - m) > 1.0:
                short_sus += 1
    # 隣接半音差、調外
    semi = sum(1 for x, y in zip(notes[:-1], notes[1:]) if abs(x[2] - y[2]) == 1)
    outkey = sum(1 for n in notes if scale_pcs and n[2] % 12 not in scale_pcs)
    # 被覆と 2 半音以内一致 (フレーム)
    nm = np.full(len(midi), np.nan)
    for s_, e_, m in notes:
        nm[max(0, int(s_ / HOP_S)):int(e_ / HOP_S)] = m
    v = voiced & (np.arange(len(midi)) >= int(chars[0][0] / HOP_S) if chars else voiced)
    cov = float(np.mean(~np.isnan(nm[v]))) if v.any() else 0.0
    ok = v & ~np.isnan(nm)
    agree2 = float(np.mean(np.abs(nm[ok] - midi[ok]) <= 2)) if ok.any() else 0.0
    agree1 = float(np.mean(np.abs(nm[ok] - midi[ok]) <= 0.75)) if ok.any() else 0.0
    return dict(n=len(notes), over_split=over_split, over_conn=over_conn, missed=missed, pitch_err=pitch_err,
                pitch_err_pct=round(100.0 * pitch_err / max(1, n_long), 2), merged_mov=merged_mov, short_sus=short_sus, oct_err=oct_err,
                semi=semi, outkey=outkey, coverage=round(cov * 100, 1), agree2=round(agree2 * 100, 1), agree1=round(agree1 * 100, 1))


def fmt(m):
    return (f"音 {m['n']}, 過分割 {m['over_split']}, 過連結 {m['over_conn']}, 区切り無し {m['missed']}, 外れ {m['pitch_err']} ({m['pitch_err_pct']}%), 併合 {m['merged_mov']}, 短音疑い {m['short_sus']}, "
            f"オクターブ {m['oct_err']}, 隣接半音差 {m['semi']}, 調外 {m['outkey']}, 被覆 {m['coverage']}%, 2半音以内 {m['agree2']}%, 0.75半音以内 {m['agree1']}%")
