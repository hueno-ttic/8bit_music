"""楽譜 (正解の音符列) が正確に分かっているボカロ曲を作る (評価用).

ボカロ曲は作者が打ち込んだ音符を合成音声に歌わせたものなので、本来は正解の楽譜がある。それを再現するため、
  1. 元曲の歌ステム (Demucs) を WORLD で分析し、声色 (スペクトル包絡) と非周期成分を取り出す
  2. 正解の楽譜を作る: Songle の文字タイミングで 1 文字 = 1 音符、高さは元の歌の文字中央の f0 を半音に丸めた値
     (元の伴奏と調・和声が合う。同じ高さの文字が続いても別の音符 = 歌い直し)
  3. 楽譜から VOCALOID 風の音高曲線を作る: 音の変わり目のポルタメント (境界の 20〜50 ms 前から 30〜70 ms 後まで。
     実際のボカロ曲で測った「文字の頭と f0 の変わり目のずれ」に合わせた)、
     長い音のビブラート (0.25 秒後から, 5.5〜6.5 Hz, ±0.2〜0.5 半音)、ときどきのしゃくり (下から 1〜2 半音)、
     オーバーシュート、わずかな揺らぎ
  4. 元の声色・子音のまま新しい音高曲線で歌い直させ、元の伴奏 (ドラム・ベース・その他) と混ぜる
出力: ~/.cache/8bit_music/synth_gt/{slug}.wav と {slug}.json (正解の音符 [(start, end, midi)])
使い方: .venv/bin/python tools/make_synth_gt.py greenlights roki ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C  # noqa: E402

OUT = C.CACHE / "synth_gt"
HOP = 0.01


def score_from_song(slug: str):
    """正解の楽譜 [(start, end, midi)] (文字ごと)."""
    from chiptune.score_model import merge_short_chars
    d = C.features(slug)
    midi, voiced = d["midi"], d["voiced"].astype(bool)
    _, chars, _ = C.songle_data(slug)
    chars = merge_short_chars(chars)
    notes = []
    for i, (s, e) in enumerate(chars):
        nxt = chars[i + 1][0] if i + 1 < len(chars) else e
        a, b = int(s / HOP), int(e / HOP)
        if b - a < 4:
            continue
        q0, q1 = a + (b - a) // 4, max(b - (b - a) // 4, a + (b - a) // 4 + 1)
        v = voiced[q0:q1]
        if v.sum() < 2 or voiced[a:b].mean() < 0.4:
            continue
        p = int(np.round(np.median(midi[q0:q1][v])))
        end = min(nxt, e + 0.15) if nxt - e < 0.15 else e      # 次の文字まで (小さな隙間は埋める)
        notes.append([float(s), float(end), p])
    # 歌っていない (有声で無い) 尻尾を削る
    out = []
    for s, e, p in notes:
        a, b = int(s / HOP), int(e / HOP)
        vv = np.where(voiced[a:b])[0]
        if len(vv) == 0:
            continue
        e2 = min(e, (a + vv[-1] + 1) * HOP + 0.03)
        if e2 - s >= 0.05:
            out.append((s, e2, p))
    return out


def vocaloid_f0(notes, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """楽譜 → VOCALOID 風の f0 曲線 (MIDI, 歌っていない所は nan)."""
    f = np.full(n_frames, np.nan)
    for i, (s, e, p) in enumerate(notes):
        a, b = int(round(s / HOP)), min(n_frames, int(round(e / HOP)))
        if b <= a:
            continue
        t = np.arange(b - a) * HOP
        curve = np.full(b - a, float(p))
        prev = notes[i - 1] if i > 0 and s - notes[i - 1][1] < 0.05 else None
        if prev is not None and prev[2] != p:
            # ポルタメント: 境界の d1 秒前から d2 秒後まで滑らかに移る (実際のボカロ曲で、f0 が中間を横切るのは
            # 文字の頭から中央値 +10 ms (四分位 -28〜+56 ms) だったのに合わせる) + 小さなオーバーシュート
            ds = float(np.clip(rng.normal(0.01, 0.03), -0.04, 0.06))   # 変わり目の位置の揺らぎ (実測のばらつき相当)
            d1, d2 = max(0.005, rng.uniform(0.02, 0.05) - ds), max(0.01, rng.uniform(0.03, 0.07) + ds)
            k1, k2 = int(d1 / HOP), min(len(t), int(d2 / HOP))
            x = np.linspace(0, 1, k1 + k2 + 1)
            shape = 0.5 - 0.5 * np.cos(np.pi * x)
            trans = prev[2] + (p - prev[2]) * shape
            curve[:k2] = trans[k1 + 1:k1 + 1 + k2]
            pa = max(0, a - k1)
            if a > pa:
                f[pa:a] = trans[k1 - (a - pa) + 1:k1 + 1]
            curve += 0.15 * np.sign(p - prev[2]) * np.exp(-((t - d2 - 0.03) / 0.03) ** 2)
        elif rng.random() < 0.3 and e - s > 0.15:
            # しゃくり: 1〜2 半音下から上がる
            depth = rng.uniform(0.8, 2.0)
            k = min(len(t), int(rng.uniform(0.06, 0.1) / HOP))
            curve[:k] = p - depth * (1 - np.linspace(0, 1, k)) ** 2
        if e - s > 0.3:
            # ビブラート
            depth = rng.uniform(0.2, 0.5)
            rate = rng.uniform(5.5, 6.5)
            ramp = np.clip((t - 0.25) / 0.15, 0, 1)
            curve += depth * ramp * np.sin(2 * np.pi * rate * t)
        curve += rng.normal(0, 0.03, len(curve))
        f[a:b] = curve
    return f


def make(slug: str, seed: int = 0, repeat_prob: float = 0.0, out_dir: Path | None = None, tag: str | None = None):
    """repeat_prob > 0 なら、その確率で文字の高さを直前の文字と同じにする (同じ高さの歌い直しを増やした学習用データ)."""
    import pyworld as pw
    import soundfile as sf
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    out_dir = out_dir or OUT
    tag = tag or slug
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sr = C.SR
    y = load_mono(C.SONGS / f"{slug}.webm", sr=sr)
    y = y / (np.abs(y).max() + 1e-9)
    stems = separate(y, sr)
    voc = stems["vocals"].astype(np.float64)
    d = C.features(slug)
    f0_orig = d["f0"].astype(np.float64)
    n = len(f0_orig)
    tt = np.arange(n) * HOP
    sp = pw.cheaptrick(voc, f0_orig, tt, sr)
    ap = pw.d4c(voc, f0_orig, tt, sr)
    notes = score_from_song(slug)
    if repeat_prob > 0:
        notes = [notes[0]] + [(s_, e_, notes[i - 1][2] if (s_ - notes[i - 1][1] < 0.05 and rng.random() < repeat_prob) else p_)
                              for i, (s_, e_, p_) in enumerate(notes) if i > 0]
    fm = vocaloid_f0(notes, n, rng)
    # 有声にするのは、楽譜の音がある所のうち元の歌も有声だった所 (子音・息は元のまま)
    voiced = (f0_orig > 0) & np.isfinite(fm)
    f0_new = np.where(voiced, 440.0 * 2 ** ((np.nan_to_num(fm, nan=69.0) - 69) / 12), 0.0)
    # 楽譜の音が無い所の有声音 (歌詞に無いコーラス等) は消す (正解に無い歌を残さない)
    ap2 = ap.copy()
    ap2[(f0_orig > 0) & ~voiced] = 1.0
    sp2 = sp.copy()
    sp2[(f0_orig > 0) & ~voiced] *= 1e-4
    vs = pw.synthesize(f0_new, sp2, ap2, sr, frame_period=HOP * 1000)
    vs = vs[:len(voc)]
    if len(vs) < len(voc):
        vs = np.pad(vs, (0, len(voc) - len(vs)))
    g = np.sqrt(np.mean(voc ** 2) / (np.mean(vs ** 2) + 1e-12))
    acc = stems["drums"] + stems["bass"] + stems["other"]
    mix = acc + g * vs
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.9
    sf.write(out_dir / f"{tag}.wav", mix.astype(np.float32), sr)
    sf.write(out_dir / f"{tag}_vocal.wav", (g * vs).astype(np.float32), sr)
    json.dump({"notes": notes, "seed": seed, "repeat_prob": repeat_prob}, open(out_dir / f"{tag}.json", "w"))
    return len(notes)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print(s, make(s), "notes", flush=True)
