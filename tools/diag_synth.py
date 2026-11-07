"""楽譜の分かっている曲で、誤った音符がパイプラインのどの段階で生じたかを調べる (開発用).
歌ステム (分離後) の f0 が正解の高さを持っていたか / 有声だったか / 音量ゲートで落ちたかを数える."""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C

HOP = 0.01


def pipeline_f0(slug):
    import librosa
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    from chiptune.pitch import rmvpe_f0
    fp = C.CACHE / "synth_gt" / f"{slug}_pipef0.npz"
    if fp.exists():
        z = np.load(fp); return z["f0raw"], z["rms"]
    y = load_mono(C.CACHE / "synth_gt" / f"{slug}.wav", C.SR)
    voc = separate(y / (np.abs(y).max() + 1e-9), C.SR)["vocals"]
    t, f0 = rmvpe_f0(voc, C.SR, thred=0.03)
    rms = librosa.feature.rms(y=voc, frame_length=2048, hop_length=int(round(HOP * C.SR)))[0]
    n = min(len(f0), len(rms)); np.savez(fp, f0raw=f0[:n], rms=rms[:n]); return f0[:n], rms[:n]


if __name__ == "__main__":
    out = Path(sys.argv[1]); cnt = Counter()
    for tag in sys.argv[2:]:
        slug = tag.rsplit("_", 1)[0]
        ref = json.load(open(C.CACHE / "synth_gt" / f"{slug}.json"))["notes"]
        est = json.load(open(out / f"{tag}_notes.json"))
        f0, rms = pipeline_f0(slug)
        gate = rms >= np.percentile(rms, 95) * 0.04
        m = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440), np.nan)
        ef = np.full(len(m) + 100, np.nan)
        for s, e, p in est: ef[int(round(s / HOP)):int(round(e / HOP))] = p
        for s, e, p in ref:
            a, b = int(round(s / HOP)), int(round(e / HOP)); L = b - a; c0, c1 = a + L // 4, b - L // 4
            mid = ef[a + L // 2]
            if np.isfinite(mid) and mid == p: continue
            seg = m[c0:c1]; vv = np.isfinite(seg); gg = gate[c0:c1]
            if vv.mean() < 0.3: why = "f0 unvoiced (RMVPE)"
            elif (vv & gg).mean() < 0.3: why = "dropped by volume gate"
            else:
                med = np.median(seg[vv])
                if abs(med - p) < 0.5: why = "f0 correct -> lost later" + (" (silent)" if not np.isfinite(mid) else " (wrong pitch)")
                elif abs(abs(med - p) - 12) < 0.6: why = "f0 octave error"
                else: why = f"f0 off ({'<' if abs(med-p) < 1.5 else '>='}1.5st)"
            cnt[why] += 1
    tot = sum(cnt.values())
    for k, v in cnt.most_common(): print(f"  {k:40s} {v:5d} ({100*v/tot:.0f}%)")
