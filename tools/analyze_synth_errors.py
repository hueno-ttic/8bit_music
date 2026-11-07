"""楽譜の分かっている曲での誤りを種類別に数える (tools/eval_synth.py の出力を使う).
使い方: .venv/bin/python tools/analyze_synth_errors.py <eval_out/synth_xxx> <tag> [<tag> ...]"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C

HOP = 0.01


def analyze(out: Path, tag: str):
    slug = tag.rsplit("_", 1)[0]
    ref = json.load(open(C.CACHE / "synth_gt" / f"{slug}.json"))["notes"]
    est = json.load(open(out / f"{tag}_notes.json"))
    n = int(max(ref[-1][1], est[-1][1] if est else 0) / HOP) + 10
    ef = np.full(n, np.nan)
    for s, e, m in est:
        ef[int(round(s / HOP)):int(round(e / HOP))] = m
    frames = Counter(); notes = Counter(); tot = 0
    for i, (s, e, p) in enumerate(ref):
        a, b = int(round(s / HOP)), int(round(e / HOP)); L = b - a
        for k in range(a, b):
            pos = "head" if k - a < max(1, L // 4) else ("tail" if k - a >= L - max(1, L // 4) else "body")
            x = ef[k]; tot += 1
            if not np.isfinite(x): frames[f"silent/{pos}"] += 1
            elif x != p:
                d = int(x - p)
                kind = "oct" if abs(d) in (12, 24) else (f"{d:+d}" if abs(d) <= 2 else "far")
                frames[f"{kind}/{pos}"] += 1
        # 音符単位: 始まり (±50ms に est の始まりがあるか)、高さ (中央の est)
        on = any(abs(es - s) <= 0.05 for es, ee, em in est)
        mid = ef[a + L // 2] if L else np.nan
        pitch_ok = np.isfinite(mid) and mid == p
        prev_same = i > 0 and ref[i - 1][2] == p and s - ref[i - 1][1] < 0.05
        if pitch_ok and on: notes["ok"] += 1
        elif pitch_ok and not on: notes["onset_missing" + ("(same-pitch repeat)" if prev_same else "")] += 1
        elif not np.isfinite(mid): notes["silent"] += 1
        else: notes[f"pitch_wrong{' short' if e - s < 0.15 else ''}"] += 1
    return frames, notes, tot, len(ref)


if __name__ == "__main__":
    out = Path(sys.argv[1]); F = Counter(); N = Counter(); T = R = 0
    for tag in sys.argv[2:]:
        f, nn, t, r = analyze(out, tag); F += f; N += nn; T += t; R += r
    print(f"frames (GT note time) {T}: wrong/silent {100 * sum(F.values()) / T:.1f}%")
    for k, v in F.most_common(14): print(f"   {k:16s} {100 * v / T:5.2f}%")
    print(f"notes {R}:"); [print(f"   {k:32s} {100 * v / R:5.1f}%") for k, v in N.most_common()]
