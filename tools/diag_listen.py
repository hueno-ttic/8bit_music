"""聴感の診断: 8bit の曲でメロディがメロディとして聞こえるかを数える (開発用).
  - 各パートの音量 (メロディが伴奏より目立つか)
  - 歌っている区間でのメロディの途切れ (無音・音量のくぼみの回数)
  - 音域 (メロディ / ベース / アルペジオ)
  - 完成した 8bit の曲 (全パート) に歌声向けのメロディ抽出 (RMVPE) を掛けたとき、何の音が聞こえるか
使い方: .venv/bin/python tools/diag_listen.py vampire [--file]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C


def frames_of(notes, n, hop=0.01):
    f = np.full(n, np.nan)
    for nt in notes:
        f[int(nt.start / hop):int(nt.end / hop)] = nt.midi
    return f


def main(slug, use_songle=True, score=None, follow=False):
    import librosa
    from eval_8bit import enable_fast_cache
    enable_fast_cache()
    from chiptune.analysis import SR
    from chiptune.analysis_sep import analyze_sep
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    from chiptune import synth as S
    from chiptune.pitch import rmvpe_f0
    from chiptune.master import finalize
    y = load_mono(C.SONGS / f"{slug}.webm", SR)
    if score is None:
        score = analyze_sep(y, SR, stems=separate(y / (np.abs(y).max() + 1e-9), SR), source_url=C.url_of(slug) if use_songle else None)
    sr = S.OUT_SR
    total = int(np.ceil(score.duration * sr)) + sr // 10
    melody_sig = (S.render_follow_melody(score.melody, score.contour, total, sr) if follow
                  else S.render_melody_with_echo(score.melody, total, sr, score.tempo))
    ch = dict(melody=S.MIX_MELODY * melody_sig,
              arp=S.MIX_ARP * S.render_arpeggio(score.chords, score.cells, total, sr, base_octave=score.arp_octave,
                                                bar_starts=score.bar_starts, intensity=score.intensity, melody=score.melody),
              bass=S.MIX_BASS * S.render_triangle_bass(score.bass, total, sr),
              drums=S.MIX_DRUMS * S.render_drums(score.drums, total, sr))
    d = C.features(slug); v = d["voiced"].astype(bool); vm = d["midi"]
    n = len(v); hop = sr // 100
    def rms(x): return np.sqrt(np.convolve(x[: n * hop] ** 2, np.ones(hop) / hop, "same")[::hop][:n] + 1e-12)
    R = {k: rms(x) for k, x in ch.items()}
    # 歌っている区間 (0.3 秒未満の途切れはつなぐ)
    sung = v.copy()
    gaps = np.where(~v)[0]
    i = 0
    while i < n:
        if not sung[i]:
            j = i
            while j < n and not sung[j]: j += 1
            if 0 < i and j < n and (j - i) < 30: sung[i:j] = True
            i = j
        else: i += 1
    mel_on = R["melody"] > 0.05 * np.percentile(R["melody"][R["melody"] > 1e-4], 90)
    acc = np.sqrt(R["arp"] ** 2 + R["bass"] ** 2 + R["drums"] ** 2)
    ratio = 20 * np.log10(R["melody"][sung & mel_on] / (acc[sung & mel_on] + 1e-9))
    print(f"== {slug} ({'songle' if use_songle else 'file'}, {'follow' if follow else 'notes'})")
    print(f"  level vs melody (sung time): arp {20*np.log10(np.median(R['arp'][sung])/np.median(R['melody'][sung & mel_on])):+.1f} dB, "
          f"bass {20*np.log10(np.median(R['bass'][sung])/np.median(R['melody'][sung & mel_on])):+.1f} dB, drums {20*np.log10(np.median(R['drums'][sung])/np.median(R['melody'][sung & mel_on])):+.1f} dB; melody/accomp median {np.median(ratio):+.1f} dB, <0dB {100*np.mean(ratio<0):.0f}% of time")
    # 途切れ
    off = sung & ~mel_on
    k = np.diff(np.concatenate([[0], off.astype(int), [0]])); n_gaps = int((k == 1).sum())
    print(f"  melody silent during singing: {100*off.sum()/sung.sum():.1f}% of sung time, {n_gaps} gaps ({n_gaps/(sung.sum()/100):.2f}/s)")
    mel = np.array([nt.midi for nt in score.melody]); bas = np.array([nt.midi for nt in score.bass])
    print(f"  register: vocal f0 p10/50/90 {np.percentile(vm[v],[10,50,90]).round(1)}, melody notes {np.percentile(mel,[10,50,90])}, bass {np.percentile(bas,[10,50,90])}, arp octave {score.arp_octave} (MIDI {12*score.arp_octave}-{12*score.arp_octave+11})")
    # 完成した曲から何が聞こえるか
    mix = finalize(sum(ch.values()), sr)
    t, f0 = rmvpe_f0(librosa.resample(mix, orig_sr=sr, target_sr=SR), SR)
    m = min(n, len(f0)); ex = np.where(f0[:m] > 0, librosa.hz_to_midi(np.maximum(f0[:m], 1e-3)), np.nan)
    mf = frames_of(score.melody, m); bf = frames_of(score.bass, m)
    if follow:
        fp, _ = S.follow_frames(score.melody, score.contour, score.duration)
        mf = fp[np.clip((np.arange(m) * 0.01 / S.NES_FRAME).astype(int), 0, len(fp) - 1)]
    sel = sung[:m] & np.isfinite(ex)
    def near(a, b): return np.isfinite(b) & (np.abs(a - b) < 0.6)
    def nearoct(a, b): return np.isfinite(b) & (np.abs(((a - b) + 6) % 12 - 6) < 0.6)
    print(f"  melody extractor on full 8bit mix (sung time, voiced {100*np.mean(np.isfinite(ex[sung[:m]])):.0f}%): "
          f"= 8bit melody {100*np.mean(near(ex, mf)[sel]):.0f}%, = vocal pitch {100*np.mean(near(ex, np.where(v[:m], vm[:m], np.nan))[sel]):.0f}%, "
          f"= bass (any oct) {100*np.mean(nearoct(ex, bf)[sel] & ~near(ex, mf)[sel]):.0f}%")
    return score


if __name__ == "__main__":
    main(sys.argv[1], "--file" not in sys.argv, follow="--follow" in sys.argv)
