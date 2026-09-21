"""Vocadito の各曲の RMVPE f0 を永続キャッシュへ (楽譜モデルの評価用)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chiptune.audio_io import load_mono
from chiptune.pitch import rmvpe_f0
import librosa
V = Path.home() / ".cache/8bit_music/datasets/vocadito"
OUT = Path.home() / ".cache/8bit_music/feat/vocadito"; OUT.mkdir(parents=True, exist_ok=True)
for wav in sorted(V.glob("Audio/vocadito_*.wav")):
    fp = OUT / (wav.stem + ".npz")
    if fp.exists():
        continue
    y = load_mono(wav, sr=22050); y = y / (np.abs(y).max() + 1e-9)
    t, f0 = rmvpe_f0(y, 22050, thred=0.03)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=int(round(0.01 * 22050)))[0]
    n = min(len(f0), len(rms)); t, f0, rms = t[:n], f0[:n], rms[:n]
    ref = np.percentile(rms, 95) + 1e-9
    f0 = np.where(rms >= ref * 0.04, f0, 0.0)
    np.savez(fp, t=t, f0=f0, rms=rms)
    print(wav.stem, len(t))
