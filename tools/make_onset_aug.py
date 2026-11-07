"""音節オンセット検出器の学習データを増やす: 学習曲の歌を WORLD で歌い直させた版を作り、特徴量を保存する.
  - "rep": 文字の半分を直前の文字と同じ高さで歌わせる (同じ高さの歌い直し = 子音・声色の変化だけが手掛かりの例を増やす)
  - "var": 元の高さのまま、ポルタメント・しゃくり・ビブラートを付け直す
歌詞のタイミングは元のままなので、正解の文字の始まりは Songle の文字タイミングがそのまま使える。
評価に使う曲 (開発 6 曲・HELDOUT) は使わない。出力: ~/.cache/8bit_music/onset_aug/{slug}_{rep|var}.npy (特徴量)
使い方: .venv/bin/python tools/make_onset_aug.py"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C
import make_synth_gt as G

AUG = C.CACHE / "onset_aug"


def featurize(path: Path, out: Path):
    import librosa
    from chiptune import onset_model as OM
    from chiptune.audio_io import load_mono
    from chiptune.pitch import rmvpe_f0
    from chiptune import separation as S
    S.VOCAL_MODEL = "demucs"
    y = load_mono(path, C.SR)
    voc = S.separate(y / (np.abs(y).max() + 1e-9), C.SR)["vocals"]
    t, f0 = rmvpe_f0(voc, C.SR, thred=0.03)
    rms = librosa.feature.rms(y=voc, frame_length=2048, hop_length=int(round(0.01 * C.SR)))[0]
    n = min(len(f0), len(rms)); t, f0, rms = t[:n], f0[:n], rms[:n]
    f0 = np.where(rms >= np.percentile(rms, 95) * 0.04, f0, 0.0)
    np.save(out, OM.features(voc, C.SR, t, f0, rms))


if __name__ == "__main__":
    AUG.mkdir(parents=True, exist_ok=True)
    for slug in C.TRAIN_EXTRA:
        for kind, rp, seed in (("rep", 0.5, 1), ("var", 0.0, 2)):
            tag = f"{slug}_{kind}"
            if (AUG / f"{tag}.npy").exists():
                continue
            G.make(slug, seed=seed, repeat_prob=rp, out_dir=AUG, tag=tag)
            featurize(AUG / f"{tag}.wav", AUG / f"{tag}.npy")
            (AUG / f"{tag}.wav").unlink(); (AUG / f"{tag}_vocal.wav").unlink()
            print(tag, "ok", flush=True)
