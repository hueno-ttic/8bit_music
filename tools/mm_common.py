"""マジカルミライ評価曲の共通処理: 曲の特徴量 (歌ステムの RMVPE f0, 音量) を永続キャッシュに置く."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path.home() / ".cache" / "8bit_music"
SONGS = CACHE / "testsongs"
FEAT = CACHE / "feat"
IDS = {"greenlights": "XSLhsjepelI", "aisarenakutemo": "ygY2qObZv24", "sand_planet": "AS4q9yaWJkI",
       "39music": "OuLZlZ18APQ", "hand_in_hand": "RKtoreimcQ8", "bless_mv": "a-Nf3QUFkOU",
       "future_eve": "7j0mQH0BtEU", "tenchi_kaibyaku": "8J6SMoVd5BY",
       # 調整に使っていない検証用 (held-out)
       "senbonzakura": "shs0rAiwsGQ", "shoushitsu": "VWVtIg5cdDU", "roki": "Xg-qfsKN2_E",
       "vampire": "e1xCOsgWG0M", "matryoshka": "HOz-9FzIDf0"}
HELDOUT = ["senbonzakura", "shoushitsu", "roki", "vampire", "matryoshka"]
# 音節オンセット検出器 (chiptune/onset_model) の学習だけに使う曲 (Songle に歌詞タイミングがあるもの)
TRAIN_EXTRA = {"hibana": "hxSg2Ioz3LM", "ghost_rule": "KushW6zvazM", "aikotoba3": "WptXk39wiIQ",
               "unknown_mothergoose": "P_CSdxSGfaA", "inochi": "0HYm60Mjm0k", "karakuri": "xxFkW3PCT5M",
               "donut_hole": "qnX2CdOBcDI", "kamippoina": "EHBFKhLUVig", "phony": "9QLT1Aw_45s",
               "king": "cm-l2h6GB8Q", "rettou": "oEkGC2HV7rc", "teo": "bmkY2yc1K7Q", "lower": "3sEptl-psU0",
               "tell_your_world": "PqJNc9KVIZE", "melty": "ZEy36W1xX8c", "shanti": "POy0RvJeaqM",
               "happy_synth": "xOKplMgHxxA"}
IDS.update(TRAIN_EXTRA)
SR = 22050
HOP_S = 0.01


def url_of(slug: str) -> str:
    return f"https://www.youtube.com/watch?v={IDS[slug]}"


def features(slug: str) -> dict:
    """{t, f0 (Hz, 音量ゲート後), rms, energy, midi, voiced, tonic, mode, stems 有無} を返す (キャッシュ)."""
    FEAT.mkdir(parents=True, exist_ok=True)
    fp = FEAT / f"{slug}.npz"
    if fp.exists():
        d = dict(np.load(fp))
        d["mode"] = str(d["mode"]); d["tonic"] = int(d["tonic"])
        return d
    import librosa
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    from chiptune.pitch import rmvpe_f0
    from chiptune.harmony import estimate_key
    y = load_mono(SONGS / f"{slug}.webm", sr=SR)
    y = y / (np.abs(y).max() + 1e-9)
    stems = separate(y, SR)
    voc, oth = stems["vocals"], stems["other"]
    t, f0 = rmvpe_f0(voc, SR, thred=0.03)
    rms = librosa.feature.rms(y=voc, frame_length=2048, hop_length=int(round(HOP_S * SR)))[0]
    n = min(len(f0), len(rms)); t, f0, rms = t[:n], f0[:n], rms[:n]
    ref = np.percentile(rms, 95) + 1e-9
    f0 = np.where(rms >= ref * 0.04, f0, 0.0)
    energy = np.clip(0.5 + 0.5 * rms / ref, 0.4, 1.0)
    chroma = librosa.feature.chroma_cqt(y=oth + voc, sr=SR, hop_length=512, n_chroma=12)
    tonic, mode = estimate_key(chroma.mean(axis=1))
    midi = librosa.hz_to_midi(np.maximum(f0, 1e-3))
    np.savez(fp, t=t, f0=f0, rms=rms, energy=energy, midi=midi, voiced=f0 > 0, tonic=tonic, mode=mode)
    return features(slug)


def songle_data(slug: str):
    from chiptune import songle as SG
    url = url_of(slug)
    return SG.beats(url), SG.lyrics_chars(url), SG.chords(url)


def cells16_from_beats(beats: list[dict], duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Songle の拍から 16 分セル境界と、各セルの小節内位置 (0..15) を作る."""
    bt = np.array([b["start"] / 1000.0 for b in beats]); pos = np.array([b["position"] for b in beats])
    cells, cpos = [], []
    # 先頭拍より前も拍間隔で埋める
    d0 = bt[1] - bt[0]
    k = 1
    while bt[0] - k * d0 > 0:
        k += 1
    pre = [bt[0] - j * d0 for j in range(k - 1, 0, -1)]
    pre_pos = [(pos[0] - j - 1) % 4 + 1 for j in range(k - 1, 0, -1)]
    allb = np.concatenate([pre, bt]); allp = np.concatenate([pre_pos, pos])
    for i in range(len(allb) - 1):
        for q in range(4):
            cells.append(allb[i] + (allb[i + 1] - allb[i]) * q / 4); cpos.append(((allp[i] - 1) % 4) * 4 + q)
    d = allb[-1] - allb[-2]
    tcur = allb[-1]; p = allp[-1]
    while tcur < duration:
        for q in range(4):
            cells.append(tcur + d * q / 4); cpos.append(((p - 1) % 4) * 4 + q)
        tcur += d; p = p % 4 + 1
    cells.append(tcur)
    return np.array(cells), np.array(cpos)


def fcpe_f0(slug: str) -> np.ndarray:
    """評価用の 2 つ目の f0 (FCPE, 10 ms, 無声=0)。RMVPE と一致するフレームだけを正解として信用するのに使う
    (2 つの推定器が 0.5 半音以内で一致しない所は、分離の残渣・オクターブ飛びの疑い)."""
    FEAT.mkdir(parents=True, exist_ok=True)
    fp = FEAT / f"{slug}_fcpe.npy"
    if fp.exists():
        return np.load(fp)
    import librosa
    import torch
    from torchfcpe import spawn_bundled_infer_model
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    y = load_mono(SONGS / f"{slug}.webm", sr=SR)
    y = y / (np.abs(y).max() + 1e-9)
    voc = separate(y, SR)["vocals"]
    x = librosa.resample(voc.astype(np.float32), orig_sr=SR, target_sr=16000)
    model = spawn_bundled_infer_model(device="cpu")
    n = len(features(slug)["t"])
    with torch.no_grad():
        f0 = model.infer(torch.from_numpy(x)[None, :, None], sr=16000, decoder_mode="local_argmax", threshold=0.006,
                         f0_min=60, f0_max=1400, interp_uv=False, output_interp_target_length=n)
    f0 = f0[0, :, 0].cpu().numpy().astype(np.float64)
    np.save(fp, f0)
    return f0
