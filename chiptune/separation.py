"""Demucs (Meta の音源分離モデル) で曲を vocals / drums / bass / other に分ける."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

import librosa
import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "htdemucs"
CACHE_DIR = Path(os.environ.get("CHIPTUNE_STEM_CACHE", Path.home() / ".cache" / "8bit_music" / "stems"))
_MODEL = None
_LOCK = threading.Lock()


def is_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from demucs.pretrained import get_model

            log.info("Demucs モデル %s を読み込み中…", MODEL_NAME)
            _MODEL = get_model(MODEL_NAME)
            _MODEL.eval()
    return _MODEL


def _cache_key(y: np.ndarray, sr: int) -> str:
    h = hashlib.sha1()
    h.update(f"{MODEL_NAME}:{sr}:{len(y)}".encode())
    h.update(np.ascontiguousarray(y[::97]).tobytes())
    return h.hexdigest()[:20]


def separate(y: np.ndarray, sr: int, progress=None, use_cache: bool = True) -> dict[str, np.ndarray]:
    """モノラル y (sr Hz) を分離し、各ステムをモノラル・同じ sr で返す."""
    import torch
    from demucs.apply import apply_model

    cache_file = CACHE_DIR / f"{_cache_key(y, sr)}.npz"
    if use_cache and cache_file.exists():
        try:
            data = np.load(cache_file)
            stems = {k: data[k] for k in data.files}
            if all(len(v) == len(y) for v in stems.values()):
                log.info("分離結果をキャッシュから読み込みました: %s", cache_file)
                return stems
        except Exception as e:  # noqa: BLE001
            log.warning("キャッシュの読み込みに失敗: %s", e)

    model = _load_model()
    msr = model.samplerate
    x = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=msr) if sr != msr else y.astype(np.float32)
    # (batch, channels, samples)。モデルはステレオ前提なので同じ信号を 2ch に
    wav = torch.from_numpy(np.stack([x, x]))[None]
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    device = _device()
    if progress:
        progress(f"ボーカル / ドラム / ベースを分離しています… ({device}, 1 分ほど)")
    with torch.no_grad():
        try:
            out = apply_model(model, wav, device=device, shifts=0, split=True, overlap=0.25, progress=False)
        except Exception as e:  # noqa: BLE001  (MPS で失敗したら CPU に落とす)
            if device == "cpu":
                raise
            log.warning("%s で分離に失敗したため CPU で再試行します: %s", device, e)
            out = apply_model(model, wav, device="cpu", shifts=0, split=True, overlap=0.25, progress=False)
    out = out[0] * (ref.std() + 1e-8) + ref.mean()  # (sources, ch, samples)

    stems: dict[str, np.ndarray] = {}
    for name, src in zip(model.sources, out):
        m = src.mean(0).cpu().numpy().astype(np.float32)
        if msr != sr:
            m = librosa.resample(m, orig_sr=msr, target_sr=sr)
        stems[name] = m[: len(y)] if len(m) >= len(y) else np.pad(m, (0, len(y) - len(m)))

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, **stems)
        except Exception as e:  # noqa: BLE001
            log.warning("キャッシュの保存に失敗: %s", e)
    return stems
