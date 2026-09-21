"""歌の f0 推定: RMVPE (混合音・歌ステムのどちらにも強い) を使い、無ければ CREPE / pyin に落ちる."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import librosa
import numpy as np

log = logging.getLogger(__name__)

RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
CACHE_DIR = Path(os.environ.get("CHIPTUNE_MODEL_CACHE", Path.home() / ".cache" / "8bit_music"))
HOP_S = 0.01  # RMVPE は 16 kHz / hop 160 = 10 ms
_MODEL = None


def rmvpe_weights() -> Path:
    p = CACHE_DIR / "rmvpe.pt"
    if not p.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("RMVPE の重み (約 180MB) をダウンロードしています…")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(RMVPE_URL, tmp)
        tmp.rename(p)
    return p


def is_rmvpe_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    global _MODEL
    if _MODEL is None:
        from .rmvpe_model import RMVPE

        # CPU の方が MPS より速い (畳み込みが小さく転送が支配的) ので CPU 固定
        _MODEL = RMVPE(str(rmvpe_weights()), is_half=False, device="cpu")
    return _MODEL


def rmvpe_f0(y: np.ndarray, sr: int, thred: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """(times [s], f0 [Hz], 無声=0) を 10 ms 間隔で返す."""
    x = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32)
    f0 = _load().infer_from_audio(x, thred=thred)
    f0 = np.asarray(f0, dtype=np.float64)
    return np.arange(len(f0)) * HOP_S, f0
