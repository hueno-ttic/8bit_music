"""ffmpeg (imageio-ffmpeg 同梱バイナリ) を使った音声の読み書き."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load_mono(path: str | Path, sr: int = 22050) -> np.ndarray:
    """任意の動画/音声ファイルをモノラル float32 配列として読み込む."""
    cmd = [
        ffmpeg_exe(),
        "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg で音声を取り出せませんでした: " + proc.stderr.decode(errors="replace")
        )
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError("音声トラックが見つかりませんでした")
    return y.copy()


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


def wav_to_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "192k") -> None:
    cmd = [
        ffmpeg_exe(),
        "-v", "error", "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("mp3 変換に失敗しました: " + proc.stderr.decode(errors="replace"))
