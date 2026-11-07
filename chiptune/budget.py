"""変換の時間予算 (締め切り).

Web アプリは 1 曲を最長 3 分で返す。締め切りが近いときは、重い任意の処理 (ハモリの上の声の選び直し、曲名での照合)
を省き、歌の分離が間に合わない見込みなら速いモード (basic-pitch) に切り替える。CLI では締め切りを設定しない (無制限)。
"""
from __future__ import annotations

import time

_DEADLINE: float | None = None


def set_deadline(t: float | None) -> None:
    """締め切り (time.time() の値)。None なら無制限."""
    global _DEADLINE
    _DEADLINE = t


def remaining() -> float:
    """締め切りまでの残り秒数 (無制限なら inf)."""
    return float("inf") if _DEADLINE is None else _DEADLINE - time.time()


def enough(seconds: float) -> bool:
    """残りが seconds 秒以上あるか."""
    return remaining() >= seconds
