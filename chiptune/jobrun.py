"""Web アプリの変換ジョブを別プロセスで、時間の上限付きで実行する.

- 1 曲を最長 TOTAL_LIMIT 秒 (3 分弱) で返す。変換 (YouTube のダウンロード込み) は別プロセスで動かし、子プロセスには
  締め切り (chiptune.budget) を渡して、時間が足りなければ重い任意の処理を省かせる
- それでも MAIN_LIMIT 秒を過ぎたら子プロセスを止め、取得済みの音声から速いモード (basic-pitch, 3 分の曲で 15 秒ほど) で作り直す
- 子プロセスが落ちても (GPU を使う処理が異常終了しても) サーバーは止まらない
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as _queue
import time
from dataclasses import replace
from pathlib import Path

log = logging.getLogger(__name__)

TOTAL_LIMIT = float(os.environ.get("CHIPTUNE_TOTAL_LIMIT", 175.0))   # 1 曲の処理の上限 (秒)
MAIN_LIMIT = float(os.environ.get("CHIPTUNE_MAIN_LIMIT", 145.0))     # ここまでに終わらなければ速いモードで作り直す
FALLBACK_MIN = 20.0   # 作り直しに残す最低の秒数


def _worker(q, kind: str, payload: str, work: str, stem: str, opts, deadline: float):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from . import budget
        from .convert import convert_file
        from . import convert as cv
        budget.set_deadline(deadline)

        def progress(msg: str):
            q.put(("progress", msg))

        workp = Path(work)
        if kind == "url":
            from .youtube import download_audio
            q.put(("progress", "YouTube からダウンロード中…"))
            src, stem = download_audio(payload, workp, progress)
            q.put(("downloaded", str(src), stem))
        else:
            src = Path(payload)
        out = convert_file(src, workp / f"{stem}_8bit.{opts.fmt}", opts, progress)
        q.put(("done", str(out), cv.LAST_VERIFY_REPORT))
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("conversion failed")
        q.put(("error", str(e)))


def _run_child(kind, payload, work, stem, opts, deadline, limit, progress, state):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, kind, payload, str(work), stem, opts, deadline), daemon=True)
    p.start()
    end = time.time() + limit
    result = None
    try:
        while time.time() < end:
            try:
                msg = q.get(timeout=0.5)
            except _queue.Empty:
                if not p.is_alive():
                    # 終了直前に送られたものを取り切る
                    try:
                        msg = q.get(timeout=1.0)
                    except _queue.Empty:
                        result = ("crashed",)
                        break
                else:
                    continue
            if msg[0] == "progress":
                progress(msg[1])
            elif msg[0] == "downloaded":
                state["src"], state["stem"] = msg[1], msg[2]
            else:
                result = msg
                break
        else:
            result = ("timeout",)
    finally:
        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
    return result


def run(kind: str, payload: str, work: Path, stem: str, opts, progress) -> tuple[Path, dict | None, str]:
    """kind: "url" (payload = YouTube の URL) / "file" (payload = 音声ファイルのパス).
    戻り値: (出力ファイル, 自己検証レポート, 曲名)。時間内に作れなければ RuntimeError."""
    start = time.time()
    state = {"src": payload if kind == "file" else None, "stem": stem}
    res = _run_child(kind, payload, work, stem, opts, start + MAIN_LIMIT - 5, MAIN_LIMIT, progress, state)
    if res[0] == "done":
        return Path(res[1]), res[2], state["stem"]
    if res[0] == "error":
        raise RuntimeError(res[1])
    # 時間切れ / 子プロセスの異常終了: 取得済みの音声から速いモードで作り直す
    reason = "時間内に終わらない" if res[0] == "timeout" else "処理が異常終了した"
    left = TOTAL_LIMIT - (time.time() - start)
    if not state["src"] or not Path(state["src"]).exists():
        raise RuntimeError(f"{reason}ため中止しました (YouTube からのダウンロードが終わっていません)")
    if left < FALLBACK_MIN:
        raise RuntimeError(f"{reason}ため中止しました")
    from . import analysis_ml
    mode = "ml" if analysis_ml.is_available() else "dsp"
    log.warning("%s ため、速いモード (%s) で作り直します (残り %.0f 秒)", reason, mode, left)
    progress(f"{reason}ので、速いモードで作り直しています…")
    fb_opts = replace(opts, mode=mode, song_title=None)
    res = _run_child("file", state["src"], work, state["stem"], fb_opts, time.time() + left - 3, left, progress, state)
    if res[0] == "done":
        return Path(res[1]), res[2], state["stem"]
    raise RuntimeError(f"{reason}ため中止しました" + (f": {res[1]}" if res[0] == "error" else ""))
