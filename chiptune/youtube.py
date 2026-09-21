"""yt-dlp で YouTube などの動画 URL から音声を取り出す."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

URL_RE = re.compile(r"^https?://", re.I)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s.strip()))


def _js_runtimes() -> dict:
    """yt-dlp が YouTube の難読化 JS を解くためのランタイム。deno が無ければ node を探す."""
    for name in ("deno", "node", "bun"):
        path = shutil.which(name) or next(
            (c for c in (f"/usr/local/bin/{name}", f"/opt/homebrew/bin/{name}") if Path(c).exists()), None)
        if path:
            return {name: {"path": path}}
    return {}


def _safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", title).strip(" ._")
    return name[:80] or "youtube"


def download_audio(url: str, out_dir: str | Path, progress=None) -> tuple[Path, str]:
    """URL の音声トラックだけをダウンロードし (ファイルパス, タイトル) を返す."""
    import imageio_ffmpeg
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if progress and d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            progress(f"YouTube からダウンロード中… {pct}")

    opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "ffmpeg_location": str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent),
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "max_filesize": 200 * 1024 * 1024,
    }
    rt = _js_runtimes()
    if rt:
        opts["js_runtimes"] = rt
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"ダウンロードに失敗しました: {e}") from e

    if info is None:
        raise RuntimeError("動画情報を取得できませんでした")
    if info.get("_type") == "playlist" and info.get("entries"):
        info = info["entries"][0]
    duration = info.get("duration") or 0
    if duration > 20 * 60:
        raise RuntimeError("20 分を超える動画には対応していません")

    files = sorted(out_dir.glob("source.*"))
    if not files:
        raise RuntimeError("音声ファイルが見つかりませんでした")
    return files[0], _safe_name(info.get("title") or "youtube")
