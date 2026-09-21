"""Songle / TextAlive の音楽地図 (歌詞タイミング・拍・コード) を取得する。

Songle (https://songle.jp) は産総研の能動的音楽鑑賞サービスで、YouTube などの曲について
拍・コード・メロディ・サビの自動解析結果 (+ 利用者の修正) を公開している。
TextAlive の歌詞タイミング (文字ごとの発声区間) も Songle 側に保存されており、
  https://songle.jp/songs/{code}/lyrics/latest.json
で取れる。マジカルミライの曲は概ね登録済み。
取得結果は ~/.cache/8bit_music/songle/ にキャッシュする。
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("CHIPTUNE_MODEL_CACHE", Path.home() / ".cache" / "8bit_music")) / "songle"
UA = "8bit-music-converter/1.0 (+https://songle.jp)"
NAMES = "C C# D D# E F F# G G# A A# B".split()
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}


def _get(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    try:
        return json.loads(body)
    except Exception:
        return None


def _cached(key: str, fetch):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".json")
    if p.exists():
        return json.loads(p.read_text())
    data = fetch()
    if data is not None:
        p.write_text(json.dumps(data, ensure_ascii=False))
    return data


def normalize_url(url: str) -> str | None:
    """YouTube の URL を Songle が使う 'www.youtube.com/watch?v=ID' 形式に."""
    m = re.search(r"(?:youtu\.be/|v=|/shorts/)([A-Za-z0-9_-]{11})", url)
    if not m:
        return None
    return f"www.youtube.com/watch?v={m.group(1)}"


def song_info(url: str) -> dict | None:
    key = normalize_url(url)
    if not key:
        return None
    q = urllib.parse.quote(key, safe="")
    try:
        return _cached("song_" + key, lambda: _get(f"https://widget.songle.jp/api/v1/song.json?url={q}"))
    except Exception as e:  # noqa: BLE001
        log.info("Songle に登録がありません (%s): %s", key, e)
        return None


def beats(url: str) -> list[dict] | None:
    key = normalize_url(url)
    if not key:
        return None
    q = urllib.parse.quote(key, safe="")
    try:
        d = _cached("beat_" + key, lambda: _get(f"https://widget.songle.jp/api/v1/song/beat.json?url={q}"))
    except Exception:
        return None
    return d.get("beats") if isinstance(d, dict) else d


def chords(url: str) -> list[dict] | None:
    key = normalize_url(url)
    if not key:
        return None
    q = urllib.parse.quote(key, safe="")
    try:
        d = _cached("chord_" + key, lambda: _get(f"https://widget.songle.jp/api/v1/song/chord.json?url={q}"))
    except Exception:
        return None
    return d.get("chords") if isinstance(d, dict) else d


def lyrics_chars(url: str) -> list[tuple[float, float]] | None:
    """歌詞の文字ごとの (開始秒, 終了秒)。人手修正済みの最新版を返す。無ければ None."""
    info = song_info(url)
    if not info or not info.get("code"):
        return None
    code = info["code"]
    try:
        d = _cached("lyrics_" + code, lambda: _get(f"https://songle.jp/songs/{code}/lyrics/latest.json"))
    except Exception as e:  # noqa: BLE001
        log.info("歌詞タイミングがありません (%s): %s", code, e)
        return None
    if not isinstance(d, dict) or not d.get("data"):
        return None
    out = []
    for phrase in d["data"]:
        for word in phrase:
            for ch in word:
                s, e = float(ch["start_time"]), float(ch["end_time"])
                if e > s:
                    out.append((s, e))
    out.sort()
    return out or None


def parse_chord(name: str):
    """'C#m7', 'F#', 'G#m', 'E/G#', 'Bsus4', 'N' → (根音 pc, 'maj'|'min', 構成音 pcs) または None."""
    if not name or name == "N":
        return None
    name = name.split("/")[0]
    m = re.match(r"^([A-G])([#b]?)(.*)$", name)
    if not m:
        return None
    root = m.group(1) + m.group(2)
    root = FLAT.get(root, root)
    if root not in NAMES:
        return None
    r = NAMES.index(root)
    rest = m.group(3)
    minor = rest.startswith("m") and not rest.startswith("maj")
    if "dim" in rest:
        return (r, "min", [r, (r + 3) % 12, (r + 6) % 12])
    if "sus" in rest:
        return (r, "maj", [r, (r + 5 if "4" in rest else r + 2) % 12, (r + 7) % 12])
    if minor:
        return (r, "min", [r, (r + 3) % 12, (r + 7) % 12])
    return (r, "maj", [r, (r + 4) % 12, (r + 7) % 12])
