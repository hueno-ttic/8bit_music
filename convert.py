#!/usr/bin/env python
"""コマンドライン版:  python convert.py <input.mp4 | YouTube URL> [-o out.wav] [--mp3] [--mode dsp|ml|auto] ..."""
import argparse
import logging
import tempfile
from pathlib import Path

from chiptune import ConvertOptions, convert_file
from chiptune.youtube import download_audio, is_url


def main():
    ap = argparse.ArgumentParser(description="mp4 などの曲を 8bit チップチューンに変換します")
    ap.add_argument("input", help="mp4 などのファイル、または YouTube の URL")
    ap.add_argument("-o", "--output", help="出力ファイル (既定: <入力名>_8bit.wav)")
    ap.add_argument("--mp3", action="store_true", help="mp3 で出力")
    ap.add_argument("--midi", action="store_true", help="採譜結果を MIDI で出力 (DAW で直して .mid を入力に戻せる)")
    ap.add_argument("--mode", default="auto", choices=["auto", "sep", "ml", "dsp"],
                    help="sep=ボーカル分離 (高品質・遅い), ml=basic-pitch, dsp=軽量, auto=使える中で最良")
    ap.add_argument("--duty", type=float, default=0.5, choices=[0.125, 0.25, 0.5], help="メロディの矩形波デューティ比")
    ap.add_argument("--no-arp", action="store_true", help="コードのアルペジオを鳴らさない")
    ap.add_argument("--arp-speed", type=int, default=1, choices=[1, 2])
    ap.add_argument("--no-drums", action="store_true")
    ap.add_argument("--no-bass", action="store_true")
    ap.add_argument("--no-crunch", action="store_true", help="出力の 8bit 量子化をしない")
    ap.add_argument("--sensitivity", type=float, default=0.5, help="歌の拾いやすさ 0.3〜0.7 (分離モード)")
    ap.add_argument("--tempo-mult", default="auto", choices=["auto", "1", "2"], help="テンポを倍にするか (分離モード)")
    ap.add_argument("--fill", action="store_true", help="間奏をリード楽器で埋める (実験的)")
    ap.add_argument("--no-key-snap", action="store_true", help="調への補正をしない")
    ap.add_argument("--no-quantize", action="store_true", help="メロディを格子に整えない (歌の揺れをそのまま)")
    ap.add_argument("--keep-intro", action="store_true", help="歌の無いイントロも省かずに出力する")
    ap.add_argument("--echo", action="store_true", help="メロディにエコーを付ける")
    ap.add_argument("--octave", type=int, default=0, help="メロディのオクターブ移動 (-1, 0, 1)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    tmp = None
    source_url = a.input if is_url(a.input) else None
    if is_url(a.input):
        tmp = tempfile.TemporaryDirectory()
        src, title = download_audio(a.input, tmp.name, progress=print)
        dst = Path(a.output) if a.output else Path.cwd() / f"{title}_8bit.wav"
    else:
        src = Path(a.input)
        dst = Path(a.output) if a.output else src.with_name(src.stem + "_8bit.wav")
    opts = ConvertOptions(mode=a.mode, melody_duty=a.duty, arpeggio=not a.no_arp,
                          arp_speed=a.arp_speed, drums=not a.no_drums, bass=not a.no_bass,
                          crunch=not a.no_crunch, fmt="midi" if a.midi else ("mp3" if a.mp3 else "wav"),
                          sensitivity=a.sensitivity, tempo_mult=a.tempo_mult if a.tempo_mult == "auto" else int(a.tempo_mult),
                          fill_gaps=a.fill, key_snap=not a.no_key_snap, quantize=not a.no_quantize, source_url=source_url, trim_intro=not a.keep_intro, echo=a.echo, melody_octave=a.octave)
    try:
        out = convert_file(src, dst, opts, progress=print)
    finally:
        if tmp:
            tmp.cleanup()
    print(f"→ {out}")


if __name__ == "__main__":
    main()
