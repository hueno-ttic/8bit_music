"""変換パイプライン: ファイル読み込み → 解析 → NES 風シンセ → 書き出し."""
from __future__ import annotations

import logging
import tempfile

import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .analysis import SR, analyze
from .audio_io import load_mono, wav_to_mp3, write_wav
from .midi_io import midi_to_score, score_to_midi
from .synth import OUT_SR, render

log = logging.getLogger(__name__)


@dataclass
class ConvertOptions:
    mode: str = "auto"          # "auto" | "sep" | "ml" | "dsp"
    melody_duty: float = 0.5    # 0.125 / 0.25 / 0.5
    arpeggio: bool = True
    arp_speed: int = 1          # 1 = 16分, 2 = 32分
    drums: bool = True
    bass: bool = True
    crunch: bool = True
    fmt: str = "wav"            # "wav" | "mp3" | "midi" (採譜結果を MIDI で出す)
    # --- 分離モードの採譜パラメータ
    sensitivity: float = 0.5    # 歌の拾いやすさ 0.3〜0.7
    tempo_mult: str | int = "auto"  # "auto" | 1 | 2
    fill_gaps: bool = False     # 間奏をリード楽器で埋める (誤った音が出やすいので既定は OFF)
    key_snap: bool = True       # 調に合わない短い音を寄せる
    quantize: bool = True       # メロディを格子に量子化して滑らかに (楽譜らしく)
    source_url: str | None = None  # 元の YouTube URL (Songle / TextAlive の歌詞タイミング等に使う)
    trim_intro: bool = True     # 歌の無い長いイントロは出力から省く
    # --- 演奏
    echo: bool = False          # メロディのエコー (歌の細かい動きが二重に聞こえるので既定 OFF)
    melody_octave: int = 0      # メロディのオクターブ移動


def _pick_mode(mode: str) -> str:
    from . import analysis_ml, separation

    if mode == "dsp":
        return "dsp"
    if mode == "sep":
        if not separation.is_available():
            raise RuntimeError("ボーカル分離モード (demucs / torch) がインストールされていません")
        return "sep"
    if mode == "ml":
        if not analysis_ml.is_available():
            raise RuntimeError("高精度モード (basic-pitch) がインストールされていません")
        return "ml"
    # auto: 品質順に sep > ml > dsp
    if separation.is_available():
        return "sep"
    if analysis_ml.is_available():
        return "ml"
    return "dsp"


def verify_summary(vr: dict) -> str:
    """自己検証レポートを 1 行にする (UI とログ用)."""
    if vr.get("passed"):
        return (f"自己検証 OK: 修正 {vr['iterations']} 回 (上限 {vr['max_iter']}) で 外れ {vr['n_bad']} 音 "
                f"({vr['rate'] * 100:.2f}%, 目標 {vr['target'] * 100:.1f}% 以内) / 検証 {vr['n_checked']} 音")
    return (f"自己検証 NG: {vr['iterations']} 回修正しても目標に達しませんでした。外れ {vr['n_bad']} 音 "
            f"({vr['rate'] * 100:.2f}%, 目標 {vr['target'] * 100:.1f}% 以内) / 検証 {vr['n_checked']} 音")


LAST_VERIFY_REPORT: dict | None = None


def convert_file(src: str | Path, dst: str | Path, opts: ConvertOptions | None = None,
                 progress=None) -> Path:
    """src (mp4/mp3/wav など) を 8bit 化して dst に書き出す. 戻り値は実際の出力パス."""
    opts = opts or ConvertOptions()
    src, dst = Path(src), Path(dst)

    def report(msg: str):
        log.info(msg)
        if progress:
            progress(msg)

    if src.suffix.lower() in (".mid", ".midi"):
        # DAW などで直した MIDI をそのまま演奏する
        report("MIDI を読み込んでいます…")
        score = midi_to_score(src)
        mode = "midi"
    else:
        report("音声を取り出しています…")
        y = load_mono(src, SR)
        mode = _pick_mode(opts.mode)
        report(f"採譜しています… (mode={mode})")
    if mode == "midi":
        pass
    elif mode == "sep":
        from .analysis_sep import analyze_sep
        score = analyze_sep(y, SR, progress=report, sensitivity=opts.sensitivity, tempo_mult=opts.tempo_mult,
                            fill_gaps=opts.fill_gaps, key_snap=opts.key_snap, quantize=opts.quantize,
                            source_url=opts.source_url)
    elif mode == "ml":
        from .analysis_ml import analyze_ml
        score = analyze_ml(y, SR)
    else:
        score = analyze(y, SR)
    report(f"テンポ {score.tempo:.1f} BPM / メロディ {len(score.melody)} 音 / "
           f"ベース {len(score.bass)} 音 / ドラム {len(score.drums)} 打")
    vr = getattr(score, "verify", None)
    global LAST_VERIFY_REPORT
    LAST_VERIFY_REPORT = dict(vr) if vr else None
    if vr:
        report(verify_summary(vr))

    if opts.fmt == "midi":
        dst = dst.with_suffix(".mid")
        score_to_midi(score, dst)
        report("完了 (MIDI)")
        return dst

    report("ファミコン風に演奏しています…")
    trim_start = 0.0
    if opts.trim_intro and score.melody:
        first = min(n.start for n in score.melody)
        beat = 60.0 / score.tempo if score.tempo > 0 else 0.5
        if first > 4.0:
            trim_start = max(0.0, first - beat)
            report(f"歌の無いイントロ {trim_start:.1f} 秒を省きます")
    out = render(score, OUT_SR, melody_duty=opts.melody_duty, arpeggio=opts.arpeggio,
                 drums=opts.drums, bass=opts.bass, arp_speed=opts.arp_speed, crunch=opts.crunch,
                 echo=opts.echo, melody_octave=opts.melody_octave)

    if trim_start > 0:
        k = int(trim_start * OUT_SR)
        out = out[k:].copy()
        fade = min(len(out), int(0.05 * OUT_SR))
        out[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    if opts.fmt == "mp3":
        dst = dst.with_suffix(".mp3")
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "out.wav"
            write_wav(tmp, out, OUT_SR)
            wav_to_mp3(tmp, dst)
    else:
        dst = dst.with_suffix(".wav")
        write_wav(dst, out, OUT_SR)
    report("完了")
    return dst
