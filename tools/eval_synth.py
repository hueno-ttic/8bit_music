"""楽譜の分かっているボカロ曲 (tools/make_synth_gt.py) で、8bit に変換したメロディを楽譜と比べる.

  - 変換パイプラインで 8bit の WAV (全パート / メロディだけ) を書き出す
  - 正解の楽譜と、8bit の音符列を mir_eval の音符単位の指標で比べる (論文の標準)
      COnP   : 音の始まり (±50 ms) と高さ (±50 cent) が合った音符の F 値
      COnPOff: さらに終わり (長さの 20% か 50 ms) も合った音符の F 値
      COn    : 始まりだけ
  - 聴感に近い指標: 楽譜の音が鳴っている時間のうち、書き出した 8bit の WAV が正しい半音を鳴らしている割合 (TPA)、
    楽譜に音が無い時間に 8bit が鳴っている割合 (TFA)
使い方: .venv/bin/python tools/eval_synth.py vampire roki [--no-songle]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C  # noqa: E402
from eval_8bit import enable_fast_cache, track_rendered  # noqa: E402

GT = C.CACHE / "synth_gt"
HOP = 0.01
ORACLE_VOCAL = False
# 曲名照合の評価: 利用者が入力しそうな曲名と、手元のファイルの頭のずれ (秒, 正なら無音を足す / 負なら頭を切る)
TITLES = {"greenlights": ("グリーンライツ・セレナーデ Omoi", 2.5), "aisarenakutemo": ("愛されなくても君がいる ピノキオピー", -3.0),
          "sand_planet": ("砂の惑星 ハチ", 1.2), "39music": ("39みゅーじっく みきとP", -1.7), "hand_in_hand": ("Hand in Hand livetune", 3.1),
          "bless_mv": ("ブレス・ユア・ブレス 和田たけあき", -2.2), "senbonzakura": ("千本桜 黒うさP", 4.0),
          "shoushitsu": ("初音ミクの消失 cosMo", -2.6), "roki": ("ロキ みきとP", 1.9), "vampire": ("ヴァンパイア DECO*27", -5.3),
          "matryoshka": ("マトリョシカ ハチ", 2.8)}
TITLE_MATCH = False


STYLE = "follow"


def follow_notes(score):
    """歌に沿わせたメロディ (1/60 秒ごとの音高・音量) を音符に区切る: 半音が変わる所と、音量のくぼみからの立ち上がり (歌い直し)."""
    from chiptune.synth import NES_FRAME, follow_frames
    pitch, vol = follow_frames(score.melody, score.contour, score.duration)
    notes, cur = [], None
    for i in range(len(pitch)):
        p = pitch[i]
        reattack = i >= 2 and np.isfinite(p) and vol[i] - min(vol[i - 2:i]) >= 3
        if not np.isfinite(p) or cur is None or p != cur[2] or reattack:
            if cur is not None:
                notes.append((cur[0], i * NES_FRAME, cur[2]))
            cur = [i * NES_FRAME, None, p] if np.isfinite(p) else None
    if cur is not None:
        notes.append((cur[0], len(pitch) * NES_FRAME, cur[2]))
    return [(s, e, int(m)) for s, e, m in notes if e - s >= 0.02]


def note_metrics(ref, est) -> dict:
    import mir_eval
    if not len(est):
        return dict(COnP=0.0, COnPOff=0.0, COn=0.0)
    ri = np.array([[s, e] for s, e, _ in ref]); rp = np.array([440 * 2 ** ((m - 69) / 12) for _, _, m in ref])
    ei = np.array([[s, max(e, s + 0.011)] for s, e, _ in est]); ep = np.array([440 * 2 ** ((m - 69) / 12) for _, _, m in est])
    t = mir_eval.transcription
    _, _, fop, _ = t.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    _, _, fopo, _ = t.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=0.2)
    _, _, fo = t.onset_precision_recall_f1(ri, ei)
    return dict(COnP=fop, COnPOff=fopo, COn=fo)


def frame_metrics(ref, est_midi: np.ndarray) -> dict:
    n = len(est_midi)
    gt = np.full(n, np.nan)
    for s, e, m in ref:
        gt[int(round(s / HOP)):int(round(e / HOP))] = m
    on = np.isfinite(gt)
    ev = np.isfinite(est_midi)
    ok = on & ev & (np.abs(np.round(est_midi) - gt) < 0.5)
    lo, hi = int(ref[0][0] / HOP), int(ref[-1][1] / HOP)
    rng = np.zeros(n, bool); rng[lo:hi] = True
    return dict(TPA=float(ok[on].mean()), TFA=float((ev & ~on & rng).sum() / max(1, (~on & rng).sum())))


def run(slug: str, use_songle: bool, out: Path, fast: bool = False) -> dict:
    from chiptune.analysis import SR
    from chiptune.analysis_sep import analyze_sep
    from chiptune.audio_io import load_mono, write_wav
    from chiptune.separation import separate
    from chiptune.synth import OUT_SR, render
    ref = [tuple(x) for x in json.load(open(GT / f"{slug}.json"))["notes"]]
    y = load_mono(GT / f"{slug}.wav", SR)
    t0 = time.time()
    stems = separate(y / (np.abs(y).max() + 1e-9), SR)
    if ORACLE_VOCAL:
        # 分離が完璧だった場合の上限: 歌ステムを合成前のきれいな歌に差し替える
        v = load_mono(GT / f"{slug}_vocal.wav", SR)
        v = v / (np.abs(y).max() + 1e-9)
        stems = dict(stems); stems["vocals"] = np.pad(v, (0, max(0, len(y) - len(v))))[:len(y)]
    match_info = {}
    if TITLE_MATCH:
        # ファイル入力 + 曲名: 頭の長さを変えたファイルとして扱い、曲名で Songle と照合する
        from chiptune import songle_match
        title, sh = TITLES[slug]
        k = int(abs(sh) * SR)
        y = np.concatenate([np.zeros(k, np.float32), y]) if sh > 0 else y[k:]
        ref = [(s_ + sh, e_ + sh, m_) for s_, e_, m_ in ref if s_ + sh >= 0]
        stems = separate(y / (np.abs(y).max() + 1e-9), SR)
        m = songle_match.find(y, SR, title)
        match_info = dict(found=m is not None, offset_err=round(m.offset - sh, 3) if m else None)
        score = analyze_sep(y, SR, stems=stems, source_url=m.url if m else None, songle_offset=m.offset if m else 0.0)
        use_songle = False
    else:
        score = analyze_sep(y, SR, stems=stems, source_url=C.url_of(slug) if use_songle else None)
    tag = f"{slug}_{'title' if TITLE_MATCH else ('songle' if use_songle else 'file')}"
    out.mkdir(parents=True, exist_ok=True)
    est_notes = [(n.start, n.end, n.midi) for n in score.melody]
    if STYLE == "follow" and score.contour is not None:
        est_notes = follow_notes(score)
    if fast:
        # 開発用: WAV を作らず音符列 (歌に沿わせる方式なら 1/60 秒ごとの高さ) で測る
        fr = np.full(int(score.duration / HOP) + 10, np.nan)
        if STYLE == "follow" and score.contour is not None:
            from chiptune.synth import NES_FRAME, follow_frames
            fp, _ = follow_frames(score.melody, score.contour, score.duration)
            fr = fp[np.clip((np.arange(len(fr)) * HOP / NES_FRAME).astype(int), 0, len(fp) - 1)]
        else:
            for s_, e_, m_ in est_notes:
                fr[int(round(s_ / HOP)):int(round(e_ / HOP))] = m_
        res = dict(slug=slug, tag=tag, n_ref=len(ref), n_est=len(est_notes), sec=round(time.time() - t0, 1), **match_info)
        res.update({k: round(100 * v, 1) for k, v in note_metrics(ref, est_notes).items()})
        res.update({k: round(100 * v, 1) for k, v in frame_metrics(ref, fr).items()})
        json.dump(est_notes, open(out / f"{tag}_notes.json", "w"))
        return res
    write_wav(out / f"{tag}_8bit.wav", render(score, OUT_SR, melody_style=STYLE), OUT_SR)
    mel = render(score, OUT_SR, arpeggio=False, drums=False, bass=False, crunch=False, melody_style=STYLE)
    write_wav(out / f"{tag}_melody.wav", mel, OUT_SR)
    est_audio = track_rendered(mel, OUT_SR)
    res = dict(slug=slug, tag=tag, n_ref=len(ref), n_est=len(est_notes), sec=round(time.time() - t0, 1), **match_info)
    res.update({k: round(100 * v, 1) for k, v in note_metrics(ref, est_notes).items()})
    res.update({k: round(100 * v, 1) for k, v in frame_metrics(ref, est_audio).items()})
    json.dump(est_notes, open(out / f"{tag}_notes.json", "w"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--no-songle", action="store_true")
    ap.add_argument("--out", default=str(C.CACHE / "eval_out" / "synth"))
    ap.add_argument("--title-match", action="store_true", help="ファイル入力 + 曲名で Songle と照合する場合を測る (頭をずらしたファイル)")
    ap.add_argument("--oracle-vocal", action="store_true", help="開発用: 分離が完璧な場合 (きれいな歌) の上限を測る")
    ap.add_argument("--style", default="follow", choices=["follow", "notes"], help="メロディの鳴らし方")
    ap.add_argument("--fast", action="store_true", help="開発用: 重い解析をキャッシュし、WAV を作らず音符列で測る")
    ap.add_argument("--set", default="", help='開発用: chiptune.analysis_sep の属性を上書き (JSON)')
    a = ap.parse_args()
    if a.set:
        from chiptune import analysis_sep as AS
        for k, v in json.loads(a.set).items():
            if isinstance(v, dict) and isinstance(getattr(AS, k, None), dict):
                getattr(AS, k).update(v)
            else:
                setattr(AS, k, v)
    if a.fast:
        enable_fast_cache()
    global STYLE
    STYLE = a.style
    if a.title_match:
        global TITLE_MATCH
        TITLE_MATCH = True
    if a.oracle_vocal:
        global ORACLE_VOCAL
        ORACLE_VOCAL = True
    if a.slugs == ["all"]:
        a.slugs = ["greenlights", "aisarenakutemo", "sand_planet", "39music", "hand_in_hand", "bless_mv"] + C.HELDOUT
    rows = []
    for s in a.slugs:
        r = run(s, not a.no_songle, Path(a.out), fast=a.fast)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    keys = ["COnP", "COnPOff", "COn", "TPA", "TFA"]
    print("MEAN", json.dumps({k: round(float(np.mean([r[k] for r in rows])), 1) for k in keys}), flush=True)


if __name__ == "__main__":
    main()
