"""実際に 8bit 曲を作って、メロディが元の歌をどれだけ再現しているかを測る.

  1. 曲を変換パイプライン (analyze_sep) に通し、8bit の WAV (全パート) とメロディだけの WAV を書き出す
  2. メロディだけの WAV (矩形波) の音高を pyin で追跡する (= 実際に鳴った 8bit メロディ)
  3. 元の曲の歌ステムの f0 (RMVPE) と 10 ms ごとに比べる (mir_eval の melody 指標に準拠)
     RPA  : 歌っているフレームのうち、8bit の音が歌の音高 ±0.5 半音以内の割合 (= 正しい音が鳴っている)
     RCA  : 同 (オクターブ違いも正解扱い)
     VR   : 歌っているフレームのうち、8bit のメロディが鳴っている割合
     VFA  : 歌っていないフレームのうち、8bit のメロディが鳴ってしまっている割合
     NA   : 8bit が鳴っていて歌もあるフレームのうち、音高が合っている割合 (鳴った音の正確さ)
     OA   : 全フレームの総合正解率
  4. 歌詞の文字タイミング (Songle/TextAlive, 人手修正済み) がある曲は、文字を正解のリズムとして
     音の頭の再現率/適合率 (onR/onP/onF) と、文字ごとの音高の正解率 (mPA/mPAc/mPAv) も出す。
     audio 側は、音の頭も音高も 8bit の WAV から検出した値で測る (鳴り始め・音高の変化・音量のくぼみ / pyin)
使い方: .venv/bin/python tools/eval_8bit.py greenlights [--no-songle] [--out DIR]
        .venv/bin/python tools/eval_8bit.py --heldout          (調整に使っていない検証曲)
        .venv/bin/python tools/eval_8bit.py --fast --no-songle (開発用の高速評価: WAV を作らない)
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

HOP_S = 0.01
EVAL_SR = 16000


def track_rendered(wav: np.ndarray, sr: int) -> np.ndarray:
    """8bit メロディ (矩形波, 単音) の音高を 10 ms ごとに MIDI で返す (鳴っていない = nan)."""
    import librosa
    x = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=EVAL_SR)
    hop = int(EVAL_SR * HOP_S)
    f0, voiced, _ = librosa.pyin(x, fmin=55.0, fmax=2100.0, sr=EVAL_SR, frame_length=1024, hop_length=hop,
                                 center=True, fill_na=np.nan, max_transition_rate=100.0)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]
    on = voiced & (rms > 0.02 * (np.percentile(rms[rms > 0], 95) if np.any(rms > 0) else 1.0))
    return np.where(on, librosa.hz_to_midi(np.where(on, f0, 1.0)), np.nan)


def rendered_onsets(wav: np.ndarray, sr: int, est_midi: np.ndarray) -> np.ndarray:
    """8bit メロディの音の頭 (秒): 鳴り始め・音高の変化・音量のくぼみからの立ち上がり (同音の歌い直し)."""
    import librosa
    hop = int(sr * HOP_S)
    db = 20 * np.log10(librosa.feature.rms(y=wav, frame_length=hop * 2, hop_length=hop)[0] + 1e-6)
    n = min(len(db), len(est_midi))
    db, m = db[:n], est_midi[:n]
    on = np.isfinite(m)
    out = []
    for i in range(3, n - 3):
        if not on[i]:
            continue
        if not on[i - 1] and not on[i - 2]:
            out.append(i)
        elif on[i - 1] and abs(np.nanmedian(m[i:i + 3]) - np.nanmedian(m[i - 3:i])) >= 0.7 and abs(m[i] - m[i - 1]) >= 0.5:
            out.append(i)
        elif db[i] - db[i - 3:i].min() >= 4.0 and db[i - 1] - db[i - 3:i].min() < 4.0:
            out.append(i)
    t = np.array(out) * HOP_S
    if len(t):
        keep = [t[0]]
        for x in t[1:]:
            if x - keep[-1] >= 0.04:
                keep.append(x)
        t = np.array(keep)
    return t


def mora_metrics(note_list, onsets, chars, ref_midi, ref_voiced, tol: float = 0.07, ref2_midi=None,
                 est_frames=None) -> dict:
    """歌詞の文字 (Songle/TextAlive の人手修正済みタイミング) を正解のリズムとして使う.
    onR : 歌われた文字の始まりのうち、8bit の音の頭が ±tol 秒以内にある割合 (リズムの再現)
    onP : 8bit の音の頭のうち、文字の始まりに対応するものの割合 (余計な音の切れ目が少ない)
    mPA : 文字ごとの音高 (文字の中央 50% の歌の f0 を丸めた値) と、そこで鳴っている 8bit の音が一致する割合
    mPAc: 同じく、音高がはっきりした文字 (安定していて半音の中央 ±0.3 以内) だけでの一致率 (曖昧さの無い誤り)
    mPAv: mPAc のうち、2 つ目の f0 推定器 (FCPE) も同じ半音と判定した文字だけでの一致率 (正解の f0 自体の誤りを除く)
    short: 70 ms 未満の音の割合 / repBad: 同じ高さで続く音の切れ目のうち、文字の始まりに無い (根拠の無い) ものの割合
    """
    from chiptune.score_model import merge_short_chars
    chars = merge_short_chars(chars)
    n = len(ref_midi)
    # est_frames: 8bit の WAV から検出した音高 (10 ms). 無ければ音符列から
    est = notes_to_frames(note_list, n) if est_frames is None else np.pad(est_frames[:n], (0, max(0, n - len(est_frames))),
                                                                          constant_values=np.nan)
    if est_frames is not None:
        est = np.round(est)   # 矩形波の検出値 (ビブラート込み) を半音に
    sung, pitch_ok, pitch_n, clear_ok, clear_n, ver_ok, ver_n = [], 0, 0, 0, 0, 0, 0
    for s_, e_ in chars:
        a, b = int(s_ / HOP_S), int(e_ / HOP_S)
        if b - a < 4 or b > n or ref_voiced[a:b].sum() < 4:
            continue
        sung.append(s_)
        q0, q1 = a + (b - a) // 4, b - (b - a) // 4
        v = ref_voiced[q0:q1] & np.isfinite(est[q0:q1])
        if v.sum() < 2:
            continue
        pitch_n += 1
        rv_ = ref_midi[q0:q1][v]
        rm = float(np.median(rv_))
        rp = int(np.round(rm))
        ep = float(np.median(est[q0:q1][v]))
        pitch_ok += int(abs(ep - rp) < 0.5)
        # はっきりした音 (安定していて半音の中央に近い): 正解が曖昧でないので、外れは確実な誤り
        if abs(rm - rp) <= 0.3 and np.percentile(rv_, 90) - np.percentile(rv_, 10) <= 0.8:
            clear_n += 1
            clear_ok += int(abs(ep - rp) < 0.5)
            # 2 つ目の f0 推定器 (FCPE) も同じ半音と言っている音節 = 正解の f0 自体が確かな音節
            if ref2_midi is not None:
                r2 = ref2_midi[q0:q1]
                r2 = r2[np.isfinite(r2)]
                if len(r2) >= 0.6 * (q1 - q0) and int(np.round(np.median(r2))) == rp:
                    ver_n += 1
                    ver_ok += int(abs(ep - rp) < 0.5)
    sung = np.array(sung)
    on = np.asarray(onsets, dtype=float)
    lo, hi = (sung.min() - 0.5, sung.max() + 1.0) if len(sung) else (0, 0)
    on_in = on[(on >= lo) & (on <= hi)]
    rec = float(np.mean([np.min(np.abs(on - c)) <= tol for c in sung])) if len(sung) and len(on) else 0.0
    prec = float(np.mean([np.min(np.abs(sung - o)) <= tol for o in on_in])) if len(on_in) and len(sung) else 0.0
    f = 2 * rec * prec / (rec + prec + 1e-9)
    # 細切れ (ブツブツ途切れて聞こえる) の指標: 70 ms 未満の音の割合と、同じ高さの音の切れ目のうち文字の始まりに無いものの割合
    nl = sorted(note_list, key=lambda x: x.start)
    short = float(np.mean([x.end - x.start < 0.07 for x in nl])) if nl else 0.0
    rep = [b.start for a, b in zip(nl[:-1], nl[1:]) if a.midi == b.midi and b.start - a.end < 0.02]
    rep_bad = float(np.mean([np.min(np.abs(sung - r)) > tol for r in rep])) if rep and len(sung) else 0.0
    res = dict(onR=rec, onP=prec, onF=f, mPA=pitch_ok / max(1, pitch_n), mPAc=clear_ok / max(1, clear_n),
               short=short, repBad=rep_bad)
    if ref2_midi is not None:
        res["mPAv"] = ver_ok / max(1, ver_n)
        res["nV"] = ver_n / 100.0   # (表示が 100 倍されるので)
    return res


def _ref2(slug: str):
    """FCPE の f0 を MIDI に (無声 = nan). torchfcpe が無ければ None."""
    try:
        f = C.fcpe_f0(slug)
    except Exception:
        return None
    return np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0), np.nan)


def notes_to_frames(notes, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    for nt in notes:
        a, b = int(round(nt.start / HOP_S)), int(round(nt.end / HOP_S))
        out[max(0, a):min(n, b)] = nt.midi
    return out


def melody_metrics(ref_midi: np.ndarray, ref_voiced: np.ndarray, est: np.ndarray) -> dict:
    n = min(len(ref_midi), len(est))
    r, rv, e = ref_midi[:n], ref_voiced[:n], est[:n]
    ev = ~np.isnan(e)
    d = np.where(rv & ev, np.abs(e - r), np.inf)
    dc = np.where(rv & ev, np.abs(((e - r) + 6) % 12 - 6), np.inf)
    ok = d <= 0.5
    okc = dc <= 0.5
    both = rv & ev
    return dict(
        RPA=float(ok[rv].mean()) if rv.any() else 0.0,
        RCA=float(okc[rv].mean()) if rv.any() else 0.0,
        VR=float(ev[rv].mean()) if rv.any() else 0.0,
        VFA=float(ev[~rv].mean()) if (~rv).any() else 0.0,
        NA=float(ok[both].mean()) if both.any() else 0.0,
        NA1=float((d[both] <= 1.0).mean()) if both.any() else 0.0,
        OCT=float(((dc <= 0.5) & (d > 0.5))[both].mean()) if both.any() else 0.0,
        OA=float((ok | (~rv & ~ev)).mean()),
    )


def enable_fast_cache():
    """開発用: 重い解析 (RMVPE・pyin・chroma・ビート検出) の結果を入力のハッシュでディスクにキャッシュする.
    採譜の後段 (音符化) を何度も試すときに使う。結果は通常実行と同じ."""
    import functools
    import hashlib
    import pickle
    import librosa
    from chiptune import pitch as P

    cdir = C.CACHE / "fastcache"
    cdir.mkdir(parents=True, exist_ok=True)

    def memo(fn, name):
        @functools.wraps(fn)
        def wrap(*args, **kw):
            h = hashlib.sha1(name.encode())
            for a in list(args) + sorted(kw.items()):
                if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
                    h.update(a[0].encode()); a = a[1]
                h.update(np.ascontiguousarray(a).tobytes() if isinstance(a, np.ndarray) else repr(a).encode())
            fp = cdir / f"{name}_{h.hexdigest()}.pkl"
            if fp.exists():
                return pickle.loads(fp.read_bytes())
            r = fn(*args, **kw)
            fp.write_bytes(pickle.dumps(r))
            return r
        return wrap

    P.rmvpe_f0 = memo(P.rmvpe_f0, "rmvpe")
    librosa.pyin = memo(librosa.pyin, "pyin")
    librosa.feature.chroma_cqt = memo(librosa.feature.chroma_cqt, "chroma")
    librosa.beat.beat_track = memo(librosa.beat.beat_track, "beat")


FOLD_DIR: Path | None = None


def _use_fold_model(slug: str):
    """交差検証: その曲を学習に使っていない音節オンセット検出器 (FOLD_DIR/*.pt, *.json の exclude) を使う."""
    from chiptune import onset_model as OM
    for js in sorted(FOLD_DIR.glob("*.json")):
        meta = json.loads(js.read_text())
        if slug in meta.get("excluded", []):
            OM.WEIGHTS = js.with_suffix(".pt"); OM._NET = None
            return
    raise RuntimeError(f"{slug} を除いた検出器がありません")


def run(slug: str, use_songle: bool, out_dir: Path, mode: str = "sep", fast: bool = False) -> dict:
    import librosa
    from chiptune.analysis import SR
    from chiptune.analysis_sep import analyze_sep
    from chiptune.audio_io import load_mono, write_wav
    from chiptune.separation import separate
    from chiptune.synth import OUT_SR, render

    y = load_mono(C.SONGS / f"{slug}.webm", SR)
    if FOLD_DIR is not None:
        _use_fold_model(slug)
    t0 = time.time()
    if mode == "sep":
        yn = y / (np.abs(y).max() + 1e-9)
        stems = separate(yn, SR)
        score = analyze_sep(y, SR, stems=stems, source_url=C.url_of(slug) if use_songle else None)
    elif mode == "ml":
        from chiptune.analysis_ml import analyze_ml
        score = analyze_ml(y, SR)
    else:
        from chiptune.analysis import analyze
        score = analyze(y, SR)
    t_an = time.time() - t0
    tag = f"{slug}_{'songle' if use_songle else 'file'}" + ("" if mode == "sep" else f"_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = C.features(slug)
    ref_midi, ref_voiced = d["midi"], d["voiced"].astype(bool)
    if fast:
        # 音符列だけで評価 (WAV は作らない)
        m_notes = melody_metrics(ref_midi, ref_voiced, notes_to_frames(score.melody, len(ref_midi)))
        _, chars, _ = C.songle_data(slug)
        if chars:
            m_notes.update(mora_metrics(score.melody, [nt.start for nt in score.melody], chars, ref_midi, ref_voiced,
                                        ref2_midi=_ref2(slug)))
        np.save(out_dir / f"{tag}_notes.npy", np.array([(nt.start, nt.end, nt.midi) for nt in score.melody]))
        return dict(slug=slug, tag=tag, n_notes=len(score.melody), tempo=round(score.tempo, 1), sec=round(t_an, 1),
                    notes={k: round(v * 100, 1) for k, v in m_notes.items()})
    full = render(score, OUT_SR)
    write_wav(out_dir / f"{tag}_8bit.wav", full, OUT_SR)
    mel = render(score, OUT_SR, arpeggio=False, drums=False, bass=False, crunch=False)
    write_wav(out_dir / f"{tag}_melody.wav", mel, OUT_SR)

    est_audio = track_rendered(mel, OUT_SR)
    est_notes = notes_to_frames(score.melody, len(ref_midi))
    m_audio = melody_metrics(ref_midi, ref_voiced, est_audio)
    m_notes = melody_metrics(ref_midi, ref_voiced, est_notes)
    _, chars, _ = C.songle_data(slug)
    if chars:
        on_audio = rendered_onsets(mel, OUT_SR, est_audio)
        m_audio.update(mora_metrics(score.melody, on_audio, chars, ref_midi, ref_voiced, ref2_midi=_ref2(slug),
                                    est_frames=est_audio))
        m_notes.update(mora_metrics(score.melody, [nt.start for nt in score.melody], chars, ref_midi, ref_voiced,
                                    ref2_midi=_ref2(slug)))
    res = dict(slug=slug, tag=tag, n_notes=len(score.melody), tempo=round(score.tempo, 1), sec=round(t_an, 1),
               audio={k: round(v * 100, 1) for k, v in m_audio.items()},
               notes={k: round(v * 100, 1) for k, v in m_notes.items()})
    np.savez(out_dir / f"{tag}_frames.npz", ref=ref_midi, ref_voiced=ref_voiced, est_audio=est_audio, est_notes=est_notes,
             notes=np.array([(nt.start, nt.end, nt.midi) for nt in score.melody]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", default=["greenlights", "aisarenakutemo", "sand_planet", "39music",
                                                  "hand_in_hand", "bless_mv", "future_eve"])
    ap.add_argument("--no-songle", action="store_true", help="ファイル入力相当 (Songle の情報を使わない)")
    ap.add_argument("--mode", default="sep")
    ap.add_argument("--out", default=str(Path.home() / ".cache" / "8bit_music" / "eval_out"))
    ap.add_argument("--fast", action="store_true", help="開発用: 重い解析をキャッシュし、WAV を作らず音符列だけで評価")
    ap.add_argument("--heldout", action="store_true", help="検証用の曲 (調整に使っていない曲) で評価")
    ap.add_argument("--set", default="", help='開発用: chiptune.analysis_sep の属性を上書き (JSON, 例 {"SM_PARAMS": {"in_key_w": 2}})')
    ap.add_argument("--folds", default="", help="開発用: 交差検証した音節検出器のディレクトリ")
    a = ap.parse_args()
    if a.folds:
        global FOLD_DIR
        FOLD_DIR = Path(a.folds)
    if a.set:
        from chiptune import analysis_sep as AS
        for k, v in json.loads(a.set).items():
            if isinstance(v, dict) and isinstance(getattr(AS, k, None), dict):
                getattr(AS, k).update(v)
            else:
                setattr(AS, k, v)
    if a.heldout:
        a.slugs = C.HELDOUT
    if a.fast:
        enable_fast_cache()
    rows = []
    for s in a.slugs:
        r = run(s, not a.no_songle, Path(a.out), a.mode, fast=a.fast)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    keys = ["RPA", "RCA", "VR", "VFA", "NA", "OA", "onR", "onP", "onF", "mPA", "mPAc", "mPAv", "short", "repBad"]
    for src in ("audio", "notes"):
        if src not in rows[0]:
            continue
        avg = {k: round(float(np.mean([r[src][k] for r in rows if k in r[src]])), 1) for k in keys}
        print(f"MEAN[{src}]", json.dumps(avg), flush=True)


if __name__ == "__main__":
    main()
