"""楽譜モデル推定器 (chiptune.score_model) をマジカルミライ曲で評価し、r29 (歌詞境界方式) と比較する.
使い方: .venv/bin/python tools/eval_sm.py greenlights [--w_score 2.5 --gamma 0.6 ...]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import os; os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C
import mm_metrics as M
from chiptune import score_model as SM
from chiptune.harmony import scale_pcs


def baseline_notes(slug: str, chars, cells16):
    fp = C.FEAT / f"{slug}_r29.json"
    if fp.exists():
        return json.load(open(fp))
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    from chiptune.analysis_sep import _melody_from_lyrics, _subdivide
    y = load_mono(C.SONGS / f"{slug}.webm", sr=C.SR); y = y / (np.abs(y).max() + 1e-9)
    voc = separate(y, C.SR)["vocals"]
    d = C.features(slug)
    out, _ = _melody_from_lyrics(voc, C.SR, chars, _subdivide(cells16, 2), key=(d["tonic"], d["mode"]))
    notes = [(float(s), float(e), int(m)) for s, e, m, _, _ in out]
    json.dump(notes, open(fp, "w"))
    return notes


def lyric_onset_prior(lattice, chars, on=2.0, inside=-2.0, cross=-2.5):
    """歌詞の文字境界 → 格子点ごとの (音の始まりの事前, またぎ越しの事前)."""
    lp = np.zeros(len(lattice) - 1); cr = np.zeros(len(lattice) + 1)
    for s_, e_ in chars:
        i = int(np.argmin(np.abs(lattice[:-1] - s_)))
        j0, j1 = np.searchsorted(lattice[:-1], s_ + 0.03), np.searchsorted(lattice[:-1], e_ - 0.03)
        lp[j0:j1] = np.minimum(lp[j0:j1], inside)
        lp[i] = on
        if abs(lattice[i] - s_) <= 0.04:
            cr[i] = cross
    return lp, cr


def run(slug: str, params: dict, verbose=True):
    d = C.features(slug)
    beats, chars, sg_chords = C.songle_data(slug)
    dur = float(d["t"][-1])
    cells16, cpos = C.cells16_from_beats(beats, dur)
    div = params.pop("div", 2)
    lattice = SM.make_lattice(cells16, div)
    chars = SM.merge_short_chars(chars) if chars else chars
    ms = params.pop("max_shift", None)
    if chars:
        lattice = SM.warp_lattice(lattice, [c[0] for c in chars], max_shift=ms)
    beat_pos = np.repeat(cpos, div)[: len(lattice) - 1] * div + np.tile(np.arange(div), len(cpos))[: len(lattice) - 1]
    on, inside, cross = params.pop("on", 2.0), params.pop("inside", -2.0), params.pop("cross", -2.5)
    onset, cr = lyric_onset_prior(lattice, chars, on, inside, cross) if chars else (None, None)
    chord_pcs = None
    if params.get("w_chord", 0) > 0 and sg_chords:
        from chiptune import songle as SG
        parsed = [(c["start"] / 1000.0, (c["start"] + c["duration"]) / 1000.0, SG.parse_chord(c.get("name"))) for c in sg_chords]
        chord_pcs = SM.chord_pcs_for_lattice(lattice, parsed)
    params["chord_pcs"] = chord_pcs
    t0 = time.time()
    notes = SM.estimate_notes(d["t"], d["midi"], d["voiced"], lattice, tonic=d["tonic"], mode=d["mode"],
                              onset_prior=onset, beat_pos=beat_pos, div=div, cross_prior=cr, **params)
    el = time.time() - t0
    sp = scale_pcs(d["tonic"], d["mode"])
    m = M.evaluate(notes, chars, d["midi"], d["voiced"], sp)
    if verbose:
        print(f"[score_model {el:.0f}s] {M.fmt(m)}")
    return notes, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slug"); ap.add_argument("--w_score", type=float, default=3.0); ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--on", type=float, default=2.0); ap.add_argument("--inside", type=float, default=-3.0)
    ap.add_argument("--rest_logp", type=float, default=-0.3); ap.add_argument("--trans_frames", type=int, default=4)
    ap.add_argument("--div", type=int, default=2); ap.add_argument("--cross", type=float, default=-6.0); ap.add_argument("--in_key_w", type=float, default=10.0); ap.add_argument("--max_shift", type=float, default=None); ap.add_argument("--rest_voiced", type=float, default=-3.0); ap.add_argument("--w_chord", type=float, default=0.0); ap.add_argument("--no_baseline", action="store_true")
    a = ap.parse_args()
    d = C.features(a.slug); beats, chars, _ = C.songle_data(a.slug)
    sp = scale_pcs(d["tonic"], d["mode"])
    if not a.no_baseline:
        cells16, _ = C.cells16_from_beats(beats, float(d["t"][-1]))
        base = baseline_notes(a.slug, chars, cells16)
        print(f"[r29       ] {M.fmt(M.evaluate(base, chars, d['midi'], d['voiced'], sp))}")
    run(a.slug, dict(w_score=a.w_score, gamma=a.gamma, on=a.on, inside=a.inside, rest_logp=a.rest_logp, trans_frames=a.trans_frames, div=a.div, cross=a.cross, in_key_w=a.in_key_w, w_chord=a.w_chord, max_shift=a.max_shift, rest_voiced=a.rest_voiced))
