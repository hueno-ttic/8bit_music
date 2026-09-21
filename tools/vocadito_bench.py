"""Vocadito で 音符推定 (HMM 経路 vs 楽譜モデル) を mir_eval で比較. 歌詞・拍の情報なし (一様格子)."""
import sys, argparse
from pathlib import Path
import numpy as np, librosa, mir_eval
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chiptune import notes as N, score_model as SM
from chiptune.harmony import estimate_key
V = Path.home() / ".cache/8bit_music/datasets/vocadito"
F = Path.home() / ".cache/8bit_music/feat/vocadito"
HMM = dict(stay=0.998, sigma=0.8, min_dur=0.08, unv_stay=0.9)


def gt(i):
    a = np.loadtxt(V / f"Annotations/Notes/vocadito_{i}_notesA1.csv", delimiter=",", ndmin=2)
    return np.c_[a[:, 0], a[:, 0] + a[:, 2]], a[:, 1]


def to_arrays(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[s, e] for s, e, *_ in notes]); p = np.array([librosa.midi_to_hz(m) for _, _, m, *_ in notes])
    return iv, p


def key_of(f0):
    m = librosa.hz_to_midi(f0[f0 > 0]); h = np.bincount(np.round(m).astype(int) % 12, minlength=12).astype(float)
    return estimate_key(h)


def valleys_from_rms(t, rms, prominence_db=4.0, min_width_ms=30.0):
    from scipy.signal import find_peaks
    db = 20 * np.log10(rms + 1e-9)
    pk, _ = find_peaks(-db, prominence=prominence_db, width=min_width_ms / 10.0)
    return t[pk]


def run_sm(t, f0, step, rms=None, on=0.0, cross=0.0, prom=4.0, **kw):
    midi = librosa.hz_to_midi(np.maximum(f0, 1e-3)); voiced = f0 > 0
    lattice = np.arange(0, t[-1] + step, step)
    tonic, mode = key_of(f0)
    onset = cr = None
    if rms is not None and (on or cross):
        vs = valleys_from_rms(t, rms, prom)
        lattice = SM.warp_lattice(lattice, vs, max_shift=step / 2)
        onset = np.zeros(len(lattice) - 1); cr = np.zeros(len(lattice) + 1)
        for v in vs:
            i = int(np.argmin(np.abs(lattice[:-1] - v)))
            if abs(lattice[i] - v) <= step / 2:
                onset[i] = on; cr[i] = cross
    return SM.estimate_notes(t, midi, voiced, lattice, tonic=tonic, mode=mode, div=1, onset_prior=onset, cross_prior=cr, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=float, default=0.05); ap.add_argument("--w_score", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=0.8); ap.add_argument("--note_cost", type=float, default=-1.0); ap.add_argument("--rest_logp", type=float, default=-0.3)
    ap.add_argument("--hmm", action="store_true"); ap.add_argument("--on", type=float, default=0.0); ap.add_argument("--cross", type=float, default=0.0); ap.add_argument("--prom", type=float, default=4.0); a = ap.parse_args()
    res = {"hmm": [], "sm": []}
    for i in range(1, 41):
        d = np.load(F / f"vocadito_{i}.npz"); t, f0 = d["t"], d["f0"]
        ref_iv, ref_p = gt(i)
        outs = {}
        if a.hmm:
            h = N.hmm_segment(t, f0, **HMM); h = N.drop_register_outliers(h); outs["hmm"] = to_arrays(h)
        outs["sm"] = to_arrays(run_sm(t, f0, a.step, rms=d["rms"], on=a.on, cross=a.cross, prom=a.prom, w_score=a.w_score, gamma=a.gamma, note_cost=a.note_cost, rest_logp=a.rest_logp))
        for k, (iv, p) in outs.items():
            P, R, F1, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_iv, ref_p, iv, p, onset_tolerance=0.05, pitch_tolerance=50, offset_ratio=None)
            P2, R2, F2, _ = mir_eval.transcription.precision_recall_f1_overlap(ref_iv, ref_p, iv, p, onset_tolerance=0.05, pitch_tolerance=50, offset_ratio=0.2)
            res[k].append((F1, F2, len(iv) / max(1, len(ref_iv))))
    for k, v in res.items():
        if v:
            v = np.array(v); print(f"{k:4s} F(onset+pitch) {v[:,0].mean():.3f}  F(+offset) {v[:,1].mean():.3f}  音数比 {v[:,2].mean():.2f}")
