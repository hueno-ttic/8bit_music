"""統合版 (_melody_from_score_model: 楽譜モデル + 後処理 + フレーズ接続) を 6 曲で r29 と比較."""
import sys, json, logging
from pathlib import Path
import os; os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C, mm_metrics as M
from chiptune.analysis_sep import score_model_notes
from chiptune.harmony import scale_pcs
from chiptune import songle as SG
logging.basicConfig(level=logging.INFO, format="%(message)s")
import os
PARAMS = json.loads(os.environ.get("SM_PARAMS", "{}"))
songs = sys.argv[1:] or ["greenlights", "aisarenakutemo", "sand_planet", "39music", "hand_in_hand", "bless_mv"]
for slug in songs:
    d = C.features(slug); beats, chars, sgc = C.songle_data(slug)
    parsed = [(c["start"] / 1000.0, (c["start"] + c["duration"]) / 1000.0, SG.parse_chord(c.get("name"))) for c in (sgc or [])]
    out = score_model_notes(d["t"], d["f0"], d["energy"], chars, beats, (d["tonic"], d["mode"]), chords=parsed, params=PARAMS)
    notes = [(s, e, m) for s, e, m, _, _ in out]
    json.dump(notes, open(C.FEAT / f"{slug}_int.json", "w"))
    sp = scale_pcs(d["tonic"], d["mode"])
    print(f"{slug:15s} [統合版 {PARAMS}] {M.fmt(M.evaluate(notes, chars, d['midi'], d['voiced'], sp))}", flush=True)
    fp = C.FEAT / f"{slug}_r29.json"
    if fp.exists():
        print(f"{slug:15s} [r29   ] {M.fmt(M.evaluate(json.load(open(fp)), chars, d['midi'], d['voiced'], sp))}", flush=True)
