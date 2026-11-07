"""音節オンセット検出器 (chiptune/onset_model.py) を学習する.

教師: Songle / TextAlive の歌詞の文字タイミング (人手修正済み)。歌ステム (Demucs) から特徴量を作り、
文字の始まり (±20 ms をぼかしたターゲット) を 1 次元 CNN で当てる。損失は歌詞のある区間だけで取り、
信用できない区間 (単語内で機械的に等分された文字、漢字など長い文字の内側) は除く。
  学習曲: mm_common の開発 6 曲 + TRAIN_EXTRA (検証用の HELDOUT は使わない)
  早期終了と検出しきい値: 学習曲から外した VAL 曲で決める
使い方:
  .venv/bin/python tools/train_onset.py                       → chiptune/onset_model.pt を更新し、HELDOUT で評価
  .venv/bin/python tools/train_onset.py --exclude a,b --out X  → a, b を除いて学習 (交差検証用)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_common as C  # noqa: E402
from chiptune import onset_model as OM  # noqa: E402
from chiptune.score_model import merge_short_chars  # noqa: E402

DEV = ["greenlights", "aisarenakutemo", "sand_planet", "39music", "hand_in_hand", "bless_mv"]
VAL = ["teo", "lower"]
FEAT_DIR = C.CACHE / "onset_feat"


def song_features(slug: str) -> np.ndarray:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    fp = FEAT_DIR / f"{slug}.npy"
    if fp.exists():
        return np.load(fp)
    from chiptune.audio_io import load_mono
    from chiptune.separation import separate
    y = load_mono(C.SONGS / f"{slug}.webm", sr=C.SR)
    y = y / (np.abs(y).max() + 1e-9)
    voc = separate(y, C.SR)["vocals"]
    d = C.features(slug)
    X = OM.features(voc, C.SR, d["t"], d["f0"], d["rms"])
    np.save(fp, X)
    return X


def song_onsets(slug: str) -> np.ndarray:
    """歌われている (有声フレームが 4 つ以上ある) 文字の始まり."""
    d = C.features(slug)
    rv = d["voiced"]
    _, chars, _ = C.songle_data(slug)
    chars = merge_short_chars(chars)
    return np.array([s for s, e in chars if rv[int(s / OM.HOP_S):int(e / OM.HOP_S)].sum() >= 4])


def _lyric_words(slug: str):
    """Songle の歌詞データの単語 [[(start, end), ...], ...] (文字単位). 無ければ None."""
    from chiptune import songle as SG
    info = SG.song_info(C.url_of(slug))
    if not info or not info.get("code"):
        return None
    fp = SG.CACHE_DIR / f"lyrics_{info['code']}.json"
    if not fp.exists():
        SG.lyrics_chars(C.url_of(slug))
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    return [[(float(c["start_time"]), float(c["end_time"])) for c in w] for ph in d.get("data") or [] for w in ph]


def unreliable_spans(slug: str, long_char: float = 0.28):
    """教師として信用できない区間: 単語の中で機械的に等分された文字の境界 (人手で合わせたのは単語の境界だけ) と、
    長い文字の内側 (漢字 1 文字が 2〜3 音節にわたることがあり、中の音節の始まりが書かれていない)。"""
    words = _lyric_words(slug) or []
    spans = []
    for w in words:
        if len(w) >= 2 and np.std([e - s for s, e in w]) < 0.002:
            spans.append((w[0][0] + 0.04, w[-1][1] - 0.03))
        for s_, e_ in w:
            if e_ - s_ > long_char:
                spans.append((s_ + 0.06, e_ - 0.03))
    return [(a, b) for a, b in spans if b > a]


def song_coverage(slug: str, T: int, pad: float = 0.3, bridge: float = 1.0) -> np.ndarray:
    """歌詞データがある区間 (フレームの bool). Songle の歌詞は一部しか登録されていない曲があり
    (例: 3 分半の速い曲に 200〜300 文字)、歌詞の無い所の音節を「始まりではない」と教えると検出器が鈍るので、
    損失と評価は歌詞のある区間 (文字 ±pad 秒、bridge 秒未満の文字間) だけで取る."""
    _, chars, _ = C.songle_data(slug)
    chars = merge_short_chars(chars)
    cov = np.zeros(T, bool)
    prev_e = None
    for s, e in chars:
        a = s - (bridge if prev_e is not None and s - prev_e < bridge else pad)
        cov[max(0, int(a / OM.HOP_S)):min(T, int((e + pad) / OM.HOP_S))] = True
        prev_e = e
    for a, b in unreliable_spans(slug):
        cov[max(0, int(a / OM.HOP_S)):min(T, int(b / OM.HOP_S))] = False
    return cov


def _target(onsets: np.ndarray, T: int, cov: np.ndarray):
    y = np.zeros(T, np.float32)
    for s in onsets:
        i = int(round(s / OM.HOP_S))
        for k, w in ((0, 1.0), (-1, 0.6), (1, 0.6), (-2, 0.25), (2, 0.25)):
            if 0 <= i + k < T:
                y[i + k] = max(y[i + k], w)
    return y, cov.copy()


AUG_DIR = C.CACHE / "onset_aug"


def load(slugs):
    """slugs の要素は曲名、または "曲名:rep" / "曲名:var" (tools/make_onset_aug.py で作った歌い直し版)."""
    out = []
    for s in slugs:
        if ":" in s:
            base, kind = s.split(":")
            X = np.load(AUG_DIR / f"{base}_{kind}.npy")
            on = song_onsets(base)
            cov = song_coverage(base, X.shape[1])
            on = on[cov[np.clip((on / OM.HOP_S).astype(int), 0, len(cov) - 1)]]
            y, m = _target(on, X.shape[1], cov)
            out.append((s, X, y, m, on))
            continue
        X = song_features(s)
        on = song_onsets(s)
        cov = song_coverage(s, X.shape[1])
        on = on[cov[np.clip((on / OM.HOP_S).astype(int), 0, len(cov) - 1)]]
        y, m = _target(on, X.shape[1], cov)
        out.append((s, X, y, m, on))
    return out


def prf(det, ref, tol=0.05, cov=None):
    """1 対 1 対応での適合率・再現率・F 値 (歌詞のある区間 cov の検出だけを数える)."""
    det = np.asarray(det)
    if cov is not None:
        det = det[cov[np.clip((det / OM.HOP_S).astype(int), 0, len(cov) - 1)]] if len(det) else det
    else:
        det = det[(det >= ref.min() - 0.5) & (det <= ref.max() + 1.0)]
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        if not len(det):
            break
        dd = np.abs(det - r)
        dd[used] = 9
        k = int(np.argmin(dd))
        if dd[k] <= tol:
            used[k] = True
            tp += 1
    p = tp / max(1, len(det))
    r_ = tp / max(1, len(ref))
    return p, r_, 2 * p * r_ / (p + r_ + 1e-9)


def evaluate(net, data, thr, tol=0.05):
    return np.array([prf(OM.pick_peaks(OM.posterior(X, net), thr), on, tol, m) for _, X, _, m, on in data])


def _augment(x, yy, mm, rng):
    L = x.shape[1]
    x = x.copy()
    x[:OM.N_MELS] += rng.normal() * 0.3                                          # 音量
    x[:OM.N_MELS] = np.roll(x[:OM.N_MELS], rng.integers(-2, 3), axis=0)         # 声質 (フォルマント)
    x[OM.N_MELS] += rng.normal() * 0.1 * (x[OM.N_MELS + 1] > 0)                 # 音域
    x[:OM.N_MELS] += rng.normal(size=x[:OM.N_MELS].shape) * 0.1
    r = rng.uniform(0.88, 1.12)                                                  # テンポ ±12%
    idx = np.clip((np.arange(int(L / r)) * r).astype(int), 0, L - 1)[:L]
    if len(idx) < L:
        idx = np.concatenate([idx, np.full(L - len(idx), L - 1)])
    x, mm = x[:, idx], mm[idx]
    pos = np.searchsorted(idx, np.where(yy >= 1.0)[0])
    y2 = np.zeros(L, np.float32)
    for q in pos:
        for k, w in ((0, 1.0), (-1, 0.6), (1, 0.6), (-2, 0.25), (2, 0.25)):
            if 0 <= q + k < L:
                y2[q + k] = max(y2[q + k], w)
    return x, y2, mm


THRS = np.arange(0.05, 0.75, 0.05)


def train(data, val, *, ch=32, drop=0.3, epochs=60, seed=0, gru=0, log=print):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = OM._build_net(ch, drop, gru)
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(4.0))
    best = (-1.0, None, 0, 0.5)
    for ep in range(epochs):
        net.train()
        for _ in range(60):
            xb, yb, mb = [], [], []
            for _ in range(16):
                _, X, y, m, _ = data[rng.integers(len(data))]
                L = 800
                a = int(rng.integers(0, X.shape[1] - L))
                x, yy, mm = _augment(X[:, a:a + L], y[a:a + L], m[a:a + L], rng)
                xb.append(x); yb.append(yy); mb.append(mm)
            xb, yb, mb = (torch.from_numpy(np.stack(z)) for z in (xb, yb, mb))
            loss = (bce(net(xb), yb) * mb).sum() / (mb.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 2 == 1:
            net.eval()
            scores = {th: evaluate(net, val, th)[:, 2].mean() for th in THRS}
            th = max(scores, key=scores.get)
            if scores[th] > best[0]:
                best = (scores[th], copy.deepcopy(net.state_dict()), ep, float(th))
            log(f"  epoch {ep}: val F {scores[th]:.3f} (thr {th:.2f}) best {best[0]:.3f}@{best[2]}")
    net.load_state_dict(best[1])
    net.eval()
    return net, best[3], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="学習から除く曲 (カンマ区切り, 交差検証用)")
    ap.add_argument("--out", default=str(OM.WEIGHTS))
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gru", type=int, default=0, help="双方向 GRU の隠れ次元 (0 なら畳み込みだけ)")
    ap.add_argument("--aug", action="store_true", help="tools/make_onset_aug.py の歌い直し版も学習に使う")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    ex = set(filter(None, a.exclude.split(",")))
    train_songs = [s for s in DEV + list(C.TRAIN_EXTRA) if s not in ex and s not in VAL and (C.SONGS / f"{s}.webm").exists()]
    if a.aug:
        # 歌い直し版 (同じ高さの文字を増やした版など)。VAL と除外曲の版は使わない
        train_songs += [f"{s}:{k}" for s in C.TRAIN_EXTRA for k in ("rep", "var")
                        if s not in ex and s not in VAL and (AUG_DIR / f"{s}_{k}.npy").exists()]
    print("train:", len(train_songs), "songs; val:", VAL, flush=True)
    tr, val = load(train_songs), load(VAL)
    net, thr, fval = train(tr, val, ch=a.ch, drop=a.drop, epochs=a.epochs, seed=a.seed, gru=a.gru, log=lambda m: print(m, flush=True))
    out = Path(a.out)
    torch.save(net.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps({"ch": a.ch, "gru": a.gru, "thr": thr, "val_f": round(fval, 4),
                                                    "train": train_songs, "val": VAL, "excluded": sorted(ex)}, ensure_ascii=False, indent=1))
    held = load(C.HELDOUT)
    for tol in (0.05, 0.07):
        r = evaluate(net, held, thr, tol)
        print(f"HELDOUT tol {int(tol * 1000)}ms: P {r[:, 0].mean():.3f} R {r[:, 1].mean():.3f} F {r[:, 2].mean():.3f}  "
              + " ".join(f"{s}:{x:.2f}" for s, x in zip(C.HELDOUT, r[:, 2])), flush=True)


if __name__ == "__main__":
    main()
