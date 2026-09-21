"""Score と標準 MIDI ファイル (SMF) の相互変換.

DAW で書き起こし結果を手直しして再レンダリングできるようにする.

トラック構成 (Type-1, PPQ=480):
  - "tempo"  : set_tempo / time_signature のみのコンダクタトラック
  - "melody" : ch0, program 80 (Square Lead)
  - "bass"   : ch1, program 38 (Synth Bass 1)
  - "chords" : ch2, program 81 (Saw Lead). 1 拍ごとのピッチクラスを 4 オクターブ (60+pc) の同時発音で書く
  - "drums"  : ch9, GM 準拠 (kick=36, snare=38, hat=42), 各ヒットは 16 分音符長

読み込みはこの逆. トラックはまず名前 (大小無視) で判別し, 名前が無ければ
ch9 → drums, それ以外は出現順に melody / bass / chords とみなす.
テンポ変化は最初の set_tempo だけを採用する (それ以降は無視).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import mido
import numpy as np

from .analysis import Hit, Note, Score

PPQ = 480
DEFAULT_BPM = 120.0

CH_MELODY, CH_BASS, CH_CHORDS, CH_DRUMS = 0, 1, 2, 9
PROGRAMS = {"melody": (CH_MELODY, 80), "bass": (CH_BASS, 38), "chords": (CH_CHORDS, 81)}
CHORD_OCTAVE_BASE = 60          # C4
CHORD_VELOCITY = 0.8
DRUM_NOTE = {"kick": 36, "snare": 38, "hat": 42}
DRUM_KIND = {35: "kick", 36: "kick", 38: "snare", 40: "snare", 42: "hat", 44: "hat", 46: "hat"}
TRACK_ROLES = ("melody", "bass", "chords", "drums")


# --------------------------------------------------------------------------- 書き出し

def _vel_to_midi(v: float) -> int:
    return int(min(127, max(1, round(v * 127))))


def _sorted_track(name: str, events: list[tuple[int, int, mido.Message]],
                  head: list[mido.MetaMessage] = ()) -> mido.MidiTrack:
    """(絶対 tick, 順序キー, Message) のリストを delta time のトラックに直す.

    同一 tick では note_off (順序キー 0) を note_on (1) より先に置く.
    """
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name, time=0), *head])
    last = 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1], e[2].note)):
        track.append(msg.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _note_events(notes: list[tuple[float, float, int, float]], channel: int,
                 sec_to_tick) -> list[tuple[int, int, mido.Message]]:
    """(start, end, midi, velocity) の列を note_on / note_off イベントに展開する."""
    ev = []
    for start, end, midi, vel in notes:
        s = sec_to_tick(start)
        e = max(s + 1, sec_to_tick(end))  # 長さ 0 の音は 1 tick にする
        ev.append((s, 1, mido.Message("note_on", channel=channel, note=int(midi), velocity=_vel_to_midi(vel))))
        ev.append((e, 0, mido.Message("note_off", channel=channel, note=int(midi), velocity=0)))
    return ev


def score_to_midi(score: Score, path: str) -> None:
    """Score を Type-1 の MIDI ファイルとして書き出す."""
    bpm = score.tempo if score.tempo and score.tempo > 0 else DEFAULT_BPM
    us_per_beat = mido.bpm2tempo(bpm)  # 整数 µs に丸めたテンポで tick を計算し, 読み込み側と一致させる
    sec_per_tick = us_per_beat / 1e6 / PPQ

    def sec_to_tick(t: float) -> int:
        return int(round(t / sec_per_tick))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    mid.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name="tempo", time=0),
        mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("end_of_track", time=0),
    ]))

    def program(role: str) -> list[mido.Message]:
        ch, prog = PROGRAMS[role]
        return [mido.Message("program_change", channel=ch, program=prog, time=0)]

    mid.tracks.append(_sorted_track("melody", _note_events(
        [(n.start, n.end, n.midi, n.velocity) for n in score.melody], CH_MELODY, sec_to_tick), program("melody")))
    mid.tracks.append(_sorted_track("bass", _note_events(
        [(n.start, n.end, n.midi, n.velocity) for n in score.bass], CH_BASS, sec_to_tick), program("bass")))
    chord_notes = [(s, e, CHORD_OCTAVE_BASE + pc, CHORD_VELOCITY)
                   for s, e, pcs in score.chords for pc in sorted(set(pcs))]
    mid.tracks.append(_sorted_track("chords", _note_events(chord_notes, CH_CHORDS, sec_to_tick), program("chords")))
    hit_len = PPQ // 4
    drum_ev = []
    for h in score.drums:
        if h.kind not in DRUM_NOTE:
            continue
        s = sec_to_tick(h.time)
        drum_ev.append((s, 1, mido.Message("note_on", channel=CH_DRUMS, note=DRUM_NOTE[h.kind],
                                           velocity=_vel_to_midi(h.velocity))))
        drum_ev.append((s + hit_len, 0, mido.Message("note_off", channel=CH_DRUMS, note=DRUM_NOTE[h.kind], velocity=0)))
    mid.tracks.append(_sorted_track("drums", drum_ev))
    mid.save(path)


# --------------------------------------------------------------------------- 読み込み

@dataclass
class _RawNote:
    start: float
    end: float
    midi: int
    velocity: float
    channel: int


def _track_notes(track: mido.MidiTrack, tick_to_sec) -> list[_RawNote]:
    """トラックの note_on / note_off を (秒単位の) ノート列にまとめる."""
    active: dict[tuple[int, int], tuple[int, int]] = {}
    notes: list[_RawNote] = []
    tick = 0

    def close(key: tuple[int, int], end_tick: int) -> None:
        s, v = active.pop(key)
        notes.append(_RawNote(tick_to_sec(s), tick_to_sec(max(end_tick, s + 1)), key[1], v / 127.0, key[0]))

    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            key = (msg.channel, msg.note)
            if key in active:  # 同じ音が重なったら前の音を閉じる
                close(key, tick)
            active[key] = (tick, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in active:
                close(key, tick)
    for key in list(active):
        close(key, tick)
    notes.sort(key=lambda n: (n.start, n.midi))
    return notes


def _assign_roles(mid: mido.MidiFile, tick_to_sec) -> dict[str, list[_RawNote]]:
    """各トラックを melody / bass / chords / drums に割り当てる."""
    roles: dict[str, list[_RawNote]] = defaultdict(list)
    unnamed: list[list[_RawNote]] = []
    for track in mid.tracks:
        notes = _track_notes(track, tick_to_sec)
        if not notes:
            continue
        name = next((m.name for m in track if m.type == "track_name"), "").strip().lower()
        role = next((r for r in TRACK_ROLES if r in name), None)
        if role is None and all(n.channel == CH_DRUMS for n in notes):
            role = "drums"
        if role is not None:
            roles[role].extend(notes)
        else:
            unnamed.append(notes)
    for role, notes in zip(("melody", "bass", "chords"), unnamed):
        roles[role].extend(notes)
    for notes in roles.values():
        notes.sort(key=lambda n: (n.start, n.midi))
    return roles


def midi_to_score(path: str, duration: float | None = None) -> Score:
    """MIDI ファイルを読み込んで Score に戻す.

    cells は 0 から曲末まで一様な 16 分音符グリッド. コードは 1 拍 (60/tempo 秒) ごとに,
    拍頭 (少し後ろに 1/8 拍のマージンを取る) で鳴っているノートのピッチクラス集合とする.
    duration を与えると, MIDI の最終イベントより長い場合にそちらを採用する.
    """
    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat or PPQ
    us_per_beat = next((m.tempo for tr in mid.tracks for m in tr if m.type == "set_tempo"),
                       mido.bpm2tempo(DEFAULT_BPM))
    bpm = float(mido.tempo2bpm(us_per_beat))
    sec_per_tick = us_per_beat / 1e6 / ppq

    def tick_to_sec(t: int) -> float:
        return t * sec_per_tick

    roles = _assign_roles(mid, tick_to_sec)
    melody = [Note(n.start, n.end, n.midi, n.velocity) for n in roles["melody"]]
    bass = [Note(n.start, n.end, n.midi, n.velocity) for n in roles["bass"]]
    drums = [Hit(n.start, DRUM_KIND[n.midi], n.velocity) for n in roles["drums"] if n.midi in DRUM_KIND]

    beat = 60.0 / bpm
    cell = beat / 4.0
    ends = [n.end for r in roles.values() for n in r]
    total = max(max(ends, default=0.0), duration or 0.0, cell)
    n_cells = math.ceil(total / cell - 1e-6)
    cells = np.arange(n_cells + 1, dtype=float) * cell
    cells[-1] = total  # analysis.py と同じく最後の境界は曲末に合わせる

    chord_notes = roles["chords"]
    chords: list[tuple[float, float, list[int]]] = []
    for k in range(math.ceil(total / beat - 1e-6)):
        t0, t1 = k * beat, min((k + 1) * beat, total)
        probe = t0 + beat / 8.0
        pcs = sorted({n.midi % 12 for n in chord_notes if n.start <= probe < n.end})
        chords.append((t0, t1, pcs))

    return Score(duration=total, tempo=bpm, cells=cells,
                 melody=melody, bass=bass, chords=chords, drums=drums)
