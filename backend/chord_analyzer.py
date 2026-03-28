import os
import logging
import numpy as np
import librosa

# Suppress verbose basic-pitch debug output
logging.getLogger("basic_pitch").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from basic_pitch.inference import predict as bp_predict


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ── Krumhansl-Schmuckler key profiles ───────────────────────────────────────
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(y: np.ndarray, sr: int) -> tuple[str, str]:
    """Detect musical key using Krumhansl-Schmuckler algorithm.
    Returns (root, quality) e.g. ('G', 'minor')."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)

    best_r = -np.inf
    best_root = 'C'
    best_quality = 'major'

    for i, note in enumerate(NOTE_NAMES):
        for profile, quality in [(_KS_MAJOR, 'major'), (_KS_MINOR, 'minor')]:
            rolled = np.roll(profile, i)
            r = float(np.corrcoef(rolled, chroma_sum)[0, 1])
            if r > best_r:
                best_r = r
                best_root = note
                best_quality = quality

    return best_root, best_quality


# Root, major third (+4), perfect fifth (+7)
_MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
# Root, minor third (+3), perfect fifth (+7)
_MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

_TEMPLATES: list[np.ndarray] = []
_LABELS: list[tuple[str, str]] = []

for _i, _note in enumerate(NOTE_NAMES):
    for _template, _quality in [(_MAJOR, "major"), (_MINOR, "minor")]:
        t = np.roll(_template, _i)
        t = t / np.linalg.norm(t)
        _TEMPLATES.append(t)
        _LABELS.append((_note, _quality))

TEMPLATE_MATRIX = np.array(_TEMPLATES)  # (24, 12)

CONFIDENCE_THRESHOLD = 0.65
ENERGY_MIN = 0.1   # minimum total note-overlap-seconds per beat window
HOP_LENGTH = 512
SR = 22050
POLYPHONY_WARNING_THRESHOLD = 6

# ---------------------------------------------------------------------------
# MIDI → ABC notation pitch map (MIDI 36–91)
# ---------------------------------------------------------------------------
MIDI_TO_ABC: dict[int, str] = {
    # Octave 2: C2–B2
    36: "C,,", 37: "^C,,", 38: "D,,", 39: "^D,,", 40: "E,,", 41: "F,,",
    42: "^F,,", 43: "G,,", 44: "^G,,", 45: "A,,", 46: "^A,,", 47: "B,,",
    # Octave 3: C3–B3
    48: "C,", 49: "^C,", 50: "D,", 51: "^D,", 52: "E,", 53: "F,",
    54: "^F,", 55: "G,", 56: "^G,", 57: "A,", 58: "^A,", 59: "B,",
    # Octave 4: C4–B4
    60: "C", 61: "^C", 62: "D", 63: "^D", 64: "E", 65: "F",
    66: "^F", 67: "G", 68: "^G", 69: "A", 70: "^A", 71: "B",
    # Octave 5: C5–B5
    72: "c", 73: "^c", 74: "d", 75: "^d", 76: "e", 77: "f",
    78: "^f", 79: "g", 80: "^g", 81: "a", 82: "^a", 83: "b",
    # Octave 6: C6–G6
    84: "c'", 85: "^c'", 86: "d'", 87: "^d'", 88: "e'", 89: "f'",
    90: "^f'", 91: "g'",
}


def analyze_chords(audio_path: str) -> dict:
    """
    Analyze chord progressions using Spotify basic-pitch for note detection.

    Returns:
    {
        "chords": [{"time", "end", "chord", "quality", "confidence"}, ...],
        "multi_instrument_warning": bool,
        "max_polyphony": int,
        "bpm": float,
        "beat_times": list[float],
        "note_events": [{"start", "end", "pitch"}, ...],
    }
    """
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # librosa.beat_track often detects 2× actual tempo for slow/ballad songs.
    # Heuristic: if bpm > 130 and halving gives a natural range (55–110), use the half.
    if bpm > 130:
        bpm_half = bpm / 2
        if 55 <= bpm_half <= 110:
            bpm = bpm_half

    key_root, key_quality = detect_key(y, sr)

    if len(beat_frames) == 0:
        return {
            "chords": [{"time": 0.0, "end": duration, "chord": "N", "quality": "", "confidence": 0.0}],
            "multi_instrument_warning": False,
            "max_polyphony": 0,
            "bpm": bpm,
            "key_root": key_root,
            "key_quality": key_quality,
            "beat_times": [],
            "note_events": [],
        }

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    n_beats = len(beat_times)

    # basic-pitch note detection: (start_s, end_s, pitch_midi, velocity, pitch_bend)
    _, _, raw_notes = bp_predict(audio_path)

    max_polyphony = _compute_max_polyphony(raw_notes)

    # Build pitch-class histograms per beat window
    histograms = np.zeros((n_beats, 12), dtype=float)
    window_weights = np.zeros(n_beats, dtype=float)

    for note in raw_notes:
        note_start, note_end, pitch_midi = float(note[0]), float(note[1]), int(note[2])
        pc = pitch_midi % 12
        for i in range(n_beats):
            win_start = beat_times[i]
            win_end = float(beat_times[i + 1]) if i + 1 < n_beats else duration
            overlap = max(0.0, min(note_end, win_end) - max(note_start, win_start))
            if overlap > 0:
                weight = overlap / max(win_end - win_start, 1e-8)
                histograms[i, pc] += weight
                window_weights[i] += weight

    # Template matching per beat
    events = []
    for i in range(n_beats):
        t = float(beat_times[i])
        end = float(beat_times[i + 1]) if i + 1 < n_beats else duration

        if window_weights[i] < ENERGY_MIN:
            events.append({"time": t, "end": end, "chord": "N", "quality": "", "confidence": 0.0})
            continue

        vec = histograms[i]
        norm = np.linalg.norm(vec) + 1e-8
        scores = TEMPLATE_MATRIX @ (vec / norm)
        best = int(np.argmax(scores))
        conf = float(scores[best])

        if conf < CONFIDENCE_THRESHOLD:
            events.append({"time": t, "end": end, "chord": "N", "quality": "", "confidence": 0.0})
        else:
            root, quality = _LABELS[best]
            events.append({
                "time": t, "end": end,
                "chord": root, "quality": quality,
                "confidence": round(conf, 3),
            })

    chords_out = _merge_consecutive(events)
    if os.getenv("CHORD_APPLY_LEARNED_PRIORS", "").strip().lower() in ("1", "true", "yes"):
        try:
            from chord_correction_log import apply_learned_priors

            chords_out = _merge_consecutive(apply_learned_priors(chords_out))
        except Exception as exc:
            logging.getLogger("music-assistant").warning("apply_learned_priors skipped: %s", exc)

    return {
        "chords": chords_out,
        "multi_instrument_warning": max_polyphony >= POLYPHONY_WARNING_THRESHOLD,
        "max_polyphony": max_polyphony,
        "bpm": bpm,
        "key_root": key_root,
        "key_quality": key_quality,
        "beat_times": beat_times.tolist(),
        "note_events": [
            {"start": float(n[0]), "end": float(n[1]), "pitch": int(n[2])}
            for n in raw_notes
        ],
    }


def quantize_notes(
    note_events: list[dict],
    beat_times: list[float],
    bpm: float,
    grid: int = 16,
    keep_lowest: bool = False,
) -> list[dict]:
    """
    Convert raw note events to quantized beat-grid notes for sheet music.

    note_events:  [{"start", "end", "pitch"}, ...]
    beat_times:   seconds per beat (from librosa)
    grid:         subdivisions per beat (16 = 16th note resolution)
    keep_lowest:  True for bass (keep lowest pitch), False for melody (keep highest)

    Returns: [{"pitch": int, "beat_start": float, "duration_beats": float}, ...]
    """
    if not note_events or len(beat_times) < 2:
        return []

    bt = np.array(beat_times)

    def time_to_beat(t: float) -> float:
        if t <= bt[0]:
            return 0.0
        if t >= bt[-1]:
            interval = float(bt[-1] - bt[-2])
            return float(len(bt) - 1) + (t - float(bt[-1])) / max(interval, 1e-8)
        idx = int(np.searchsorted(bt, t, side="right")) - 1
        idx = max(0, min(idx, len(bt) - 2))
        interval = float(bt[idx + 1] - bt[idx])
        return idx + (t - float(bt[idx])) / max(interval, 1e-8)

    def snap(b: float) -> float:
        return round(b * grid) / grid

    # Filter out notes shorter than half a grid slot (noise reduction)
    min_dur = (60.0 / max(bpm, 1)) / grid * 0.5
    filtered = [n for n in note_events if (n["end"] - n["start"]) >= min_dur]

    raw = []
    for n in filtered:
        bs = snap(time_to_beat(n["start"]))
        be = snap(time_to_beat(n["end"]))
        dur = max(1.0 / grid, be - bs)
        raw.append((bs, dur, n["pitch"]))

    # Keep single voice per grid slot (top for melody, bottom for bass)
    by_slot: dict[int, tuple] = {}
    for bs, dur, pitch in raw:
        key = int(round(bs * grid))
        if key not in by_slot:
            by_slot[key] = (bs, dur, pitch)
        else:
            existing = by_slot[key][2]
            if (keep_lowest and pitch < existing) or (not keep_lowest and pitch > existing):
                by_slot[key] = (bs, dur, pitch)

    result = sorted(by_slot.values(), key=lambda x: x[0])
    MAX_NOTES = 64
    return [
        {"pitch": p, "beat_start": round(bs, 6), "duration_beats": round(dur, 6)}
        for bs, dur, p in result[:MAX_NOTES]
    ]


def notes_to_abc(
    quantized: list[dict],
    bpm: float,
    chords: list[dict],
    beat_times: list[float],
    clef: str = "treble",
) -> str:
    """
    Generate ABC notation string from quantized notes + chord events.

    quantized:   output of quantize_notes()
    chords:      output of analyze_chords()["chords"]
    beat_times:  seconds per beat
    clef:        "treble" | "bass"
    """
    if not quantized:
        return ""

    GRID = 16
    BEATS_PER_MEASURE = 4
    SLOTS_PER_MEASURE = BEATS_PER_MEASURE * GRID  # 64 slots per measure

    abc_header = (
        f"X:1\nT:\nQ:1/4={int(round(bpm))}\nM:4/4\nL:1/16\n"
        f"K:C clef={clef}\n"
    )

    # Map beat index → chord label string
    chord_at_beat: dict[int, str] = {}
    for c in chords:
        if c["chord"] == "N":
            continue
        for i, t in enumerate(beat_times):
            if t >= c["time"] - 0.05:
                label = c["chord"] + ("m" if c["quality"] == "minor" else "")
                if i not in chord_at_beat:
                    chord_at_beat[i] = f'"{label}"'
                break

    # Build slot array
    last_beat = max(n["beat_start"] + n["duration_beats"] for n in quantized)
    n_measures = max(1, int(np.ceil(last_beat / BEATS_PER_MEASURE)) + 1)
    total_slots = n_measures * SLOTS_PER_MEASURE

    slot: list[tuple | None] = [None] * total_slots
    for n in quantized:
        s = int(round(n["beat_start"] * GRID))
        d = max(1, int(round(n["duration_beats"] * GRID)))
        if s < total_slots:
            slot[s] = (n["pitch"], min(d, total_slots - s))

    # Generate ABC tokens
    tokens: list[str] = []
    i = 0
    while i < total_slots:
        beat_idx = i // GRID
        if i % GRID == 0 and beat_idx in chord_at_beat:
            tokens.append(chord_at_beat[beat_idx])
        if i > 0 and i % SLOTS_PER_MEASURE == 0:
            tokens.append("|")

        if slot[i] is not None:
            pitch, dur = slot[i]
            clamped = max(36, min(91, pitch))
            abc_note = MIDI_TO_ABC.get(pitch) or MIDI_TO_ABC.get(clamped, "C")
            # Truncate duration if a new note starts before it ends
            advance = dur
            for j in range(1, dur):
                if i + j < total_slots and slot[i + j] is not None:
                    advance = j
                    break
            tokens.append(f"{abc_note}{advance}")
            i += advance
        else:
            # Accumulate rest until next note, beat, or measure boundary
            rest_len = 1
            while (
                i + rest_len < total_slots
                and slot[i + rest_len] is None
                and (i + rest_len) % SLOTS_PER_MEASURE != 0
                and (i + rest_len) % GRID != 0
            ):
                rest_len += 1
            tokens.append(f"z{rest_len}")
            i += rest_len

    return abc_header + " ".join(tokens) + " |]"


def _compute_max_polyphony(note_events) -> int:
    """Count the maximum number of simultaneously sounding notes."""
    if not note_events:
        return 0
    timeline = []
    for note in note_events:
        timeline.append((float(note[0]), 1))
        timeline.append((float(note[1]), -1))
    timeline.sort(key=lambda x: (x[0], x[1]))
    max_poly = current = 0
    for _, delta in timeline:
        current += delta
        if current > max_poly:
            max_poly = current
    return max_poly


def _merge_consecutive(events: list[dict]) -> list[dict]:
    if not events:
        return []
    merged = [events[0].copy()]
    for ev in events[1:]:
        last = merged[-1]
        if ev["chord"] == last["chord"] and ev["quality"] == last["quality"]:
            last["end"] = ev["end"]
            last["confidence"] = max(last["confidence"], ev["confidence"])
        else:
            merged.append(ev.copy())
    return merged
