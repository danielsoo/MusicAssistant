"""
Cambridge Multi-track data → auto-generate training pairs

v2: add full-instrument mix (mixture.wav)
  - Sum all instrument tracks in the ZIP → mixture.wav (like a real song)
  - acoustic.wav / electric.wav stored at levels relative to mixture
  - Training learns mixture → (acoustic, electric) separation vs real audio

Usage:
    1. Put ZIP files from cambridge-mt.com into downloads/
    2. python scripts/prepare_training_data.py
    3. Training pairs appear under training_data/pairs/

Output layout:
    training_data/
    └── pairs/
        ├── 000_angela_milk_cow/
        │   ├── mixture.wav    ← sum of all instruments (input)
        │   ├── acoustic.wav   ← acoustic guitar (level vs mixture)
        │   ├── electric.wav   ← electric guitar (level vs mixture)
        │   └── meta.json
        ├── 001_.../
        ...
"""

import re
import json
import shutil
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# --- Paths --------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
DOWNLOADS  = ROOT / "downloads"
PAIRS_DIR  = ROOT / "training_data" / "pairs"
SR         = 44100

# --- Instrument keyword heuristics ---------------------------------------------
ACOUSTIC_KW = [
    "acoustic", "gac", "ac gtr", "ac.gtr", "acousticgtr",
    "acoustic guitar", "acou", "ag ", "ag_", "a.gtr",
    "nylon", "fingerpick", "acoustic_guitar",
]
ELECTRIC_KW = [
    "electric", "gel", "elec", "el gtr", "el.gtr", "electricgtr",
    "electric guitar", "eg ", "eg_", "e.gtr", "e gtr",
    "distortion", "overdrive", "lead guitar", "rhythm guitar",
    "electric_guitar",
]
# Excluded when classifying guitar vs non-guitar
EXCLUDE_KW = [
    "bass", "vocal", "voice", "drum", "kick", "snare",
    "piano", "keys", "synth", "organ", "room", "reverb",
    "overhead", "click", "metronome", "midi", "ambience",
    "mix", "stem", "master", "ref",
]
# Excluded from mixture sum (pre-mixed stems, click tracks, etc.)
MIX_EXCLUDE_KW = ["mix", "stem", "master", "ref", "click", "metronome", "midi"]


def classify_track(filename: str) -> str | None:
    """Classify filename as acoustic or electric guitar. Returns 'acoustic'|'electric'|None."""
    name = filename.lower().replace("-", " ").replace("_", " ")
    for kw in EXCLUDE_KW:
        if kw in name:
            return None
    for kw in ELECTRIC_KW:
        if kw in name:
            return "electric"
    for kw in ACOUSTIC_KW:
        if kw in name:
            return "acoustic"
    return None


def is_mix_excluded(filename: str) -> bool:
    """True if this file should be excluded from the mixture sum (reference mix, click, etc.)."""
    name = filename.lower().replace("-", " ").replace("_", " ")
    return any(kw in name for kw in MIX_EXCLUDE_KW)


def load_raw(path: Path, sr: int = SR) -> np.ndarray:
    """Load audio without normalization (for mixture level math)."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def load_and_normalize(path: Path, sr: int = SR) -> np.ndarray:
    """Load audio and peak-normalize (legacy single-track use)."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    peak = np.abs(y).max()
    if peak > 1e-6:
        y = y / peak * 0.9
    return y


def save_pair(
    pair_dir: Path,
    mixture: np.ndarray,
    acoustic: np.ndarray,
    electric: np.ndarray,
    meta: dict,
) -> bool:
    """Save mixture / acoustic / electric at the same length."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    min_len = min(len(mixture), len(acoustic), len(electric))
    if min_len < SR * 10:
        return False
    mixture  = mixture[:min_len]
    acoustic = acoustic[:min_len]
    electric = electric[:min_len]
    sf.write(str(pair_dir / "mixture.wav"),  mixture,  SR, subtype="PCM_24")
    sf.write(str(pair_dir / "acoustic.wav"), acoustic, SR, subtype="PCM_24")
    sf.write(str(pair_dir / "electric.wav"), electric, SR, subtype="PCM_24")
    with open(pair_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return True


def process_zip(zip_path: Path, pair_idx: int) -> list[dict]:
    """Process one ZIP → build training pair(s)."""
    results = []
    extract_dir = zip_path.parent / f"_tmp_{zip_path.stem}"

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        wav_files = list(extract_dir.rglob("*.wav")) + list(extract_dir.rglob("*.WAV"))
        if not wav_files:
            print(f"  No WAV: {zip_path.name}")
            return results

        # Classify acoustic / electric
        acoustic_files, electric_files = [], []
        for wav in wav_files:
            label = classify_track(wav.name)
            if label == "acoustic":
                acoustic_files.append(wav)
            elif label == "electric":
                electric_files.append(wav)

        print(f"  Acoustic: {[f.name for f in acoustic_files]}")
        print(f"  Electric: {[f.name for f in electric_files]}")

        if not acoustic_files or not electric_files:
            print(f"  → Cannot build pair (missing one side)")
            return results

        # --- mixture: sum all non-excluded WAVs at raw level
        mix_candidates = [w for w in wav_files if not is_mix_excluded(w.name)]
        raw_tracks: list[np.ndarray] = []
        for wav in mix_candidates:
            try:
                raw_tracks.append(load_raw(wav))
            except Exception as e:
                print(f"    Load failed ({wav.name}): {e}")

        if not raw_tracks:
            print(f"  → mixture failed (no tracks loaded)")
            return results

        # Same length then sum
        min_len = min(len(t) for t in raw_tracks)
        full_mix_raw = sum(t[:min_len] for t in raw_tracks)

        # Peak-normalize mixture → apply same scale to guitar stems (preserve ratio)
        peak = np.abs(full_mix_raw).max()
        if peak < 1e-6:
            print(f"  → silent mixture, skip")
            return results
        scale = 0.9 / peak
        mixture = full_mix_raw * scale

        # Guitar stems: raw merge then same scale
        def merge_raw(paths: list[Path]) -> np.ndarray:
            merged = None
            for p in paths:
                try:
                    y = load_raw(p)[:min_len]
                except Exception:
                    continue
                merged = y if merged is None else merged + y[:len(merged)]
            return merged

        acoustic_raw = merge_raw(acoustic_files)
        electric_raw = merge_raw(electric_files)

        if acoustic_raw is None or electric_raw is None:
            print(f"  → guitar track load failed")
            return results

        # Levels relative to mixture
        acoustic = acoustic_raw[:min_len] * scale
        electric = electric_raw[:min_len] * scale

        slug     = re.sub(r"[^\w]", "_", zip_path.stem.lower())[:40]
        pair_dir = PAIRS_DIR / f"{pair_idx:04d}_{slug}"

        meta = {
            "source_zip":     zip_path.name,
            "acoustic_tracks": [f.name for f in acoustic_files],
            "electric_tracks": [f.name for f in electric_files],
            "mix_tracks":      [w.name for w in mix_candidates],
            "sample_rate":     SR,
            "duration_sec":    round(min_len / SR, 1),
            "has_mixture":     True,
        }

        if save_pair(pair_dir, mixture, acoustic, electric, meta):
            results.append(meta)
            print(f"  ✓ saved: {pair_dir.name}  ({meta['duration_sec']}s, "
                  f"mix={len(mix_candidates)} tracks)")
        else:
            print(f"  → too short, skip (<10s)")

    except Exception as e:
        print(f"  error: {e}")
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

    return results


def main():
    DOWNLOADS.mkdir(exist_ok=True)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(DOWNLOADS.glob("*.zip")) + sorted(DOWNLOADS.glob("*.ZIP"))

    if not zip_files:
        print(f"No ZIP files: {DOWNLOADS}")
        print("Put ZIPs downloaded from cambridge-mt.com into downloads/.")
        return

    print(f"Processing {len(zip_files)} ZIP file(s)\n")

    all_pairs = []
    for i, zp in enumerate(zip_files):
        print(f"[{i+1}/{len(zip_files)}] {zp.name}")
        pairs = process_zip(zp, len(all_pairs))
        all_pairs.extend(pairs)
        print()

    # Summary
    summary = {
        "total_pairs": len(all_pairs),
        "pairs": all_pairs,
    }
    with open(PAIRS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"=== done ===")
    print(f"Created {len(all_pairs)} training pair(s) → {PAIRS_DIR}")


if __name__ == "__main__":
    main()
