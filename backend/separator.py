import os
import subprocess
import sys
from pathlib import Path


STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

def list_demucs_models() -> list[str]:
    """Comma-separated Demucs `--name` values (env DEMUCS_MODEL_NAMES).

    Default is a single model for speed. Use e.g. ``htdemucs_6s,htdemucs_ft`` to
    compare multiple backends (each name runs a full Demucs pass).
    """
    raw = (os.getenv("DEMUCS_MODEL_NAMES") or "htdemucs_6s").strip()
    if not raw:
        raw = "htdemucs_6s"
    return [m.strip() for m in raw.split(",") if m.strip()]


def separate_stems(input_path: str, output_dir: str, model_name: str = "htdemucs_6s") -> dict[str, str]:
    """
    Separate audio into 6 stems using Demucs.

    Returns dict mapping stem name -> absolute WAV path.
    Output layout: {output_dir}/{model_name}/{song_stem}/{stem}.wav
    """
    song_stem = Path(input_path).stem

    cmd = [
        sys.executable, "-m", "demucs",
        "--name", model_name,
        "--out", output_dir,
        input_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs ({model_name}) failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )

    stem_dir = Path(output_dir) / model_name / song_stem
    if not stem_dir.exists():
        raise RuntimeError(f"Expected Demucs output dir not found: {stem_dir}")

    stems = {}
    for stem in STEMS:
        wav = stem_dir / f"{stem}.wav"
        if wav.exists():
            stems[stem] = str(wav)

    if not stems:
        raise RuntimeError(f"No stem WAV files found in {stem_dir}")

    return stems
