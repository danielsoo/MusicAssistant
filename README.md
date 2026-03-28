# Music Assistant

Web app for **stem separation** (Demucs), **chord detection** (Basic Pitch + template matching), **key / BPM**, optional **guitar type** separation, and **editable chord timelines** with persistence and optional **correction-based learning**.

---

## Features

- Upload audio (MP3, WAV, FLAC, and other common formats) and run a background analysis job.
- **Demucs** multi-stem output; optional **multiple Demucs models** for comparison (`DEMUCS_MODEL_NAMES`).
- Per-stem **chord timeline**, dual **AI vs Edit** rows, playback sync with highlight and seek-on-click.
- **Manual chord edits** auto-saved to the server; optional logging of your edits as ground truth under `data/` for priors / future training.
- **Stem mixer** (Web Audio) and optional **MongoDB** or **SQLite** fallback for auth and saved jobs.
- Optional **Google OAuth** when `GOOGLE_CLIENT_ID` is set.

---

## Requirements

| Item | Notes |
|------|--------|
| **Python** | **3.10 or 3.11** recommended (Basic Pitch / TensorFlow stack; 3.12+ often lacks matching wheels on macOS). |
| **ffmpeg** | Required by Demucs (e.g. `brew install ffmpeg` on macOS). |
| **PyTorch** | Pulled in via Demucs / model inference; install per your platform. |

---

## Quick start

```bash
git clone https://github.com/danielsoo/MusicAssistant.git
cd MusicAssistant

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cp .env.example .env
# Edit .env: MongoDB optional; app falls back to local SQLite if Mongo is unavailable.

cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — the UI is served from `frontend/index.html`.

On macOS with Homebrew Python 3.11, you can use **`./run.sh`** from the repo root (creates the venv, installs deps when `requirements.txt` changes, starts Uvicorn).

---

## Environment variables

Copy **`.env.example`** to **`.env`**. Highlights:

| Variable | Purpose |
|----------|---------|
| `MONGODB_URI` / `MONGODB_DB` | Cloud or local MongoDB; leave unset or broken TLS triggers SQLite fallback. |
| `GOOGLE_CLIENT_ID` | Google Sign-In for the web client. |
| `DEMUCS_MODEL_NAMES` | Comma-separated Demucs `--name` values (default: single fast model; more models = longer runs). |
| `CHORD_APPLY_LEARNED_PRIORS` | Set to `1` to apply accumulated label priors from your saved edits (solo / lab use). |
| `CHORD_PRIOR_*` | Tunables for prior strength (see `.env.example`). |

---

## Repository layout

```
MusicAssistant/
├── backend/           # FastAPI app (main.py), chord pipeline, optional models (*.pt)
├── frontend/          # Single-page UI (index.html)
├── scripts/           # Data prep, training helpers, chord prior summary
├── mobile-app/        # Separate React Native–style client scaffold (optional)
├── data/              # Created at runtime for chord logs / priors (gitignored)
└── run.sh             # macOS-oriented dev launcher
```

Pretrained **`.pt`** checkpoints (classifier / separator) are included for local inference when present. Large assets (**IRMAS**, `downloads/`, `training_data/`, local **`.mp3`**) are **gitignored**; add them locally or use Git LFS if you need them in a remote.

---

## Chord learning (optional)

Saving edits triggers:

1. Append rows to **`data/chord_corrections.jsonl`** (AI vs your chords + metadata).
2. Update **`data/chord_learned_priors.json`** (aligned AI→user label counts).

With **`CHORD_APPLY_LEARNED_PRIORS=1`**, new analyses can softly remap low-confidence segments toward your most frequent corrections. Summarize priors:

```bash
python scripts/summarize_chord_priors.py
```

---

## License

All rights reserved unless you add an explicit `LICENSE` file.
