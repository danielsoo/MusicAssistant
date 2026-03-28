import uuid
import time
import re
import logging
import hashlib
import os
import secrets
import sqlite3
import json
from pathlib import Path
from threading import Thread
from dotenv import load_dotenv

try:
    from pymongo import ASCENDING, MongoClient
    from pymongo.errors import DuplicateKeyError, PyMongoError
except Exception:  # pragma: no cover - optional dependency in local dev
    MongoClient = None
    ASCENDING = None
    DuplicateKeyError = Exception
    PyMongoError = Exception

try:
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2 import id_token as google_id_token
except Exception:  # pragma: no cover - optional dependency in local dev
    GoogleRequest = None
    google_id_token = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("music-assistant")

# Load environment variables from repository root .env and backend/.env.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import Response
from pydantic import BaseModel

from chord_correction_log import record_user_chords_as_truth
from separator import list_demucs_models, separate_stems
from chord_analyzer import analyze_chords, quantize_notes, notes_to_abc
from guitar_classifier import (
    MODEL_PATH as GUITAR_CLASSIFIER_MODEL_PATH,
    classify_guitar,
)
from guitar_separator_inference import (
    MODEL_PATH as GUITAR_SEPARATOR_MODEL_PATH,
    separate_guitar,
)
from guitar_extractor_inference import extract_guitars, MODEL_PATH as EXTRACTOR_MODEL_PATH

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Music Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR    = Path("/tmp/music-assistant/uploads")
OUTPUT_DIR    = Path("/tmp/music-assistant/outputs")
DB_PATH       = Path("/tmp/music-assistant/music_assistant.db")
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"
MONGODB_URI   = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB    = os.getenv("MONGODB_DB", "music_assistant").strip()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm"}

JOBS: dict[str, dict] = {}
MONGO_CLIENT = None
MONGO_DB = None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _init_sqlite_schema() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS saved_jobs (
            job_id      TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            filename    TEXT NOT NULL,
            created_at  REAL NOT NULL,
            global_key  TEXT,
            result_json TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def _mongo_failover(reason: str | BaseException | None = None) -> None:
    """Drop Mongo client after TLS / network failures so requests use SQLite."""
    global MONGO_CLIENT, MONGO_DB
    if reason is not None:
        log.warning("MongoDB unavailable (%s); switching to local SQLite at %s", reason, DB_PATH)
    if MONGO_CLIENT is not None:
        try:
            MONGO_CLIENT.close()
        except Exception:
            pass
    MONGO_CLIENT = None
    MONGO_DB = None
    _init_sqlite_schema()


def _init_db():
    global MONGO_CLIENT, MONGO_DB

    if MONGODB_URI:
        if MongoClient is None:
            raise RuntimeError("MONGODB_URI is set but pymongo is not installed")
        try:
            MONGO_CLIENT = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=10_000,
            )
            MONGO_DB = MONGO_CLIENT[MONGODB_DB]

            # Authentication and library indexes.
            MONGO_DB.users.create_index([( "username", ASCENDING)], unique=True)
            MONGO_DB.users.create_index([( "email", ASCENDING)], unique=True, sparse=True)
            MONGO_DB.users.create_index([( "google_sub", ASCENDING)], unique=True, sparse=True)
            MONGO_DB.sessions.create_index([( "token", ASCENDING)], unique=True)
            MONGO_DB.sessions.create_index([( "user_id", ASCENDING)])
            MONGO_DB.saved_jobs.create_index([( "job_id", ASCENDING)], unique=True)
            MONGO_DB.saved_jobs.create_index([( "user_id", ASCENDING), ("created_at", ASCENDING)])
            log.info("MongoDB connected: db=%s", MONGODB_DB)
            return
        except Exception as exc:
            log.warning(
                "MongoDB connection failed (%s); using local SQLite at %s",
                exc,
                DB_PATH,
            )
            MONGO_CLIENT = None
            MONGO_DB = None

    _init_sqlite_schema()


def _use_mongo() -> bool:
    return MONGO_DB is not None


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return secrets.compare_digest(key.hex(), key_hex)
    except Exception:
        return False


def _get_user(token: str) -> dict | None:
    if not token:
        return None

    if _use_mongo():
        try:
            session = MONGO_DB.sessions.find_one({"token": token}, {"_id": 0, "user_id": 1})
            if not session:
                return None
            user = MONGO_DB.users.find_one(
                {"id": session["user_id"]},
                {"_id": 0, "id": 1, "username": 1, "full_name": 1, "age": 1, "email": 1},
            )
            return user or None
        except PyMongoError as exc:
            _mongo_failover(exc)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT u.id, u.username FROM sessions s "
        "JOIN users u ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    )
    row = c.fetchone()
    conn.close()
    return {"id": row[0], "username": row[1]} if row else None


def _token_from(request: Request) -> str:
    return request.headers.get("Authorization", "").removeprefix("Bearer ").strip()


def _create_user(
    user_id: str,
    username: str,
    password_hash: str,
    full_name: str | None = None,
    age: int | None = None,
    email: str | None = None,
    google_sub: str | None = None,
):
    now = time.time()
    normalized_email = (email or "").strip().lower() or None
    if _use_mongo():
        try:
            MONGO_DB.users.insert_one({
                "id": user_id,
                "username": username,
                "password_hash": password_hash,
                "full_name": (full_name or "").strip() or None,
                "age": age,
                "email": normalized_email,
                "google_sub": google_sub,
                "created_at": now,
            })
            return
        except DuplicateKeyError:
            raise HTTPException(409, "Username or email already taken")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?,?,?,?)",
            (user_id, username, password_hash, now),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Username already taken")


def _find_user_by_identifier(identifier: str) -> dict | None:
    ident = (identifier or "").strip()
    if not ident:
        return None

    if _use_mongo():
        query = {"$or": [{"username": ident}, {"email": ident.lower()}]}
        user = MONGO_DB.users.find_one(query, {"_id": 0, "id": 1, "username": 1, "password_hash": 1})
        return user or None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (ident,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2]}


def _make_unique_username(seed: str) -> str:
    base = "".join(ch for ch in seed.lower() if ch.isalnum() or ch == "_")[:24] or "user"
    candidate = base
    if _use_mongo():
        while MONGO_DB.users.find_one({"username": candidate}, {"_id": 1}):
            candidate = f"{base}{secrets.randbelow(10000):04d}"
        return candidate

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    while True:
        c.execute("SELECT 1 FROM users WHERE username = ?", (candidate,))
        if not c.fetchone():
            conn.close()
            return candidate
        candidate = f"{base}{secrets.randbelow(10000):04d}"


def _upsert_google_user(google_sub: str, email: str, full_name: str | None) -> dict:
    normalized_email = email.strip().lower()
    if _use_mongo():
        user = MONGO_DB.users.find_one({"google_sub": google_sub}, {"_id": 0})
        if not user and normalized_email:
            user = MONGO_DB.users.find_one({"email": normalized_email}, {"_id": 0})

        if not user:
            seed = normalized_email.split("@", 1)[0] if normalized_email else "user"
            username = _make_unique_username(seed)
            user_id = uuid.uuid4().hex
            _create_user(
                user_id=user_id,
                username=username,
                password_hash=_hash_password(secrets.token_hex(16)),
                full_name=full_name,
                age=None,
                email=normalized_email,
                google_sub=google_sub,
            )
            return {"id": user_id, "username": username, "full_name": full_name, "email": normalized_email}

        updates = {"google_sub": google_sub}
        if normalized_email:
            updates["email"] = normalized_email
        if full_name:
            updates["full_name"] = full_name
        MONGO_DB.users.update_one({"id": user["id"]}, {"$set": updates})
        user.update(updates)
        return {"id": user["id"], "username": user["username"], "full_name": user.get("full_name"), "email": user.get("email")}

    # SQLite fallback: create a local account from Google profile if needed.
    username = _make_unique_username(normalized_email.split("@", 1)[0] if normalized_email else "user")
    user_id = uuid.uuid4().hex
    _create_user(user_id, username, _hash_password(secrets.token_hex(16)))
    return {"id": user_id, "username": username, "full_name": full_name, "email": normalized_email}


def _create_session(token: str, user_id: str):
    now = time.time()
    if _use_mongo():
        MONGO_DB.sessions.insert_one({"token": token, "user_id": user_id, "created_at": now})
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)", (token, user_id, now))
    conn.commit()
    conn.close()


def _delete_session(token: str):
    if _use_mongo():
        MONGO_DB.sessions.delete_one({"token": token})
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def _list_saved_jobs(user_id: str) -> list[dict]:
    if _use_mongo():
        docs = MONGO_DB.saved_jobs.find(
            {"user_id": user_id},
            {"_id": 0, "job_id": 1, "filename": 1, "created_at": 1, "global_key": 1},
        ).sort("created_at", -1).limit(50)
        return list(docs)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT job_id, filename, created_at, global_key FROM saved_jobs "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [{"job_id": r[0], "filename": r[1], "created_at": r[2], "global_key": r[3] or ""} for r in rows]


def _get_saved_result_with_owner(job_id: str) -> tuple[dict | None, str | None]:
    if _use_mongo():
        doc = MONGO_DB.saved_jobs.find_one({"job_id": job_id}, {"_id": 0, "result_json": 1, "user_id": 1})
        if not doc:
            return None, None
        return doc.get("result_json"), doc.get("user_id")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT result_json, user_id FROM saved_jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None, None
    return json.loads(row[0]), row[1]


def _get_saved_result(job_id: str) -> dict | None:
    if _use_mongo():
        doc = MONGO_DB.saved_jobs.find_one({"job_id": job_id}, {"_id": 0, "result_json": 1})
        return (doc or {}).get("result_json") if doc else None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT result_json FROM saved_jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _save_job_to_db(job_id: str, job: dict):
    try:
        stems_out = {}
        for stem_name, data in job["stems"].items():
            stems_out[stem_name] = {
                "audio_url":               data["audio_url"],
                "wav_path":                data.get("wav_path", ""),
                "chords":                  data["chords"],
                "manual_chords":           data.get("manual_chords"),
                "multi_instrument_warning":data.get("multi_instrument_warning", False),
                "max_polyphony":           data.get("max_polyphony", 0),
                "bpm":                     data.get("bpm", 0.0),
                "key_root":                data.get("key_root", ""),
                "key_quality":             data.get("key_quality", ""),
                "abc_notation":            data.get("abc_notation", ""),
                "guitar_type":             data.get("guitar_type"),
                "guitar_confidence":       data.get("guitar_confidence"),
            }
        result = {
            "job_id":     job_id,
            "filename":   job["filename"],
            "global_key": job.get("global_key", ""),
            "stems":      stems_out,
        }
        if _use_mongo():
            try:
                MONGO_DB.saved_jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "job_id": job_id,
                        "user_id": job["user_id"],
                        "filename": job["filename"],
                        "created_at": job["created_at"],
                        "global_key": job.get("global_key", ""),
                        "result_json": result,
                    }},
                    upsert=True,
                )
            except PyMongoError as exc:
                _mongo_failover(exc)
        if not _use_mongo():
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO saved_jobs "
                "(job_id, user_id, filename, created_at, global_key, result_json) "
                "VALUES (?,?,?,?,?,?)",
                (job_id, job["user_id"], job["filename"], job["created_at"],
                 job.get("global_key", ""), json.dumps(result)),
            )
            conn.commit()
            conn.close()
        log.info("[%s] ✓ saved to DB", job_id[:8])
    except Exception as exc:
        log.error("[%s] DB save failed: %s", job_id[:8], exc)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _init_db()


# ---------------------------------------------------------------------------
# Background processor
# ---------------------------------------------------------------------------

def _timbre_role_match(stem_key: str, info: dict) -> bool | None:
    """
    IRMAS-style classifier vs stem label: True/False/None (inconclusive).
    Used only for QA hints, not to override separation.
    """
    gt = info.get("guitar_type", "unknown")
    conf = float(info.get("confidence", 0))
    n_chunks = int(info.get("n_chunks", 0))
    if gt == "unknown" or n_chunks < 3:
        return None
    if gt == "mixed":
        return None
    if stem_key.startswith("acoustic_guitar"):
        if gt == "electric" and conf >= 0.55:
            return False
        return True
    if stem_key.startswith("electric_guitar"):
        if gt == "acoustic" and conf >= 0.55:
            return False
        return True
    return None


def _enrich_guitar_stems_with_timbre_classifier(job_id: str, job: dict) -> None:
    """
    Run IRMAS-trained classify_guitar on separated acoustic/electric WAVs (QA overlay).
    Does nothing if guitar_classifier.pt is missing.
    """
    if not GUITAR_CLASSIFIER_MODEL_PATH.exists():
        return
    for stem_key in list(job["stems"].keys()):
        if not (
            stem_key.startswith("acoustic_guitar")
            or stem_key.startswith("electric_guitar")
        ):
            continue
        entry = job["stems"].get(stem_key)
        if not entry or not entry.get("wav_path"):
            continue
        info = classify_guitar(entry["wav_path"])
        entry["timbre_classifier"] = info
        entry["timbre_matches_role"] = _timbre_role_match(stem_key, info)
        log.info(
            "[%s] timbre QA %s: type=%s conf=%.2f chunks=%s matches_role=%s",
            job_id[:8],
            stem_key,
            info.get("guitar_type"),
            float(info.get("confidence", 0)),
            info.get("n_chunks"),
            entry["timbre_matches_role"],
        )


def _variant_slug(model_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip()).strip("_")
    return s or "model"


def _ingest_wav_stem(job_id: str, job: dict, stem_name: str, wav_path: str) -> None:
    """Run chord analysis and register one stem (vocals, bass, guitar_XXX, …)."""
    t1 = time.time()
    log.info("[%s] ○ analyzing: %s", job_id[:8], stem_name)
    result = analyze_chords(wav_path)
    log.info(
        "[%s]   chords detected: %d, BPM=%.1f",
        job_id[:8], len(result["chords"]), result.get("bpm", 0),
    )
    abc_string = ""
    if stem_name != "drums" and result.get("note_events"):
        qnotes = quantize_notes(
            result["note_events"], result["beat_times"], result["bpm"],
            keep_lowest=(stem_name == "bass"),
        )
        abc_string = notes_to_abc(
            qnotes, result["bpm"], result["chords"], result["beat_times"],
            clef="bass" if stem_name == "bass" else "treble",
        )
    job["stems"][stem_name] = {
        "wav_path":                 wav_path,
        "chords":                   result["chords"],
        "multi_instrument_warning": result["multi_instrument_warning"],
        "max_polyphony":            result["max_polyphony"],
        "bpm":                      result.get("bpm", 0.0),
        "key_root":                 result.get("key_root", ""),
        "key_quality":              result.get("key_quality", ""),
        "abc_notation":             abc_string,
        "audio_url":                f"/audio/{job_id}/{stem_name}",
        "guitar_type":              None,
        "guitar_confidence":        None,
    }
    log.info("[%s] ✓ %s done (%.1fs)", job_id[:8], stem_name, time.time() - t1)


def _attach_ae_from_split(
    job_id: str, job: dict, split: dict[str, str], ac_key: str, el_key: str,
) -> None:
    """Register acoustic + electric stems from separate_guitar() paths."""
    for sub_name, path_key in [(ac_key, "acoustic"), (el_key, "electric")]:
        sub_path = split[path_key]
        sub_result = analyze_chords(sub_path)
        sub_abc = ""
        if sub_result.get("note_events"):
            qn = quantize_notes(
                sub_result["note_events"],
                sub_result["beat_times"],
                sub_result["bpm"],
            )
            sub_abc = notes_to_abc(
                qn, sub_result["bpm"],
                sub_result["chords"],
                sub_result["beat_times"],
            )
        job["stems"][sub_name] = {
            "wav_path":                 sub_path,
            "chords":                   sub_result["chords"],
            "multi_instrument_warning": False,
            "max_polyphony":            sub_result.get("max_polyphony", 0),
            "bpm":                      sub_result.get("bpm", 0.0),
            "key_root":                 sub_result.get("key_root", ""),
            "key_quality":              sub_result.get("key_quality", ""),
            "abc_notation":             sub_abc,
            "audio_url":                f"/audio/{job_id}/{sub_name}",
            "guitar_type":              "acoustic" if sub_name.startswith("acoustic_guitar") else "electric",
            "guitar_confidence":        1.0,
        }
        log.info("[%s] ✓ %s done", job_id[:8], sub_name)


def _process_job(job_id: str):
    job = JOBS[job_id]
    fname = job["filename"]
    try:
        log.info("[%s] ▶ start: %s", job_id[:8], fname)
        job["status"] = "separating"
        t0 = time.time()

        models = list_demucs_models()
        backend_stems: dict[str, dict[str, str]] = {}
        for model in models:
            try:
                backend_stems[model] = separate_stems(
                    job["input_path"], str(OUTPUT_DIR), model_name=model
                )
            except Exception as exc:
                log.warning("[%s] Demucs model %s failed: %s", job_id[:8], model, exc)

        if not backend_stems:
            raise RuntimeError(
                "No Demucs output — set DEMUCS_MODEL_NAMES in .env (e.g. htdemucs_6s or htdemucs_6s,htdemucs_ft) "
                "and ensure each name is valid for your demucs install."
            )

        primary_model = next(iter(backend_stems.keys()))
        stems_primary = backend_stems[primary_model]
        log.info(
            "[%s] ✓ separated (%d backend(s), primary=%s, %.1fs): %s",
            job_id[:8],
            len(backend_stems),
            primary_model,
            time.time() - t0,
            list(stems_primary.keys()),
        )

        job["status"] = "analyzing"
        job["stems"] = {}

        # Non-guitar stems once (from primary Demucs run)
        for stem_name, wav_path in stems_primary.items():
            if stem_name == "guitar":
                continue
            _ingest_wav_stem(job_id, job, stem_name, wav_path)

        has_sep = GUITAR_SEPARATOR_MODEL_PATH.exists()
        n_back = len(backend_stems)

        for model, sm in backend_stems.items():
            slug = _variant_slug(model)
            gw = sm.get("guitar")
            if not gw:
                continue
            if not has_sep:
                _ingest_wav_stem(job_id, job, f"guitar_{slug}", gw)
                continue
            out_dir = Path(OUTPUT_DIR) / f"{job_id}_guitar_{slug}"
            try:
                split = separate_guitar(gw, str(out_dir))
            except Exception as exc:
                log.error("[%s] ✗ separate_guitar (%s): %s", job_id[:8], model, exc)
                split = None
            if split:
                log.info("[%s] ✓ guitar A/E split for model %s", job_id[:8], model)
                _attach_ae_from_split(
                    job_id,
                    job,
                    split,
                    f"acoustic_guitar_{slug}",
                    f"electric_guitar_{slug}",
                )
            else:
                log.warning("[%s] separate_guitar failed for %s — raw guitar stem", job_id[:8], model)
                _ingest_wav_stem(job_id, job, f"guitar_{slug}", gw)

        if has_sep and os.getenv("GUITAR_COMPARE_FULLMIX", "").strip().lower() in (
            "1", "true", "yes",
        ):
            log.info("[%s] ○ guitar_separator on full mix (GUITAR_COMPARE_FULLMIX)...", job_id[:8])
            out_fm = Path(OUTPUT_DIR) / f"{job_id}_guitar_fullmix"
            try:
                split_fm = separate_guitar(job["input_path"], str(out_fm))
            except Exception as exc:
                log.error("[%s] ✗ fullmix guitar_separator: %s", job_id[:8], exc)
                split_fm = None
            if split_fm:
                _attach_ae_from_split(
                    job_id,
                    job,
                    split_fm,
                    "acoustic_guitar_fullmix",
                    "electric_guitar_fullmix",
                )

        if EXTRACTOR_MODEL_PATH.exists() and n_back == 1:
            log.info("[%s] ○ guitar extractor (single backend)...", job_id[:8])
            t_ext = time.time()
            guitar_out_dir = str(Path(job["input_path"]).parent / f"{job_id}_guitars")
            try:
                ext = extract_guitars(job["input_path"], guitar_out_dir)
            except Exception as ext_exc:
                log.error("[%s] ✗ guitar extractor error: %s", job_id[:8], ext_exc)
                ext = None

            if ext:
                log.info(
                    "[%s] ✓ guitar extractor (%.1fs)  ac=%s  el=%s",
                    job_id[:8],
                    time.time() - t_ext,
                    ext.get("ac_present"),
                    ext.get("el_present"),
                )
                for sub_name, path_key, present_key in [
                    ("acoustic_guitar_extractor", "acoustic", "ac_present"),
                    ("electric_guitar_extractor", "electric", "el_present"),
                ]:
                    if not ext.get(present_key):
                        continue
                    sub_path = ext[path_key]
                    sub_result = analyze_chords(sub_path)
                    sub_abc = ""
                    if sub_result.get("note_events"):
                        qn = quantize_notes(
                            sub_result["note_events"],
                            sub_result["beat_times"],
                            sub_result["bpm"],
                        )
                        sub_abc = notes_to_abc(
                            qn,
                            sub_result["bpm"],
                            sub_result["chords"],
                            sub_result["beat_times"],
                        )
                    job["stems"][sub_name] = {
                        "wav_path":                 sub_path,
                        "chords":                   sub_result["chords"],
                        "multi_instrument_warning": False,
                        "max_polyphony":            sub_result.get("max_polyphony", 0),
                        "bpm":                      sub_result.get("bpm", 0.0),
                        "key_root":                 sub_result.get("key_root", ""),
                        "key_quality":              sub_result.get("key_quality", ""),
                        "abc_notation":             sub_abc,
                        "audio_url":                f"/audio/{job_id}/{sub_name}",
                        "guitar_type":              "acoustic" if "acoustic" in sub_name else "electric",
                        "guitar_confidence":        1.0,
                    }
                    log.info("[%s] ✓ %s analyzed", job_id[:8], sub_name)
            else:
                log.warning("[%s] ✗ guitar extraction failed", job_id[:8])

        _enrich_guitar_stems_with_timbre_classifier(job_id, job)

        global_key = ""
        for pref in ("vocals", "piano", "bass", "drums", "other"):
            s = job["stems"].get(pref)
            if s and s.get("key_root"):
                q = "Major" if s["key_quality"] == "major" else "minor"
                global_key = f"{s['key_root']} {q}"
                break
        if not global_key:
            for k in sorted(job["stems"].keys()):
                if not k.startswith("acoustic_guitar"):
                    continue
                s = job["stems"][k]
                if s and s.get("key_root"):
                    q = "Major" if s["key_quality"] == "major" else "minor"
                    global_key = f"{s['key_root']} {q}"
                    break
        job["global_key"] = global_key

        log.info("[%s] ✓✓ all done: %s (key=%s)", job_id[:8], fname, global_key)
        job["status"] = "done"

        if job.get("user_id"):
            _save_job_to_db(job_id, job)

    except Exception as exc:
        log.error("[%s] ✗ error during processing: %s", job_id[:8], exc, exc_info=True)
        job["status"] = "error"
        job["error"] = str(exc)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class _RegisterPayload(BaseModel):
    full_name: str | None = None
    age: int | None = None
    email: str | None = None
    username: str
    password: str

class _LoginPayload(BaseModel):
    identifier: str
    password: str

class _GoogleAuthPayload(BaseModel):
    credential: str


@app.get("/auth/google-client-id")
async def google_client_id():
    return {"client_id": GOOGLE_CLIENT_ID}


@app.post("/auth/register")
async def register(payload: _RegisterPayload):
    username = payload.username.strip()
    if len(username) < 3 or len(username) > 32:
        raise HTTPException(400, "Username must be 3–32 characters")
    if len(payload.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if payload.age is not None and (payload.age < 13 or payload.age > 120):
        raise HTTPException(400, "Age must be between 13 and 120")

    user_id = uuid.uuid4().hex
    pw_hash = _hash_password(payload.password)
    _create_user(
        user_id=user_id,
        username=username,
        password_hash=pw_hash,
        full_name=payload.full_name,
        age=payload.age,
        email=payload.email,
    )

    token = secrets.token_hex(32)
    _create_session(token, user_id)
    return {
        "token": token,
        "username": username,
        "full_name": (payload.full_name or "").strip() or None,
        "age": payload.age,
        "email": (payload.email or "").strip().lower() or None,
    }


@app.post("/auth/login")
async def login(payload: _LoginPayload):
    user = _find_user_by_identifier(payload.identifier)

    if not user or not _verify_password(payload.password, user["password_hash"]):
        raise HTTPException(401, "Invalid username or password")

    token = secrets.token_hex(32)
    _create_session(token, user["id"])
    return {"token": token, "username": user["username"]}


@app.post("/auth/google")
async def google_auth(payload: _GoogleAuthPayload):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(503, "Google login is not configured")
    if GoogleRequest is None or google_id_token is None:
        raise HTTPException(503, "google-auth package is not installed")

    try:
        info = google_id_token.verify_oauth2_token(payload.credential, GoogleRequest(), GOOGLE_CLIENT_ID)
    except Exception:
        raise HTTPException(401, "Invalid Google credential")

    issuer = info.get("iss")
    if issuer not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(401, "Invalid token issuer")
    if not info.get("email_verified"):
        raise HTTPException(401, "Google email is not verified")

    sub = info.get("sub")
    email = (info.get("email") or "").strip()
    name = (info.get("name") or "").strip() or None
    if not sub or not email:
        raise HTTPException(401, "Google profile is missing required fields")

    user = _upsert_google_user(google_sub=sub, email=email, full_name=name)
    token = secrets.token_hex(32)
    _create_session(token, user["id"])
    return {
        "token": token,
        "username": user["username"],
        "full_name": user.get("full_name"),
        "email": user.get("email"),
    }


@app.post("/auth/logout")
async def logout(request: Request):
    token = _token_from(request)
    if token:
        _delete_session(token)
    return {"ok": True}


@app.get("/auth/me")
async def me(request: Request):
    user = _get_user(_token_from(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


@app.get("/my-jobs")
async def my_jobs(request: Request):
    user = _get_user(_token_from(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    return _list_saved_jobs(user["id"])


@app.get("/saved-result/{job_id}")
async def saved_result(job_id: str, request: Request):
    user = _get_user(_token_from(request))
    if not user:
        raise HTTPException(401, "Not authenticated")

    # In-memory first
    job = JOBS.get(job_id)
    if job and job["status"] == "done" and job.get("user_id") == user["id"]:
        stems_out = {}
        for stem_name, data in job["stems"].items():
            stems_out[stem_name] = {
                "audio_url":               data["audio_url"],
                "chords":                  data["chords"],
                "manual_chords":           data.get("manual_chords"),
                "multi_instrument_warning":data.get("multi_instrument_warning", False),
                "max_polyphony":           data.get("max_polyphony", 0),
                "bpm":                     data.get("bpm", 0.0),
                "key_root":                data.get("key_root", ""),
                "key_quality":             data.get("key_quality", ""),
                "abc_notation":            data.get("abc_notation", ""),
                "guitar_type":             data.get("guitar_type"),
                "guitar_confidence":       data.get("guitar_confidence"),
            }
        return {"job_id": job_id, "filename": job["filename"],
                "global_key": job.get("global_key", ""), "stems": stems_out}

    # DB
    result_json, owner_id = _get_saved_result_with_owner(job_id)
    if not result_json:
        raise HTTPException(404, "Job not found")
    if owner_id != user["id"]:
        raise HTTPException(403, "Forbidden")
    return result_json


# ---------------------------------------------------------------------------
# Main endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))


@app.post("/analyze", status_code=202)
async def analyze(request: Request, audio_file: UploadFile = File(...)):
    ext = Path(audio_file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    user = _get_user(_token_from(request))
    job_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    with open(save_path, "wb") as f:
        while chunk := await audio_file.read(1024 * 1024):
            f.write(chunk)

    JOBS[job_id] = {
        "id":         job_id,
        "status":     "queued",
        "filename":   audio_file.filename,
        "created_at": time.time(),
        "input_path": str(save_path),
        "stems":      {},
        "error":      None,
        "user_id":    user["id"] if user else None,
    }

    Thread(target=_process_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id":      job_id,
        "status":      job["status"],
        "filename":    job["filename"],
        "created_at":  job["created_at"],
        "error":       job.get("error"),
    }


@app.get("/result/{job_id}")
async def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "error":
        raise HTTPException(500, job.get("error", "Unknown error"))
    if job["status"] != "done":
        return Response(
            content='{"detail": "Processing not complete"}',
            status_code=202,
            media_type="application/json",
        )

    stems_out = {}
    for stem_name, data in job["stems"].items():
        stems_out[stem_name] = {
            "audio_url":               data["audio_url"],
            "chords":                  data["chords"],
            "manual_chords":           data.get("manual_chords"),
            "multi_instrument_warning":data.get("multi_instrument_warning", False),
            "max_polyphony":           data.get("max_polyphony", 0),
            "bpm":                     data.get("bpm", 0.0),
            "key_root":                data.get("key_root", ""),
            "key_quality":             data.get("key_quality", ""),
            "abc_notation":            data.get("abc_notation", ""),
            "guitar_type":             data.get("guitar_type"),
            "guitar_confidence":       data.get("guitar_confidence"),
        }

    return {
        "job_id":     job_id,
        "filename":   job["filename"],
        "global_key": job.get("global_key", ""),
        "stems":      stems_out,
    }


@app.get("/audio/{job_id}/{stem_name}")
async def audio(job_id: str, stem_name: str):
    wav_path = None

    job = JOBS.get(job_id)
    if job and job["status"] == "done":
        stem_data = job["stems"].get(stem_name)
        if stem_data:
            wav_path = stem_data.get("wav_path")

    if not wav_path:
        saved = _get_saved_result(job_id)
        if saved:
            stem_data = saved.get("stems", {}).get(stem_name)
            if stem_data:
                wav_path = stem_data.get("wav_path")

    if not wav_path or not Path(wav_path).exists():
        raise HTTPException(404, "Audio not available")

    return FileResponse(
        wav_path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
    )


# ---------------------------------------------------------------------------
# Manual chord persistence
# ---------------------------------------------------------------------------

class _ChordEvent(BaseModel):
    time: float
    end: float
    chord: str
    quality: str
    confidence: float

class _ChordsPayload(BaseModel):
    chords: list[_ChordEvent]


@app.put("/chords/{job_id}/{stem_name}")
async def save_chords(job_id: str, stem_name: str, payload: _ChordsPayload):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Job not found or not complete")
    stem_data = job["stems"].get(stem_name)
    if not stem_data:
        raise HTTPException(404, f"Stem '{stem_name}' not found")
    user_list = [c.model_dump() for c in payload.chords]
    stem_data["manual_chords"] = user_list
    record_user_chords_as_truth(job_id, stem_name, job, user_list)
    return {"ok": True, "learned": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
