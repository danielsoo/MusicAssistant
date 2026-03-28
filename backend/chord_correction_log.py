"""
Treat user-edited chords as ground truth: append full snapshots for offline training,
and optionally learn dominant AI→user label fixes for future analyze_chords runs.

Set CHORD_APPLY_LEARNED_PRIORS=1 to nudge new analyses using accumulated priors (solo use).
"""
from __future__ import annotations

import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("music-assistant")

_lock = threading.Lock()
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = Path(os.getenv("CHORD_LEARN_DATA_DIR", str(_REPO_ROOT / "data"))).resolve()
_LOG_PATH = _DATA_DIR / "chord_corrections.jsonl"
_PRIORS_PATH = _DATA_DIR / "chord_learned_priors.json"

# Solo lab: low thresholds so a few intentional corrections start to matter.
_MIN_CORRECTIONS_FOR_PRIOR = int(os.getenv("CHORD_PRIOR_MIN_COUNT", "3"))
_MIN_DOMINANCE = float(os.getenv("CHORD_PRIOR_MIN_RATIO", "0.45"))
_CONF_CAP = float(os.getenv("CHORD_PRIOR_CONF_CAP", "0.82"))


def _label(ev: dict) -> str:
    return f"{ev.get('chord', 'N')}:{ev.get('quality', '') or 'major'}"


def _overlap(u0: float, u1: float, v0: float, v1: float) -> float:
    lo = max(u0, v0)
    hi = min(u1, v1)
    return max(0.0, hi - lo)


def _best_ai_for_user_segment(u: dict, ai_chords: list[dict]) -> tuple[dict | None, float]:
    ut0, ut1 = float(u["time"]), float(u.get("end") or u["time"] + 0.5)
    best = None
    best_ov = 0.0
    for a in ai_chords:
        if a.get("chord") == "N":
            continue
        at0, at1 = float(a["time"]), float(a.get("end") or at0 + 0.5)
        ov = _overlap(ut0, ut1, at0, at1)
        if ov > best_ov:
            best_ov = ov
            best = a
    return best, best_ov


def _load_priors() -> dict[str, Any]:
    if not _PRIORS_PATH.exists():
        return {"corrections": {}}
    try:
        with open(_PRIORS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not read %s: %s", _PRIORS_PATH, exc)
        return {"corrections": {}}


def _save_priors(data: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _PRIORS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(_PRIORS_PATH)


def _merge_alignment_into_priors(ai_chords: list[dict], user_chords: list[dict]) -> None:
    priors = _load_priors()
    corr: dict[str, dict[str, int]] = priors.setdefault("corrections", {})
    for u in user_chords:
        if u.get("chord") == "N":
            continue
        ai_hit, ov = _best_ai_for_user_segment(u, ai_chords)
        if not ai_hit or ov < 0.08:
            continue
        la, lu = _label(ai_hit), _label(u)
        if la == lu:
            continue
        bucket = corr.setdefault(la, {})
        bucket[lu] = bucket.get(lu, 0) + 1
    _save_priors(priors)


def build_replacement_map() -> dict[str, tuple[str, str]]:
    """Maps detection label 'Root:quality' -> (new_root, new_quality) when one target dominates."""
    priors = _load_priors()
    corr: dict[str, dict[str, int]] = priors.get("corrections") or {}
    out: dict[str, tuple[str, str]] = {}
    for la, targets in corr.items():
        total = sum(targets.values())
        if total < _MIN_CORRECTIONS_FOR_PRIOR:
            continue
        best_lu, best_c = max(targets.items(), key=lambda x: x[1])
        if best_c / total < _MIN_DOMINANCE:
            continue
        parts = best_lu.split(":", 1)
        if len(parts) != 2:
            continue
        root, q = parts[0], parts[1] or "major"
        out[la] = (root, q)
    return out


def apply_learned_priors(events: list[dict]) -> list[dict]:
    if not events:
        return events
    repl = build_replacement_map()
    if not repl:
        return events
    out = []
    for ev in events:
        e = ev.copy()
        lk = _label(e)
        if e.get("chord") == "N":
            out.append(e)
            continue
        conf = float(e.get("confidence") or 0)
        if lk in repl and conf < _CONF_CAP:
            nr, nq = repl[lk]
            e["chord"] = nr
            e["quality"] = nq
        out.append(e)
    return out


def record_user_chords_as_truth(
    job_id: str,
    stem_name: str,
    job: dict,
    user_chords: list[dict],
) -> None:
    """Append JSONL row + update prior counts (user labels = ground truth)."""
    stem = (job.get("stems") or {}).get(stem_name) or {}
    ai_chords = stem.get("chords") or []
    row = {
        "ts": time.time(),
        "job_id": job_id,
        "stem_name": stem_name,
        "filename": job.get("filename"),
        "wav_path": stem.get("wav_path"),
        "bpm": stem.get("bpm"),
        "key_root": stem.get("key_root"),
        "key_quality": stem.get("key_quality"),
        "chords_ai": ai_chords,
        "chords_user_truth": user_chords,
    }
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            _merge_alignment_into_priors(ai_chords, user_chords)
    except Exception as exc:
        log.warning("Chord correction log failed (non-fatal): %s", exc)
