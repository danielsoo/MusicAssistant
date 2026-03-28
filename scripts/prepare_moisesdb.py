"""
MoisesDB → 학습 데이터 변환 스크립트

MoisesDB에서 통기타(acoustic guitar) + 일렉기타(electric guitar)가 모두 있는
트랙만 골라 우리 학습 포맷으로 저장합니다.

출력 구조:
  training_data/pairs/moisesdb_{track_id}/
    acoustic.wav   ← 통기타 세션 합산
    electric.wav   ← 일렉기타 세션 합산 (clean + distorted)
    mixture.wav    ← 전체 스템 합산 (원곡)

Usage:
    .venv/bin/python3 scripts/prepare_moisesdb.py --data_path /path/to/moisesdb
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT      = Path(__file__).parent.parent
PAIRS_DIR = ROOT / "training_data" / "pairs"
SR        = 44100


def mix_sources(paths: list[str], sr: int) -> np.ndarray | None:
    """여러 오디오 파일을 불러서 합산 (mono)."""
    if not paths:
        return None
    mixed = None
    for p in paths:
        try:
            y, _ = librosa.load(p, sr=sr, mono=True)
            mixed = y if mixed is None else mixed[:len(y)] + y[:len(mixed)]
        except Exception as e:
            print(f"    skip {Path(p).name}: {e}")
    return mixed


def save_wav(arr: np.ndarray, path: Path, sr: int):
    """피크 정규화 후 저장."""
    peak = np.abs(arr).max()
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(str(path), arr, sr, subtype="PCM_24")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="MoisesDB 압축 해제 경로")
    args = parser.parse_args()

    try:
        from moisesdb.dataset import MoisesDB
    except ImportError:
        print("moisesdb 미설치. 먼저 실행하세요:")
        print("  .venv/bin/python3 -m pip install git+https://github.com/moises-ai/moises-db.git")
        sys.exit(1)

    db = MoisesDB(data_path=args.data_path, sample_rate=SR)
    print(f"총 트랙 수: {len(db)}")

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    for i, track in enumerate(db):
        print(f"\r[{i+1}/{len(db)}] {track.name[:40]}", end="", flush=True)

        guitar_sources = track.sources.get("guitar", {})
        ac_paths = guitar_sources.get("acoustic guitar", [])
        el_paths = (guitar_sources.get("clean electric guitar", []) +
                    guitar_sources.get("distorted electric guitar", []))

        # 통기타 + 일렉 둘 다 있어야 사용
        if not ac_paths or not el_paths:
            n_skip += 1
            continue

        out_dir = PAIRS_DIR / f"moisesdb_{track.id}"
        if out_dir.exists() and (out_dir / "mixture.wav").exists():
            n_ok += 1
            continue  # 이미 처리됨

        ac_audio  = mix_sources(ac_paths, SR)
        el_audio  = mix_sources(el_paths, SR)
        if ac_audio is None or el_audio is None:
            n_skip += 1
            continue

        # 전체 믹스: 모든 스템 합산
        all_stems = []
        for stem_name in track.stems:
            stem_paths = []
            for subtype_paths in track.sources.get(stem_name, {}).values():
                stem_paths.extend(subtype_paths)
            audio = mix_sources(stem_paths, SR)
            if audio is not None:
                all_stems.append(audio)

        if not all_stems:
            n_skip += 1
            continue

        max_len = max(len(a) for a in all_stems)
        mix_audio = sum(
            np.pad(a, (0, max_len - len(a))) for a in all_stems
        )

        # 길이 맞추기
        min_len = min(len(ac_audio), len(el_audio), len(mix_audio))
        ac_audio  = ac_audio[:min_len]
        el_audio  = el_audio[:min_len]
        mix_audio = mix_audio[:min_len]

        out_dir.mkdir(parents=True, exist_ok=True)
        save_wav(ac_audio,  out_dir / "acoustic.wav", SR)
        save_wav(el_audio,  out_dir / "electric.wav", SR)
        save_wav(mix_audio, out_dir / "mixture.wav",  SR)

        n_ok += 1

    print(f"\n\n완료: {n_ok}쌍 저장  /  {n_skip}쌍 스킵 (통기타 또는 일렉 없음)")
    print(f"저장 위치: {PAIRS_DIR}")


if __name__ == "__main__":
    main()
