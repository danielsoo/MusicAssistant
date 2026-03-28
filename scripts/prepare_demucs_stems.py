"""
학습 데이터의 mixture.wav → Demucs → guitar_demucs.wav 생성

각 training pair의 mixture.wav를 Demucs(htdemucs_6s)로 처리하여
guitar_demucs.wav를 저장한다.

이후 train_separator_v5.py에서 이 파일을 모델 입력으로 사용하면
추론 시 조건과 동일한 분포로 학습할 수 있다.

Usage:
    source .venv/bin/activate
    python scripts/prepare_demucs_stems.py

약 2~5분/곡 소요 (MPS 사용 시). 이미 처리된 페어는 스킵.
"""

import shutil
import tempfile
from pathlib import Path

ROOT      = Path(__file__).parent.parent
PAIRS_DIR = ROOT / "training_data" / "pairs"


def run_demucs_on_mixture(mixture_path: Path, out_guitar_path: Path) -> bool:
    """
    mixture.wav → Demucs htdemucs_6s → guitar stem 저장.
    반환: 성공 여부
    """
    import torch
    import librosa
    import soundfile as sf
    import numpy as np
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    try:
        model = get_model("htdemucs_6s")
        model.eval()
        device = ("mps"  if torch.backends.mps.is_available() else
                  "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        target_sr = model.samplerate  # 44100

        # librosa로 로드 (이미 검증된 방식)
        y, file_sr = librosa.load(str(mixture_path), sr=None, mono=False)
        # y: (channels, T) 또는 (T,) mono

        if y.ndim == 1:
            y = np.stack([y, y])       # mono → stereo (2, T)
        elif y.shape[0] > 2:
            y = y[:2]                  # 3채널 이상 → 첫 2채널

        if file_sr != target_sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=target_sr)

        wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, T)

        with torch.no_grad():
            sources = apply_model(model, wav, device=device, progress=False)
        # sources: (1, n_sources, 2, T)
        source_names = model.sources  # ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
        guitar_idx   = source_names.index("guitar")
        guitar_wav   = sources[0, guitar_idx]        # (2, T)
        guitar_mono  = guitar_wav.mean(dim=0).cpu().numpy()  # (T,)

        sf.write(str(out_guitar_path), guitar_mono, target_sr, subtype="PCM_24")
        return True

    except Exception as e:
        print(f"    오류: {e}")
        return False


def main():
    import re

    pair_dirs = sorted([d for d in PAIRS_DIR.iterdir()
                        if d.is_dir() and (d / "acoustic.wav").exists()])

    # 중복 제거 (같은 노래 " 2" 등)
    seen, unique = set(), []
    for d in pair_dirs:
        base = re.sub(r'\s+\d+$', '', d.name)
        if base not in seen:
            seen.add(base); unique.append(d)

    print(f"총 {len(unique)}개 페어 처리 예정\n")

    done = skip = fail = 0
    for i, d in enumerate(unique):
        mix_path    = d / "mixture.wav"
        guitar_path = d / "guitar_demucs.wav"

        if guitar_path.exists():
            print(f"[{i+1}/{len(unique)}] 스킵 (이미 존재): {d.name[:50]}")
            skip += 1
            continue

        if not mix_path.exists():
            print(f"[{i+1}/{len(unique)}] 스킵 (mixture.wav 없음): {d.name[:50]}")
            skip += 1
            continue

        print(f"[{i+1}/{len(unique)}] 처리 중: {d.name[:50]}", flush=True)
        ok = run_demucs_on_mixture(mix_path, guitar_path)
        if ok:
            done += 1
            print(f"  ✓ 저장: {guitar_path.name}")
        else:
            fail += 1
            print(f"  ✗ 실패")

    print(f"\n완료: {done}개  스킵: {skip}개  실패: {fail}개")
    print(f"guitar_demucs.wav가 생성된 페어는 train_separator_v5.py에서 자동으로 사용됩니다.")


if __name__ == "__main__":
    main()
