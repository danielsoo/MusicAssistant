#!/bin/bash
set -euo pipefail

# Check ffmpeg (required by Demucs)
if ! command -v ffmpeg &>/dev/null; then
  echo "ffmpeg 없음. 설치 중..."
  brew install ffmpeg
fi

cd "$(dirname "$0")"

# Create venv if it doesn't exist (use Python 3.11 — torchaudio/torchcodec require it)
if [ ! -x ".venv/bin/python3" ]; then
  echo "가상환경 생성 중 (Python 3.11)..."
  /opt/homebrew/bin/python3.11 -m venv .venv
fi

VENV_PY="$(pwd)/.venv/bin/python3"
REQ_FILE="backend/requirements.txt"
REQ_HASH_FILE=".venv/.requirements.sha256"

calc_req_hash() {
  shasum -a 256 "$REQ_FILE" | awk '{print $1}'
}

needs_install() {
  if [ "${FORCE_PIP_INSTALL:-0}" = "1" ]; then
    return 0
  fi
  if [ ! -f "$REQ_HASH_FILE" ]; then
    return 0
  fi
  local current_hash
  current_hash="$(calc_req_hash)"
  local saved_hash
  saved_hash="$(cat "$REQ_HASH_FILE")"
  [ "$current_hash" != "$saved_hash" ]
}

if needs_install; then
  echo "패키지 설치 중..."
  if ! "$VENV_PY" -m pip install -r "$REQ_FILE"; then
    echo "의존성 설치 실패(setuptools 메타데이터 복구 시도)..."
    "$VENV_PY" -m pip install --ignore-installed --force-reinstall --no-deps "setuptools==82.0.0"
    "$VENV_PY" -m pip install -r "$REQ_FILE"
  fi
  calc_req_hash > "$REQ_HASH_FILE"
else
  echo "패키지 설치 생략 (requirements 변경 없음)"
fi

echo ""
echo "서버 시작: http://localhost:8000"
echo "Ctrl+C 로 종료"
echo ""

"$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir backend
