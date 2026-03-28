#!/usr/bin/env python3
"""
Train ``guitar_separator.pt`` (acoustic vs electric masks from a full mix).

Delegates to ``scripts/train_separator.py``. Prepare data first:

    python scripts/prepare_training_data.py

Then:

    python backend/train_guitar_separator.py

Or run the script path directly:

    python scripts/train_separator.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TRAIN = _ROOT / "scripts" / "train_separator.py"


def main() -> None:
    raise SystemExit(subprocess.call([sys.executable, str(_TRAIN), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
