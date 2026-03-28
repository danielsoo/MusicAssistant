#!/usr/bin/env python3
"""Print accumulated AI→your-label corrections from data/chord_learned_priors.json."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = root / "data" / "chord_learned_priors.json"
if not path.exists():
    print("No priors file yet — save edited chords once from the app.", file=sys.stderr)
    sys.exit(0)

data = json.loads(path.read_text(encoding="utf-8"))
corr = data.get("corrections") or {}
if not corr:
    print("Empty corrections map.")
    sys.exit(0)

print("Detection label → your preferred label (counts)\n")
for la in sorted(corr.keys()):
    targets = corr[la]
    total = sum(targets.values())
    ranked = sorted(targets.items(), key=lambda x: -x[1])
    print(f"  {la}  (n={total})")
    for lu, c in ranked[:5]:
        print(f"      → {lu}: {c} ({100 * c / total:.0f}%)")
    print()
