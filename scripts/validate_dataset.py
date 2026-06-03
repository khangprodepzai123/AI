"""Thống kê nhanh dataset trong data/raw/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.data_loader import collect_image_paths, discover_classes


def main() -> None:
    classes = discover_classes()
    if not classes:
        print(f"No classes in {config.DATA_DIR}")
        return

    paths, y, names = collect_image_paths()
    counts = {n: 0 for n in names}
    for label in y:
        counts[names[label]] += 1

    print(f"Data dir: {config.DATA_DIR}")
    print(f"Classes: {len(names)}")
    print(f"Total images: {len(paths)}")
    print("-" * 40)
    for n in names:
        print(f"  {n:12} {counts[n]:6}")
    print("-" * 40)
    mn, mx = min(counts.values()), max(counts.values())
    if mx > mn * 1.5:
        print("Note: imbalanced classes — training uses class_weight=balanced.")


if __name__ == "__main__":
    main()
