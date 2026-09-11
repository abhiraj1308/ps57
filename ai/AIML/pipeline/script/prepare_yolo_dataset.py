#!/usr/bin/env python3
"""Prepare and validate a YOLO detection dataset.

The script copies images and labels into:

output/
    train/
        images/
        labels/
    valid/
        images/
        labels/
    test/
        images/
        labels/

It preserves YOLO label contents and reports missing/malformed pairs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_split(root: Path, split: str) -> tuple[Path, Path]:
    candidates = [split]
    if split == "valid":
        candidates.append("val")

    for name in candidates:
        base = root / name
        for image_dir in (base / "images", base):
            label_dir = base / "labels"
            if image_dir.exists() and label_dir.exists():
                return image_dir, label_dir

    raise FileNotFoundError(f"Could not find {split} images/labels under {root}")


def validate_label(path: Path, nc: int | None) -> list[str]:
    errors = []
    try:
        lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    except Exception as exc:
        return [f"read error: {exc}"]

    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {n}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"line {n}: non-numeric value")
            continue

        if nc is not None and not 0 <= cls < nc:
            errors.append(f"line {n}: class {cls} outside [0,{nc-1}]")
        if any(v < 0 or v > 1 for v in vals):
            errors.append(f"line {n}: normalized coordinates outside [0,1]")

    return errors


def prepare_split(root: Path, output: Path, split: str, nc: int | None) -> tuple[int, int, int]:
    src_images, src_labels = find_split(root, split)
    dst_split = "valid" if split in {"val", "valid"} else split
    dst_images = output / dst_split / "images"
    dst_labels = output / dst_split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = missing = malformed = 0

    for image in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label = src_labels / f"{image.stem}.txt"
        if not label.exists():
            missing += 1
            continue

        errors = validate_label(label, nc)
        if errors:
            malformed += 1
            print(f"[WARN] {label}:")
            for error in errors:
                print(f"       {error}")
            continue

        shutil.copy2(image, dst_images / image.name)
        shutil.copy2(label, dst_labels / label.name)
        copied += 1

    return copied, missing, malformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Existing YOLO dataset root")
    parser.add_argument("--output", required=True, help="Prepared dataset root")
    parser.add_argument("--nc", type=int, default=None, help="Number of classes")
    args = parser.parse_args()

    root = Path(args.input)
    output = Path(args.output)

    totals = [0, 0, 0]
    for split in ("train", "valid", "test"):
        try:
            result = prepare_split(root, output, split, args.nc)
        except FileNotFoundError as exc:
            print(f"[INFO] {exc}")
            continue

        print(
            f"{split:>5}: copied={result[0]} "
            f"missing_labels={result[1]} malformed={result[2]}"
        )
        totals = [a + b for a, b in zip(totals, result)]

    print(
        f"\nTOTAL: copied={totals[0]}, "
        f"missing_labels={totals[1]}, malformed={totals[2]}"
    )


if __name__ == "__main__":
    main()
