#!/usr/bin/env python3
"""Validate and prepare a Cylinder/Manta YOLO dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_split(root: Path, split: str) -> tuple[Path, Path]:
    names = [split]
    if split == "valid":
        names.append("val")

    for name in names:
        base = root / name
        image_dir = base / "images"
        label_dir = base / "labels"
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir

    raise FileNotFoundError(f"Missing {split}/images and {split}/labels under {root}")


def validate_label(path: Path, nc: int = 2) -> list[str]:
    errors = []
    try:
        lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    except Exception as exc:
        return [f"read error: {exc}"]

    for line_no, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_no}: expected 5 fields, got {len(parts)}")
            continue

        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line {line_no}: non-numeric value")
            continue

        if not 0 <= cls < nc:
            errors.append(f"line {line_no}: invalid class {cls}")
        if any(v < 0 or v > 1 for v in vals):
            errors.append(f"line {line_no}: coordinates must be normalized to [0,1]")

    return errors


def prepare_split(root: Path, output: Path, split: str) -> tuple[int, int, int]:
    src_images, src_labels = find_split(root, split)
    dst_split = "valid" if split in {"valid", "val"} else split

    dst_images = output / dst_split / "images"
    dst_labels = output / dst_split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = missing = malformed = 0

    for image in sorted(src_images.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue

        label = src_labels / f"{image.stem}.txt"
        if not label.exists():
            missing += 1
            continue

        errors = validate_label(label)
        if errors:
            malformed += 1
            print(f"[WARN] {label}")
            for error in errors:
                print(f"       {error}")
            continue

        shutil.copy2(image, dst_images / image.name)
        shutil.copy2(label, dst_labels / label.name)
        copied += 1

    return copied, missing, malformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.input)
    output = Path(args.output)

    total = [0, 0, 0]
    for split in ("train", "valid", "test"):
        try:
            result = prepare_split(root, output, split)
        except FileNotFoundError as exc:
            print(f"[INFO] {exc}")
            continue

        print(
            f"{split:>5}: copied={result[0]} "
            f"missing_labels={result[1]} malformed={result[2]}"
        )
        total = [a + b for a, b in zip(total, result)]

    print(
        f"\nTOTAL: copied={total[0]}, "
        f"missing_labels={total[1]}, malformed={total[2]}"
    )


if __name__ == "__main__":
    main()
