#!/usr/bin/env python3
"""Visualize Cylinder/Manta Stage-2 detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for record in data.get("results", []):
        source = Path(record["source"])
        image = cv2.imread(str(source))

        if image is None:
            print(f"[WARN] Could not read {source}")
            continue

        for det in record.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            confidence = float(det["confidence"])
            class_name = det["class_name"]

            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        target = output / source.name
        cv2.imwrite(str(target), image)
        print(f"Saved: {target}")


if __name__ == "__main__":
    main()
