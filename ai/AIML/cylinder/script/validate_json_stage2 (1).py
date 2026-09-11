#!/usr/bin/env python3
"""Validate Cylinder/Manta Stage-2 JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> list[str]:
    errors = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Invalid JSON: {exc}"]

    if data.get("schema") != "PS57-cylinder-manta-stage2-v1":
        errors.append("Missing or incorrect schema.")

    mapping = data.get("class_mapping")
    if mapping != {"0": "cylider", "1": "manta"}:
        errors.append("Class mapping must be 0=cylider, 1=manta.")

    results = data.get("results")
    if not isinstance(results, list):
        errors.append("'results' must be a list.")
        return errors

    for i, record in enumerate(results):
        prefix = f"results[{i}]"

        if "image" not in record:
            errors.append(f"{prefix}: missing image")

        if "detections" not in record:
            errors.append(f"{prefix}: missing detections")
            continue

        for j, det in enumerate(record["detections"]):
            p = f"{prefix}.detections[{j}]"

            for key in ("class_id", "class_name", "confidence", "bbox", "crop"):
                if key not in det:
                    errors.append(f"{p}: missing {key}")

            if det.get("class_id") not in (0, 1):
                errors.append(f"{p}: class_id must be 0 or 1")

            if isinstance(det.get("bbox"), list) and len(det["bbox"]) != 4:
                errors.append(f"{p}.bbox must contain 4 values")

            confidence = det.get("confidence")
            if isinstance(confidence, (int, float)):
                if not 0 <= confidence <= 1:
                    errors.append(f"{p}.confidence outside [0,1]")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    errors = validate(Path(args.json_file))

    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("VALID: PS57 Cylinder/Manta Stage-2 JSON")


if __name__ == "__main__":
    main()
