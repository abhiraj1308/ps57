#!/usr/bin/env python3
"""Validate PS57 Stage-2 JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return [f"Invalid JSON: {exc}"]

    if data.get("schema") != "PS57-stage2-v1":
        errors.append("Missing or incorrect schema.")

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

        if not isinstance(record["detections"], list):
            errors.append(f"{prefix}.detections must be a list")
            continue

        for j, det in enumerate(record["detections"]):
            p = f"{prefix}.detections[{j}]"
            for key in ("class_id", "class_name", "confidence", "bbox", "crop"):
                if key not in det:
                    errors.append(f"{p}: missing {key}")

            bbox = det.get("bbox")
            if isinstance(bbox, list) and len(bbox) != 4:
                errors.append(f"{p}.bbox must contain 4 values")

            conf = det.get("confidence")
            if isinstance(conf, (int, float)) and not 0 <= conf <= 1:
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

    print("VALID: PS57 Stage-2 JSON")


if __name__ == "__main__":
    main()
