#!/usr/bin/env python3
"""Run the PS57 Stage-2 pipeline over one image or a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonar_inference import SonarInference

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="stage2_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--pad", type=int, default=20)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    engine = SonarInference(args.model, conf=args.conf, iou=args.iou)
    images = collect_images(source)

    records = []
    for index, image in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {image}")
        try:
            records.append(engine.predict_image(image, pad=args.pad))
        except Exception as exc:
            records.append({
                "image": image.name,
                "source": str(image),
                "error": str(exc),
                "detections": [],
            })

    payload = {
        "schema": "PS57-stage2-v1",
        "model": str(Path(args.model)),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "crop_padding": args.pad,
        "count": len(records),
        "results": records,
    }

    json_path = output / "predictions.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
