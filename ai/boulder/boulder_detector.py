from pathlib import Path
from typing import Optional

from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "best.pt"


def detect_boulders(
    input_image: str,
    output_path: Optional[str] = None,
    confidence: float = 0.25,
):
    """
    Detect boulders in a side-scan sonar image.

    Parameters
    ----------
    input_image : str
        Path to PNG/JPG/TIFF sonar image.

    output_path : str, optional
        Where to save the annotated output image.
        If omitted, no annotated image is saved.

    confidence : float
        Minimum confidence threshold.

    Returns
    -------
    list[dict]
        Detection results with class, confidence and bounding box.
    """

    input_path = Path(input_image)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(input_path),
        conf=confidence,
        save=False,
        verbose=False,
    )

    detections = []

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections.append(
                {
                    "class": "boulder",
                    "confidence": conf,
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                }
            )

    if output_path is not None:
        annotated = results[0].plot()
        import cv2

        cv2.imwrite(str(output_path), annotated)

    return detections


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Boulder detection module for side-scan sonar imagery."
    )

    parser.add_argument(
        "input_image",
        help="Path to input sonar image",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for annotated output image",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    args = parser.parse_args()

    detections = detect_boulders(
        args.input_image,
        args.output,
        args.conf,
    )

    print(json.dumps(detections, indent=2))