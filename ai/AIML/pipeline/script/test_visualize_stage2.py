import json
from pathlib import Path

import cv2
import numpy as np

from visualize_stage2 import main


def test_visualization_cli(tmp_path, monkeypatch):
    image_path = tmp_path / "test.png"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    payload = {
        "schema": "PS57-stage2-v1",
        "results": [{
            "image": "test.png",
            "source": str(image_path),
            "image_size": {"width": 100, "height": 100},
            "detections": [{
                "class_id": 0,
                "class_name": "Pipeline",
                "confidence": 0.9,
                "bbox": [10, 10, 50, 50],
                "crop": {},
            }],
        }],
    }

    json_path = tmp_path / "predictions.json"
    json_path.write_text(json.dumps(payload))
    output = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        ["visualize_stage2.py", "--json", str(json_path), "--output", str(output)],
    )
    main()

    assert (output / "test.png").exists()
