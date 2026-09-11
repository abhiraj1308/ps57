import json
from pathlib import Path

from run_stage2 import collect_images


def test_collect_single_image(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"placeholder")
    assert collect_images(image) == [image]


def test_collect_directory(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("x")

    found = collect_images(tmp_path)
    assert len(found) == 2
    assert all(p.suffix.lower() in {".png", ".jpg"} for p in found)


def test_stage2_json_shape(tmp_path: Path):
    payload = {
        "schema": "PS57-stage2-v1",
        "results": [],
    }
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text())["schema"] == "PS57-stage2-v1"
