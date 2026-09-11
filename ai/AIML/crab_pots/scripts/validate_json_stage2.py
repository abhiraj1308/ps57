import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

frames = data if isinstance(data, list) else data.get("frames", [])

errors = []
detections = 0

for frame in frames:
    w = frame.get("image_width")
    h = frame.get("image_height")

    if not w or not h:
        errors.append(f"{frame.get('frame_id')}: missing image dimensions")
        continue

    for det in frame.get("detections", []):
        detections += 1
        n = det.get("bbox_xywh_normalized")
        p = det.get("bbox_xyxy_pixel")

        if not n or not p or len(n) != 4 or len(p) != 4:
            errors.append(f"{frame.get('frame_id')}: invalid bbox")
            continue

        xc, yc, bw, bh = n
        x1, y1, x2, y2 = p

        if not all(0 <= v <= 1 for v in n):
            errors.append(f"{frame.get('frame_id')}: normalized bbox out of range")

        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            errors.append(f"{frame.get('frame_id')}: pixel bbox outside image")

        ex1 = (xc - bw / 2) * w
        ey1 = (yc - bh / 2) * h
        ex2 = (xc + bw / 2) * w
        ey2 = (yc + bh / 2) * h

        if max(abs(a - b) for a, b in zip((x1, y1, x2, y2),
                                           (ex1, ey1, ex2, ey2))) > 1:
            errors.append(f"{frame.get('frame_id')}: coordinate mismatch")

print("\nSTAGE-2 JSON VALIDATION")
print("=" * 40)
print(f"Frames checked       : {len(frames)}")
print(f"Detections checked   : {detections}")
print(f"Errors found         : {len(errors)}")

if errors:
    print("\nERRORS:")
    for error in errors[:20]:
        print("-", error)
    sys.exit(1)

print("\nPASS: JSON coordinate and boundary checks are valid.")