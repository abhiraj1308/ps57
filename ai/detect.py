import requests
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")

API_URL = "http://localhost:8000/detections"

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    payload = {
        "class_name": class_name,
        "confidence": confidence,
        "latitude": 19.0760,   # placeholder coords until real geodata exists
        "longitude": 72.8777,
        "width": x2 - x1,
        "height": y2 - y1,
        "status": "new",
        "priority": "high" if confidence > 0.9 else "low",
    }

    response = requests.post(API_URL, json=payload)
    print(response.status_code, response.json())