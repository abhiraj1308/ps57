"""
PS57 — CYLINDER / MANTA YOLO TRAINING SCRIPT
Dataset: Cylinder2 v6
Input: 1280x1280
Classes: 0=cylider, 1=manta

Run:
    python train_cylinder.py

Requirements:
    pip install ultralytics
"""

from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "cylinder2_data.yaml"
MODEL = "yolo26n.pt"          # change to your desired YOLO26 checkpoint
IMAGE_SIZE = 1280
EPOCHS = 50
BATCH = 8
PROJECT = "runs/ps57_cylinder"
NAME = "cylinder2_v6"

def main():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Could not find {DATA_YAML}")

    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        pretrained=True,
        verbose=True,
    )

    print("\nTraining complete.")
    print(f"Results directory: {PROJECT}/{NAME}")
    print("Best model: runs/ps57_cylinder/cylinder2_v6/weights/best.pt")

if __name__ == "__main__":
    main()
