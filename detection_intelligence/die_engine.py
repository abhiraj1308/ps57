from typing import Any, List
import json
from .schemas import (
    SSSAnomalyDetection,
    PS57Detection,
    BoundingBox
)
import uuid
import datetime

# DIE (Detection Intelligence Engineering) Logic

def calculate_intersection_area(box: BoundingBox, polygon_normalized: List[List[float]], img_width: int, img_height: int) -> float:
    """
    Calculates approximate intersection area between a bounding box and a polygon.
    Since we only have the polygon points and doing full polygon intersection in python without shapely
    is complex, we'll do a simple bounding box overlap check. We find the bounding box of the polygon.
    """
    if not polygon_normalized:
        return 0.0

    poly_x_min = min(p[0] for p in polygon_normalized) * img_width
    poly_x_max = max(p[0] for p in polygon_normalized) * img_width
    poly_y_min = min(p[1] for p in polygon_normalized) * img_height
    poly_y_max = max(p[1] for p in polygon_normalized) * img_height

    box_x_min = box.x
    box_x_max = box.x + box.width
    box_y_min = box.y
    box_y_max = box.y + box.height

    # Intersection
    inter_x_min = max(poly_x_min, box_x_min)
    inter_x_max = min(poly_x_max, box_x_max)
    inter_y_min = max(poly_y_min, box_y_min)
    inter_y_max = min(poly_y_max, box_y_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)


def filter_and_refine_detections(ai_json_output: dict) -> List[PS57Detection]:
    """
    Parses the AI output, extracts detections and segmentation polygons,
    filters false positives (e.g., overlapping with rock),
    and returns PS57Detection objects.
    """
    final_detections = []
    
    for frame in ai_json_output.get("frames", []):
        img_w = frame.get("image_width", 1280)
        img_h = frame.get("image_height", 1280)
        
        models = frame.get("models", {})
        
        # Extract segmentation polygons
        seafloor = models.get("seafloor", {})
        segmentation = seafloor.get("segmentation", {})
        polygons = segmentation.get("polygons", [])
        
        rock_polygons = [p["polygon_normalized"] for p in polygons if p["class_name"] == "rock"]
        sand_polygons = [p["polygon_normalized"] for p in polygons if p["class_name"] == "sand"]

        # Collect detections
        for model_name, model_data in models.items():
            if model_name == "seafloor":
                continue
            
            for det in model_data.get("detections", []):
                # bbox_xyxy_pixel format: [x1, y1, x2, y2]
                bbox_px = det.get("bbox_xyxy_pixel")
                if not bbox_px:
                    continue
                
                box = BoundingBox(
                    x=bbox_px[0],
                    y=bbox_px[1],
                    width=bbox_px[2] - bbox_px[0],
                    height=bbox_px[3] - bbox_px[1]
                )
                
                box_area = box.width * box.height
                ai_conf = det.get("ai_confidence", 0.0)
                
                # Check overlap with rock and sand
                rock_overlap = sum(calculate_intersection_area(box, rp, img_w, img_h) for rp in rock_polygons)
                sand_overlap = sum(calculate_intersection_area(box, sp, img_w, img_h) for sp in sand_polygons)
                
                rock_ratio = rock_overlap / box_area if box_area > 0 else 0
                sand_ratio = sand_overlap / box_area if box_area > 0 else 0
                
                # Refine confidence based on rules
                final_conf = ai_conf
                
                if rock_ratio > 0.5:
                    final_conf *= 0.5  # Penalize heavy rock overlap (likely false positive)
                elif sand_ratio > 0.3:
                    final_conf = min(1.0, final_conf * 1.2)  # Boost confidence if on sand
                
                # Filter out low confidence
                if final_conf < 0.15:
                    continue
                    
                # Determine severity
                severity = "medium"
                if final_conf > 0.6:
                    severity = "high"
                elif final_conf < 0.3:
                    severity = "low"
                    
                # Decision
                decision = "accept" if final_conf >= 0.25 else "review"
                
                # Construct PS57Detection
                detection = PS57Detection(
                    class_name=det.get("class_name", "unknown"),
                    confidence=final_conf,
                    bbox=box,
                    severity=severity,
                    status="new",
                    decision=decision,
                    image_id=frame.get("frame_id"),
                    timestamp=datetime.datetime.now().isoformat()
                )
                final_detections.append(detection)
                
    return final_detections
