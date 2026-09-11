"""Generate the demo_workflow.ipynb notebook."""
import json
from pathlib import Path

cells = []

def md(src):
    lines = src.split("\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})

def code(src):
    lines = src.split("\n")
    cells.append({"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "outputs": [], "execution_count": None})


md("""# Marine Debris Geotagging System - Demo Workflow

This notebook demonstrates the complete end-to-end pipeline:
1. Load/parse sonar metadata
2. Create AI detections
3. Run geotagging engine (pixel -> geographic coords)
4. Generate JSON and CSV reports
5. Inspect results""")

md("## 1. Setup and Imports")
code("""import sys, json, math, csv, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import random

sys.path.insert(0, str(Path('.').resolve()))

from src.models import (
    PingMetadata, Detection, SonarImageInfo, SonarSystemConfig,
    MissionMetadata, QualityFlag,
)
from src.pipeline.geotagging_engine import GeotaggingEngine
from src.reporting.report_generator import ReportGenerator
from src.reporting.qa_validator import QAValidator

print("Imports successful!")""")

md("""## 2. Generate Synthetic SSS Survey Track

Simulate a Klein 3000 side-scan sonar survey in Tampa Bay, FL.
- Vessel speed: ~4 knots, heading NE at 045 deg
- Ping rate: 10 Hz, towfish depth: 2m, altitude: 15m
- Cable out: 50m (towed system)""")

code("""pings = []
base_lat, base_lon = 27.80, -82.50
base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
speed_ms = 2.06  # ~4 knots
ping_interval = 0.1
dist_per_ping = speed_ms * ping_interval

lat, lon = base_lat, base_lon
heading = 45.0
random.seed(0)

for i in range(1000):
    dlat = dist_per_ping * math.cos(math.radians(heading)) / 111320.0
    dlon = dist_per_ping * math.sin(math.radians(heading)) / (111320.0 * math.cos(math.radians(lat)))
    lat += dlat
    lon += dlon
    ping = PingMetadata(
        ping_number=i,
        timestamp=base_time + timedelta(seconds=i * ping_interval),
        latitude=lat, longitude=lon,
        heading_deg=heading + random.gauss(0, 0.3),
        speed_knots=4.0, altitude_m=15.0,
        depth_m=2.0, slant_range_m=150.0,
        swath_width_m=300.0, cable_out_m=50.0,
        num_samples=2048,
        quality_flags=[QualityFlag.GPS_VALID, QualityFlag.ALTITUDE_VALID],
    )
    pings.append(ping)

print(f"Generated {len(pings)} pings")
print(f"Track start: ({pings[0].latitude:.5f}, {pings[0].longitude:.5f})")
print(f"Track end:   ({pings[-1].latitude:.5f}, {pings[-1].longitude:.5f})")""")

md("""## 3. Simulate AI Detections

Create 10 marine debris detections with pixel bounding boxes.
These represent the output from the AI detection model.""")

code("""random.seed(42)
detections = []
class_labels = ["ghost_net", "pipe", "tire", "debris", "wreck"]

for i in range(10):
    ping_start = random.randint(100, 800)
    ping_end = ping_start + random.randint(10, 30)
    x_center = random.randint(400, 1700)
    y_center = (ping_start + ping_end) // 2
    w_px = random.randint(30, 100)
    h_px = random.randint(10, 40)

    det = Detection(
        detection_id=f"DET_{i:03d}",
        bbox_pixels=[x_center - w_px//2, y_center - h_px//2,
                     x_center + w_px//2, y_center + h_px//2],
        confidence=round(random.uniform(0.6, 0.95), 2),
        class_label=random.choice(class_labels),
        ping_range=[ping_start, ping_end],
    )
    detections.append(det)

print(f"Created {len(detections)} detections")
for d in detections:
    print(f"  {d.detection_id}: {d.class_label} (conf={d.confidence:.2f}), bbox={d.bbox_pixels}")""")

md("## 4. Configure Sonar Image Info")
code("""image_info = SonarImageInfo(
    width_pixels=4096,
    height_pixels=1000,
    samples_per_ping=2048,
    range_m=150.0,
)

print(f"Image: {image_info.width_pixels} x {image_info.height_pixels} pixels")
print(f"Samples per ping: {image_info.samples_per_ping}")
print(f"Max range: {image_info.range_m} m")
print(f"Across-track resolution: {image_info.range_m / image_info.samples_per_ping:.4f} m/pixel")""")

md("""## 5. Run the Geotagging Engine

The engine performs layback correction, slant-to-ground range conversion,
pixel-to-geographic coordinate transformation, dimension calculation,
and positional uncertainty estimation.""")

code("""engine = GeotaggingEngine(config_path="config.yaml", sonar_system="klein_3000")
engine.load_pings_directly(pings)
results = engine.process_detections(detections, image_info, merge_overlapping=False)

print(f"Geolocated {len(results)} detections:\\n")
for gd in results:
    flags = ", ".join(f.value for f in gd.quality_flags)
    print(f"{gd.detection_id}: {gd.classification}")
    print(f"  Location: ({gd.geolocation.latitude:.6f}, {gd.geolocation.longitude:.6f})")
    print(f"  Depth: {gd.geolocation.depth_meters:.1f} m, Uncertainty: {gd.geolocation.uncertainty_meters:.2f} m")
    print(f"  Dimensions: {gd.dimensions_meters.length_m:.2f} x {gd.dimensions_meters.width_m:.2f} m")
    print(f"  Quality: [{flags}]")
    print()""")

md("## 6. Generate Reports")
code("""mission = MissionMetadata(
    survey_id="SSS_2024_001", date="2024-01-15",
    vessel="RV_Surveyor", sonar_system="Klein_System_3000",
    area_name="Tampa Bay, FL",
)

report = engine.generate_report(mission)
print(f"Survey: {report.mission_metadata.survey_id}")
print(f"Total detections: {report.summary.total_detections}")
print(f"By class: {report.summary.detections_by_class}")
print(f"Avg confidence: {report.summary.avg_confidence:.3f}")

output_dir = Path("data/sample_output")
output_dir.mkdir(parents=True, exist_ok=True)
gen = ReportGenerator(output_dir=str(output_dir))
json_path = gen.generate_json(report, "demo_report.json")
csv_path = gen.generate_csv(report, "demo_report.csv")
print(f"\\nJSON report: {json_path}")
print(f"CSV report:  {csv_path}")""")

md("## 7. Inspect JSON Report")
code("""with open(json_path) as f:
    jdata = json.load(f)

print(json.dumps(jdata["detections"][0], indent=2))""")

md("## 8. Inspect CSV Report")
code("""import pandas as pd
df = pd.read_csv(csv_path)
print(df.to_string(index=False))""")

md("## 9. QA Validation")
code("""qa = QAValidator()
qa_results = qa.validate(pings, results)
print(f"Overall pass: {qa_results['overall_pass']}")
print(f"Total flagged: {qa_results['total_flagged']}")
for rule_name, rule_result in qa_results.get("rules", {}).items():
    status = "PASS" if rule_result["pass"] else "FAIL"
    print(f"  [{status}] {rule_name}: {rule_result['details']}")""")

md("## 10. GeoJSON Export")
code("""geojson = engine.get_geojson()
print(f"Type: {geojson['type']}")
print(f"Features: {len(geojson['features'])}")
print("\\nFirst feature:")
print(json.dumps(geojson["features"][0], indent=2))""")

md("""---
## Summary

| Stage | Input | Output |
|---|---|---|
| Parse | SSS metadata CSV | `List[PingMetadata]` |
| Detect | Sonar image | `List[Detection]` (pixel bbox) |
| Geotag | Pings + Detections | `List[GeolocatedDetection]` (lat/lon) |
| Report | Survey report | JSON, CSV, GeoJSON, KML |
| QA | Pings + Detections | Validation flags |""")

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.7"},
    },
    "cells": cells,
}

Path("notebooks").mkdir(exist_ok=True)
with open("notebooks/demo_workflow.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Created: notebooks/demo_workflow.ipynb")
