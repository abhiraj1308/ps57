"""
Phase 2: End-to-end pipeline validation script.

Traces the complete flow:
  SSS metadata → parsed pings → pixel detections → sonar coords →
  lat/lon → geolocated detection → JSON/CSV report

Validates data integrity at every stage.
"""
import sys, json, csv, math, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (
    PingMetadata, Detection, SonarImageInfo, SonarSystemConfig,
    MissionMetadata, GeolocatedDetection, Geolocation, Dimensions,
    QualityFlag, SurveyReport, DetectionSummary,
)
from src.parsers.csv_parser import CSVParser
from src.parsers.json_parser import JSONParser
from src.transforms.coordinate_transformer import CoordinateTransformer
from src.transforms.projection_utils import geographic_distance
from src.pipeline.geotagging_engine import GeotaggingEngine
from src.reporting.report_generator import ReportGenerator
from src.reporting.qa_validator import QAValidator

PASS = "[OK]"
FAIL = "[FAIL]"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        msg = f"  {FAIL} {label}: {detail}"
        print(msg)
        errors.append(msg)

# -----------------------------------------------------------------------
# Stage 1: Generate synthetic sonar metadata → CSV file
# -----------------------------------------------------------------------
print("\n=== Stage 1: Generate Synthetic SSS Metadata ===")

pings = []
base_lat, base_lon = 27.80, -82.50
base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
speed_ms = 2.06  # ~4 knots
ping_interval = 0.1
dist_per_ping = speed_ms * ping_interval
lat, lon = base_lat, base_lon
heading = 45.0

import random
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

check("Generated 1000 pings", len(pings) == 1000)
check("Ping 0 coords valid", -90 <= pings[0].latitude <= 90 and -180 <= pings[0].longitude <= 180)
check("Timestamps monotonic", all(pings[i].timestamp < pings[i+1].timestamp for i in range(len(pings)-1)))

# Write CSV for parser test
out_dir = Path("data/sample_output")
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "e2e_pings.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "ping_number","timestamp","latitude","longitude",
        "heading","speed_knots","altitude","depth","slant_range","cable_out"
    ])
    w.writeheader()
    for p in pings:
        w.writerow({
            "ping_number": p.ping_number, "timestamp": p.timestamp.isoformat(),
            "latitude": p.latitude, "longitude": p.longitude,
            "heading": p.heading_deg, "speed_knots": p.speed_knots,
            "altitude": p.altitude_m, "depth": p.depth_m,
            "slant_range": p.slant_range_m, "cable_out": p.cable_out_m,
        })
check("CSV written", csv_path.exists(), str(csv_path))

# -----------------------------------------------------------------------
# Stage 2: Parse CSV back → PingMetadata list
# -----------------------------------------------------------------------
print("\n=== Stage 2: Parse CSV -> PingMetadata ===")
parser = CSVParser()
parsed = parser.parse(str(csv_path))
check("Parsed ping count", len(parsed) == 1000, f"got {len(parsed)}")
check("Parsed ping has heading_deg", hasattr(parsed[0], 'heading_deg'))
check("Parsed ping has altitude_m", hasattr(parsed[0], 'altitude_m'))
check("Round-trip lat matches", abs(parsed[0].latitude - pings[0].latitude) < 1e-6,
      f"{parsed[0].latitude} vs {pings[0].latitude}")

# -----------------------------------------------------------------------
# Stage 3: AI detections (simulated)
# -----------------------------------------------------------------------
print("\n=== Stage 3: Simulate AI Detections ===")
random.seed(42)
detections = []
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
        class_label=random.choice(["ghost_net","pipe","tire","debris"]),
        ping_range=[ping_start, ping_end],
    )
    detections.append(det)

check("10 detections created", len(detections) == 10)
check("All bbox valid", all(d.bbox_pixels[0] < d.bbox_pixels[2] for d in detections))
check("All confidence in [0,1]", all(0 <= d.confidence <= 1 for d in detections))

image_info = SonarImageInfo(
    width_pixels=4096, height_pixels=1000,
    samples_per_ping=2048, range_m=150.0,
)

# -----------------------------------------------------------------------
# Stage 4: GeotaggingEngine → pixel-to-geographic
# -----------------------------------------------------------------------
print("\n=== Stage 4: GeotaggingEngine Processing ===")
engine = GeotaggingEngine(config_path="config.yaml", sonar_system="klein_3000")
engine.load_pings_directly(pings)
check("Engine loaded pings", len(engine.pings) == 1000)

results = engine.process_detections(detections, image_info, merge_overlapping=False)
check("All detections processed", len(results) == 10, f"got {len(results)}")

# Validate each geolocated detection
for gd in results:
    check(f"  {gd.detection_id} coords valid",
          -90 <= gd.geolocation.latitude <= 90 and -180 <= gd.geolocation.longitude <= 180,
          f"lat={gd.geolocation.latitude}, lon={gd.geolocation.longitude}")
    check(f"  {gd.detection_id} has classification",
          gd.classification in ["ghost_net","pipe","tire","debris","wreck"],
          gd.classification)
    check(f"  {gd.detection_id} dimensions positive",
          gd.dimensions_meters.width_m >= 0 and gd.dimensions_meters.length_m >= 0)
    check(f"  {gd.detection_id} uncertainty reasonable",
          0 < gd.geolocation.uncertainty_meters < 500,
          f"{gd.geolocation.uncertainty_meters}")

# Check that geolocated coords are near the survey track
sample = results[0]
nearest_ping = min(pings, key=lambda p: abs(p.ping_number - sample.ping_range[0]))
dist_to_track = geographic_distance(
    sample.geolocation.longitude, sample.geolocation.latitude,
    nearest_ping.longitude, nearest_ping.latitude,
)
check("Detection near survey track", dist_to_track < 500,
      f"dist={dist_to_track:.1f}m")

# -----------------------------------------------------------------------
# Stage 5: Report generation
# -----------------------------------------------------------------------
print("\n=== Stage 5: Report Generation ===")
mission = MissionMetadata(
    survey_id="E2E_TEST_001", date="2024-01-15",
    vessel="RV_Surveyor", sonar_system="Klein_System_3000",
)
report = engine.generate_report(mission)
check("Report type", isinstance(report, SurveyReport))
check("Report detection count", report.summary.total_detections == 10)
check("Report has by-class breakdown", len(report.summary.detections_by_class) > 0)
check("Report avg_confidence > 0", report.summary.avg_confidence > 0)

gen = ReportGenerator(output_dir=str(out_dir))
json_path = gen.generate_json(report, "e2e_report.json")
csv_path_out = gen.generate_csv(report, "e2e_report.csv")
check("JSON report written", json_path.exists())
check("CSV report written", csv_path_out.exists())

# Validate JSON content
with open(json_path) as f:
    jdata = json.load(f)
check("JSON has mission_metadata", "mission_metadata" in jdata)
check("JSON has detections array", isinstance(jdata.get("detections"), list))
check("JSON detection count matches", len(jdata["detections"]) == 10)
d0 = jdata["detections"][0]
check("JSON detection has geolocation", "geolocation" in d0)
check("JSON detection has dimensions", "dimensions_meters" in d0)
check("JSON detection has ping_range", "ping_range" in d0)
check("JSON detection has quality_flags", "quality_flags" in d0)

# Validate CSV content
import pandas as pd
df = pd.read_csv(csv_path_out)
check("CSV has expected columns", set(["detection_id","lat","lon","confidence","class"]).issubset(df.columns),
      str(df.columns.tolist()))
check("CSV row count", len(df) == 10)
check("CSV lat in range", df["lat"].between(-90, 90).all())
check("CSV lon in range", df["lon"].between(-180, 180).all())

# -----------------------------------------------------------------------
# Stage 6: QA Validation
# -----------------------------------------------------------------------
print("\n=== Stage 6: QA Validation ===")
qa = QAValidator()
qa_results = qa.validate(pings, results)
check("QA ran successfully", "overall_pass" in qa_results)
check("QA has rules", len(qa_results.get("rules", {})) > 0)
print(f"  QA overall_pass: {qa_results['overall_pass']}")
print(f"  QA total_flagged: {qa_results['total_flagged']}")

# -----------------------------------------------------------------------
# Stage 7: GeoJSON export
# -----------------------------------------------------------------------
print("\n=== Stage 7: GeoJSON Export ===")
geojson = engine.get_geojson()
check("GeoJSON is FeatureCollection", geojson.get("type") == "FeatureCollection")
check("GeoJSON feature count", len(geojson.get("features", [])) == 10)
f0 = geojson["features"][0]
check("Feature has Point geometry", f0["geometry"]["type"] == "Point")
check("Feature coords are [lon, lat]",
      -180 <= f0["geometry"]["coordinates"][0] <= 180 and
      -90 <= f0["geometry"]["coordinates"][1] <= 90)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "="*60)
if errors:
    print(f"PIPELINE VALIDATION: {FAIL} {len(errors)} error(s)")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print(f"PIPELINE VALIDATION: {PASS} All checks passed!")

# Print sample output
print("\n--- Sample JSON detection ---")
print(json.dumps(jdata["detections"][0], indent=2))

print("\n--- Sample CSV row ---")
print(df.iloc[0].to_dict())
sys.exit(0)
