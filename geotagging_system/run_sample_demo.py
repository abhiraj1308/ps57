"""
Runs the full geotagging pipeline using the bundled sample files:
  - data/sample_input/sample_mission.json   (mission metadata)
  - data/sample_input/sample_detections.json (fake AI model output)

Nothing else in this repo actually reads those two files (they were
included as reference format examples only) — this script wires them
into a real run so you have one command that goes from "sample input"
to "JSON/CSV report on disk".

There's no real recorded sonar navigation log (XTF/CSV/NMEA) bundled
with the project, since that has to come from an actual survey or from
your teammates' side. This script generates a synthetic survey track
that covers the ping ranges referenced by sample_detections.json, so
you get a real end-to-end run today. Swap `build_synthetic_pings()`
for `engine.load_sonar_data("your_real_log.csv")` once you (or your
team) have an actual sonar log to point at.

Usage:
    python run_sample_demo.py
"""
import json
import math
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.models import (
    PingMetadata, Detection, SonarImageInfo, MissionMetadata, QualityFlag,
)
from src.pipeline.geotagging_engine import GeotaggingEngine
from src.reporting.report_generator import ReportGenerator
from src.reporting.qa_validator import QAValidator

SAMPLE_INPUT = Path("data/sample_input")
OUTPUT_DIR = Path("data/sample_output")


def build_synthetic_pings(max_ping_number: int) -> list[PingMetadata]:
    """Generate a plausible survey track long enough to cover every
    detection's ping_range. Replace this with a real parsed log
    (engine.load_sonar_data(...)) once you have one."""
    pings = []
    base_lat, base_lon = 27.80, -82.50
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    speed_ms = 2.06  # ~4 knots
    ping_interval = 0.1  # seconds
    dist_per_ping = speed_ms * ping_interval

    lat, lon = base_lat, base_lon
    heading = 45.0
    random.seed(0)

    for i in range(max_ping_number + 50):  # small buffer past the last detection
        dlat = dist_per_ping * math.cos(math.radians(heading)) / 111320.0
        dlon = dist_per_ping * math.sin(math.radians(heading)) / (
            111320.0 * math.cos(math.radians(lat))
        )
        lat += dlat
        lon += dlon

        pings.append(
            PingMetadata(
                ping_number=i,
                timestamp=base_time + timedelta(seconds=i * ping_interval),
                latitude=lat,
                longitude=lon,
                heading_deg=heading,
                speed_knots=4.0,
                altitude_m=15.0,
                depth_m=2.0,
                slant_range_m=150.0,
                cable_out_m=50.0,
                num_samples=2048,
                quality_flags=[QualityFlag.GPS_VALID, QualityFlag.ALTITUDE_VALID],
            )
        )
    return pings


def main():
    mission_data = json.loads((SAMPLE_INPUT / "sample_mission.json").read_text())
    mission = MissionMetadata(**mission_data)

    detections_data = json.loads((SAMPLE_INPUT / "sample_detections.json").read_text())
    detections = [Detection(**d) for d in detections_data["detections"]]

    max_ping = max(d.ping_range[1] for d in detections)
    pings = build_synthetic_pings(max_ping)

    image_info = SonarImageInfo(
        width_pixels=4096,
        height_pixels=max_ping + 50,
        samples_per_ping=2048,
        range_m=150.0,
    )

    engine = GeotaggingEngine(config_path="config.yaml", sonar_system="klein_3000")
    engine.load_pings_directly(pings)

    results = engine.process_detections(detections, image_info, merge_overlapping=False)
    print(f"Geolocated {len(results)} detections from {len(pings)} pings.\n")

    report = engine.generate_report(mission)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gen = ReportGenerator(output_dir=str(OUTPUT_DIR))
    json_path = gen.generate_json(report, "sample_demo_report.json")
    csv_path = gen.generate_csv(report, "sample_demo_report.csv")
    print(f"JSON report -> {json_path}")
    print(f"CSV report  -> {csv_path}")

    qa = QAValidator()
    qa_results = qa.validate(pings, results)
    print(f"\nQA overall_pass: {qa_results['overall_pass']}")
    print(f"QA total_flagged: {qa_results['total_flagged']}")

    print("\n--- Sample detection ---")
    d0 = results[0]
    print(f"  {d0.detection_id} ({d0.classification}, conf={d0.confidence_score})")
    print(f"  lat={d0.geolocation.latitude:.6f}, lon={d0.geolocation.longitude:.6f}")
    print(f"  size: {d0.dimensions_meters.length_m}m x {d0.dimensions_meters.width_m}m")


if __name__ == "__main__":
    main()