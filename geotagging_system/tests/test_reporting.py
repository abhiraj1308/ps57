"""Tests for report generation."""
import json
import pytest
from pathlib import Path
from src.reporting.report_generator import ReportGenerator
from src.models import (
    SurveyReport, MissionMetadata, DetectionSummary,
    GeolocatedDetection, Geolocation, Dimensions, QualityFlag
)
from datetime import datetime, timezone

@pytest.fixture
def sample_report():
    detections = [
        GeolocatedDetection(
            detection_id=f"DET_{i:03d}",
            timestamp=datetime(2024, 1, 15, 10, i, 0, tzinfo=timezone.utc),
            classification="ghost_net" if i % 2 == 0 else "pipe",
            confidence_score=0.85 + i * 0.01,
            bounding_box_pixels=[100, 200, 300, 400],
            geolocation=Geolocation(latitude=27.8 + i*0.001, longitude=-82.5 + i*0.001, depth_meters=15.0, uncertainty_meters=2.0),
            dimensions_meters=Dimensions(length_m=5.0, width_m=2.0, area_sq_meters=10.0),
            ping_range=[1000 + i*10, 1050 + i*10],
            quality_flags=[QualityFlag.GPS_VALID]
        )
        for i in range(5)
    ]
    return SurveyReport(
        mission_metadata=MissionMetadata(survey_id="TEST_001", date="2024-01-15", vessel="TestVessel"),
        detections=detections,
        summary=DetectionSummary(
            total_detections=5,
            detections_by_class={"ghost_net": 3, "pipe": 2},
            survey_area_sq_km=1.5,
            avg_confidence=0.87
        )
    )

class TestReportGenerator:
    def test_generate_json(self, sample_report, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.generate_json(sample_report, "test_report.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert 'mission_metadata' in data
        assert 'detections' in data
        assert len(data['detections']) == 5
    
    def test_generate_csv(self, sample_report, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.generate_csv(sample_report, "test_report.csv")
        assert path.exists()
        import csv as csv_mod
        with open(path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5
        assert 'detection_id' in rows[0]
        assert 'lat' in rows[0]
    
    def test_generate_geojson(self, sample_report, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.generate_geojson(sample_report, "test_report.geojson")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data['type'] == 'FeatureCollection'
        assert len(data['features']) == 5
    
    def test_generate_all(self, sample_report, tmp_path):
        gen = ReportGenerator(output_dir=str(tmp_path))
        paths = gen.generate_all(sample_report, "test_full", formats=['json', 'csv', 'geojson'])
        assert 'json' in paths
        assert 'csv' in paths
        assert 'geojson' in paths
