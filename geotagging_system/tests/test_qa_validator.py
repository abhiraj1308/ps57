"""Tests for QA/QC validator."""
import pytest
from datetime import datetime, timezone, timedelta
from src.reporting.qa_validator import QAValidator
from src.models import (
    PingMetadata, GeolocatedDetection, Geolocation, Dimensions, QualityFlag
)

@pytest.fixture
def qa_validator():
    return QAValidator({
        'max_vessel_speed_knots': 30.0,
        'max_detection_aspect_ratio': 20.0,
        'min_confidence': 0.6,
        'max_position_uncertainty_m': 50.0,
        'reject_land_detections': True
    })

@pytest.fixture
def land_detection():
    """Detection clearly on land (Kansas, USA)."""
    return GeolocatedDetection(
        detection_id="LAND_001",
        timestamp=datetime.now(timezone.utc),
        classification="debris",
        confidence_score=0.9,
        bounding_box_pixels=[100, 200, 300, 400],
        geolocation=Geolocation(latitude=38.5, longitude=-98.0, depth_meters=10.0, uncertainty_meters=2.0),
        dimensions_meters=Dimensions(length_m=5.0, width_m=2.0, area_sq_meters=10.0),
        ping_range=[100, 110],
        quality_flags=[QualityFlag.GPS_VALID]
    )

@pytest.fixture
def ocean_detection():
    """Detection in ocean (Gulf of Mexico)."""
    return GeolocatedDetection(
        detection_id="OCEAN_001",
        timestamp=datetime.now(timezone.utc),
        classification="ghost_net",
        confidence_score=0.85,
        bounding_box_pixels=[100, 200, 300, 400],
        geolocation=Geolocation(latitude=27.8, longitude=-82.5, depth_meters=15.0, uncertainty_meters=2.0),
        dimensions_meters=Dimensions(length_m=5.0, width_m=2.0, area_sq_meters=10.0),
        ping_range=[100, 110],
        quality_flags=[QualityFlag.GPS_VALID]
    )

class TestQAValidator:
    def test_confidence_threshold(self, qa_validator):
        low_conf = GeolocatedDetection(
            detection_id="LOW_001",
            timestamp=datetime.now(timezone.utc),
            classification="debris",
            confidence_score=0.3,
            bounding_box_pixels=[100, 200, 300, 400],
            geolocation=Geolocation(latitude=27.8, longitude=-82.5, depth_meters=15.0, uncertainty_meters=2.0),
            dimensions_meters=Dimensions(length_m=5.0, width_m=2.0, area_sq_meters=10.0),
            ping_range=[100, 110],
            quality_flags=[QualityFlag.GPS_VALID]
        )
        result = qa_validator.check_confidence_threshold([low_conf])
        assert len(result['flagged']) > 0
    
    def test_aspect_ratio_extreme(self, qa_validator):
        extreme = GeolocatedDetection(
            detection_id="EXT_001",
            timestamp=datetime.now(timezone.utc),
            classification="pipe",
            confidence_score=0.9,
            bounding_box_pixels=[100, 200, 300, 400],
            geolocation=Geolocation(latitude=27.8, longitude=-82.5, depth_meters=15.0, uncertainty_meters=2.0),
            dimensions_meters=Dimensions(length_m=100.0, width_m=1.0, area_sq_meters=100.0),
            ping_range=[100, 110],
            quality_flags=[QualityFlag.GPS_VALID]
        )
        result = qa_validator.check_aspect_ratios([extreme])
        assert len(result['flagged']) > 0
    
    def test_gps_jump_detection(self, qa_validator):
        """Two pings 0.1s apart but 1km away = impossible speed."""
        pings = [
            PingMetadata(
                ping_number=0, timestamp=datetime(2024,1,15,10,0,0, tzinfo=timezone.utc),
                latitude=27.80, longitude=-82.50, heading_deg=45.0,
                altitude_m=15.0, depth_m=2.0, slant_range_m=150.0
            ),
            PingMetadata(
                ping_number=1, timestamp=datetime(2024,1,15,10,0,0,100000, tzinfo=timezone.utc),
                latitude=27.81, longitude=-82.50, heading_deg=45.0,
                altitude_m=15.0, depth_m=2.0, slant_range_m=150.0
            ),
        ]
        result = qa_validator.check_gps_jumps(pings)
        assert len(result['flagged']) > 0
    
    def test_coordinate_bounds_flags_offtrack_detection(self, qa_validator):
        """A detection far from every ping in its own ping_range should be flagged."""
        pings = [
            PingMetadata(
                ping_number=i,
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i),
                latitude=27.80, longitude=-82.50, heading_deg=45.0,
                altitude_m=15.0, depth_m=2.0, slant_range_m=150.0
            )
            for i in range(100, 111)
        ]
        offtrack = GeolocatedDetection(
            detection_id="OFFTRACK_001",
            timestamp=datetime.now(timezone.utc),
            classification="debris",
            confidence_score=0.9,
            bounding_box_pixels=[100, 200, 300, 400],
            # ~1 degree of latitude away (~111km) from the track above
            geolocation=Geolocation(latitude=28.80, longitude=-82.50, depth_meters=10.0, uncertainty_meters=2.0),
            dimensions_meters=Dimensions(length_m=5.0, width_m=2.0, area_sq_meters=10.0),
            ping_range=[100, 110],
            quality_flags=[QualityFlag.GPS_VALID]
        )
        result = qa_validator.check_coordinate_bounds([offtrack], pings)
        assert "OFFTRACK_001" in result['flagged']
        assert result['pass'] is False

    def test_coordinate_bounds_passes_ontrack_detection(self, qa_validator, ocean_detection):
        pings = [
            PingMetadata(
                ping_number=i,
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i),
                latitude=27.80 + i * 0.00001, longitude=-82.50, heading_deg=45.0,
                altitude_m=15.0, depth_m=2.0, slant_range_m=150.0
            )
            for i in range(100, 111)
        ]
        result = qa_validator.check_coordinate_bounds([ocean_detection], pings)
        assert result['pass'] is True

    def test_coordinate_bounds_skips_without_track(self, qa_validator, ocean_detection):
        result = qa_validator.check_coordinate_bounds([ocean_detection], None)
        assert result['pass'] is True
        assert 'skipped' in result['details'].lower()

    def test_full_validation(self, qa_validator, ocean_detection):
        pings = [
            PingMetadata(
                ping_number=i, timestamp=datetime(2024,1,15,10,0,i, tzinfo=timezone.utc),
                latitude=27.80 + i*0.00001, longitude=-82.50, heading_deg=45.0,
                altitude_m=15.0, depth_m=2.0, slant_range_m=150.0
            )
            for i in range(10)
        ]
        report = qa_validator.validate(pings, [ocean_detection])
        assert 'overall_pass' in report
        assert 'rules' in report
