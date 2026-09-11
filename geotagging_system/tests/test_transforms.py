"""Tests for coordinate transformation engine."""
import math
import pytest
from src.transforms.coordinate_transformer import CoordinateTransformer
from src.transforms.projection_utils import (
    auto_detect_utm_epsg, geographic_distance, bearing_between,
    project_point_along_bearing, is_valid_coordinate
)

class TestProjectionUtils:
    def test_utm_zone_detection_tampa(self):
        epsg = auto_detect_utm_epsg(-82.5, 27.8)
        assert epsg == 32617  # UTM zone 17N
    
    def test_utm_zone_detection_london(self):
        epsg = auto_detect_utm_epsg(-0.1, 51.5)
        assert epsg in (32630, 32631)  # UTM zone 30N or 31N
    
    def test_utm_zone_southern_hemisphere(self):
        epsg = auto_detect_utm_epsg(151.2, -33.9)  # Sydney
        assert epsg == 32756  # UTM zone 56S
    
    def test_geographic_distance(self):
        # Distance from (0,0) to (0,1) should be ~111km
        dist = geographic_distance(0, 0, 1, 0)
        assert 111000 < dist < 112000
    
    def test_bearing_north(self):
        bearing = bearing_between(0, 0, 0, 1)
        assert abs(bearing - 0.0) < 1.0  # Should be ~0 (north)
    
    def test_bearing_east(self):
        bearing = bearing_between(0, 0, 1, 0)
        assert abs(bearing - 90.0) < 1.0  # Should be ~90 (east)
    
    def test_is_valid_coordinate(self):
        assert is_valid_coordinate(27.8, -82.5)
        assert not is_valid_coordinate(91.0, 0.0)
        assert not is_valid_coordinate(0.0, 181.0)

class TestCoordinateTransformer:
    def test_ground_range_calculation(self, sample_sonar_config):
        t = CoordinateTransformer(sample_sonar_config)
        # Known case: slant=100, alt=30 -> ground = sqrt(10000-900) = sqrt(9100) ≈ 95.39
        gr = t.calculate_ground_range(100.0, 30.0)
        assert abs(gr - 95.394) < 0.01
    
    def test_ground_range_blind_zone(self, sample_sonar_config):
        t = CoordinateTransformer(sample_sonar_config)
        # Slant range < altitude -> blind zone
        gr = t.calculate_ground_range(10.0, 30.0)
        assert gr == 0.0
    
    def test_ground_range_equal(self, sample_sonar_config):
        t = CoordinateTransformer(sample_sonar_config)
        gr = t.calculate_ground_range(30.0, 30.0)
        assert gr == 0.0
    
    def test_layback_correction(self, sample_sonar_config):
        t = CoordinateTransformer(sample_sonar_config)
        # Ship heading north, fish should be south (behind)
        fish_lon, fish_lat = t.apply_layback_correction(
            ship_lon=-82.5, ship_lat=27.8,
            heading_deg=0.0,  # North
            cable_out_m=50.0,
            fish_depth_m=5.0
        )
        assert fish_lat < 27.8  # Fish is behind (south of) ship
        assert abs(fish_lon - (-82.5)) < 0.001  # Roughly same longitude
    
    def test_layback_heading_east(self, sample_sonar_config):
        t = CoordinateTransformer(sample_sonar_config)
        fish_lon, fish_lat = t.apply_layback_correction(
            ship_lon=-82.5, ship_lat=27.8,
            heading_deg=90.0,  # East
            cable_out_m=50.0,
            fish_depth_m=5.0
        )
        assert fish_lon < -82.5  # Fish is behind (west of) ship
    
    def test_pixel_to_geographic(self, sample_sonar_config, sample_image_info, synthetic_pings):
        t = CoordinateTransformer(sample_sonar_config)
        ping = synthetic_pings[500]
        lon, lat = t.transform_pixel_to_geographic(
            x_pixel=1024,  # nadir
            y_pixel=500,
            ping_metadata=ping,
            image_info=sample_image_info
        )
        # Result should be near the ping position (nadir pixel → fish position)
        from src.transforms.projection_utils import geographic_distance
        dist = geographic_distance(lon, lat, ping.longitude, ping.latitude)
        assert dist < 200  # Within 200m (considering layback)

    def test_detection_dimensions(self, sample_sonar_config, sample_image_info, synthetic_pings):
        t = CoordinateTransformer(sample_sonar_config)
        dims = t.calculate_detection_dimensions(
            bbox_pixels=[900, 490, 1100, 510],  # 200px wide, 20px tall
            ping_metadata=synthetic_pings[500],
            image_info=sample_image_info
        )
        assert dims.width_m > 0
        assert dims.length_m > 0
        assert dims.area_sq_meters > 0
