"""Tests for geotagging pipeline."""
import pytest
from src.pipeline.geotagging_engine import GeotaggingEngine
from src.models import MissionMetadata

class TestGeotaggingEngine:
    def test_load_pings_directly(self, synthetic_pings, sample_sonar_config):
        engine = GeotaggingEngine.__new__(GeotaggingEngine)
        engine.logger = __import__('logging').getLogger('test')
        engine.config = {}
        engine.sonar_config = sample_sonar_config
        from src.transforms.coordinate_transformer import CoordinateTransformer
        engine.transformer = CoordinateTransformer(sample_sonar_config)
        engine.pings = []
        engine.geolocated_detections = []
        engine.merge_distance = 5.0
        
        engine.load_pings_directly(synthetic_pings[:100])
        assert len(engine.pings) == 100
    
    def test_process_detections(self, synthetic_pings, synthetic_detections, sample_image_info, sample_sonar_config):
        engine = GeotaggingEngine.__new__(GeotaggingEngine)
        engine.logger = __import__('logging').getLogger('test')
        engine.config = {}
        engine.sonar_config = sample_sonar_config
        from src.transforms.coordinate_transformer import CoordinateTransformer
        engine.transformer = CoordinateTransformer(sample_sonar_config)
        engine.pings = []
        engine.geolocated_detections = []
        engine.merge_distance = 5.0
        
        engine.load_pings_directly(synthetic_pings)
        # Use only first 10 detections for speed
        results = engine.process_detections(
            synthetic_detections[:10],
            sample_image_info,
            merge_overlapping=False
        )
        assert len(results) > 0
        # All results should have valid coordinates
        for det in results:
            assert -90 <= det.geolocation.latitude <= 90
            assert -180 <= det.geolocation.longitude <= 180
    
    def test_generate_report(self, synthetic_pings, synthetic_detections, sample_image_info, sample_sonar_config, sample_mission_metadata):
        engine = GeotaggingEngine.__new__(GeotaggingEngine)
        engine.logger = __import__('logging').getLogger('test')
        engine.config = {}
        engine.sonar_config = sample_sonar_config
        from src.transforms.coordinate_transformer import CoordinateTransformer
        engine.transformer = CoordinateTransformer(sample_sonar_config)
        engine.pings = []
        engine.geolocated_detections = []
        engine.merge_distance = 5.0
        
        engine.load_pings_directly(synthetic_pings)
        engine.process_detections(synthetic_detections[:5], sample_image_info, merge_overlapping=False)
        report = engine.generate_report(sample_mission_metadata)
        assert report.mission_metadata.survey_id == "SSS_2024_001"
        assert report.summary.total_detections > 0
    
    def test_get_geojson(self, synthetic_pings, synthetic_detections, sample_image_info, sample_sonar_config):
        engine = GeotaggingEngine.__new__(GeotaggingEngine)
        engine.logger = __import__('logging').getLogger('test')
        engine.config = {}
        engine.sonar_config = sample_sonar_config
        from src.transforms.coordinate_transformer import CoordinateTransformer
        engine.transformer = CoordinateTransformer(sample_sonar_config)
        engine.pings = []
        engine.geolocated_detections = []
        engine.merge_distance = 5.0
        
        engine.load_pings_directly(synthetic_pings)
        engine.process_detections(synthetic_detections[:5], sample_image_info, merge_overlapping=False)
        geojson = engine.get_geojson()
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) > 0
