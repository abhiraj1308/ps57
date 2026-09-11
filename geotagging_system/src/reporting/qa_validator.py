import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta

from src.models import (
    PingMetadata, GeolocatedDetection, QualityFlag, SurveyReport
)
from src.exceptions import ValidationError
from src.transforms.projection_utils import geographic_distance

class QAValidator:
    """
    Quality Assurance / Quality Control validation engine.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with validation thresholds from config."""
        cfg = config or {}
        self.max_speed_knots = cfg.get('max_vessel_speed_knots', 30.0)
        self.max_aspect_ratio = cfg.get('max_detection_aspect_ratio', 20.0)
        self.min_confidence = cfg.get('min_confidence', 0.6)
        self.max_uncertainty_m = cfg.get('max_position_uncertainty_m', 50.0)
        self.depth_tolerance = cfg.get('depth_consistency_tolerance', 0.20)
        self.reject_land = cfg.get('reject_land_detections', True)
        # Max plausible distance (m) between a geolocated detection and the
        # nearest ping in its own ping_range. A well-formed detection must
        # sit within roughly one swath width of the track that produced it;
        # anything farther indicates a transform error (e.g. detection
        # projected onto land or off the survey line), not an actual
        # land/sea classification. Avoids needing a bundled coastline dataset.
        self.max_offtrack_distance_m = cfg.get('max_offtrack_distance_m', 250.0)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(
        self,
        pings: List[PingMetadata],
        detections: List[GeolocatedDetection]
    ) -> Dict[str, Any]:
        """
        Run all validation rules and generate QA report.
        """
        from datetime import datetime, timezone
        
        bounds_res = self.check_coordinate_bounds(detections, pings) if self.reject_land else {'pass': True, 'flagged': [], 'details': 'Skipped'}
        jumps_res = self.check_gps_jumps(pings)
        aspect_res = self.check_aspect_ratios(detections)
        conf_res = self.check_confidence_threshold(detections)
        depth_res = self.check_depth_consistency(pings)
        uncert_res = self.check_positional_uncertainty(detections)
        
        all_rules = {
            'coordinate_bounds': bounds_res,
            'gps_jumps': jumps_res,
            'aspect_ratio': aspect_res,
            'confidence': conf_res,
            'depth_consistency': depth_res,
            'positional_uncertainty': uncert_res
        }
        
        overall_pass = all(res['pass'] for res in all_rules.values())
        
        # calculate totals based on detections only for simplicity
        flagged_set = set(bounds_res.get('flagged', [])) | \
                      set(aspect_res.get('flagged', [])) | \
                      set(conf_res.get('flagged', [])) | \
                      set(uncert_res.get('flagged', []))
                      
        return {
            'overall_pass': overall_pass,
            'rules': all_rules,
            'total_flagged': len(flagged_set),
            'total_passed': len(detections) - len(flagged_set),
            'validated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def check_coordinate_bounds(
        self,
        detections: List[GeolocatedDetection],
        pings: Optional[List[PingMetadata]] = None,
    ) -> Dict:
        """
        Flag detections that ended up implausibly far from the sonar track
        that produced them.

        This is a proxy for "is this detection actually in the water": a
        correctly geotagged detection must lie within roughly one swath
        width of the ping(s) in its own ping_range. If it's farther than
        that, either the pixel→geo transform went wrong or the detection
        is bogus — either way it's not trustworthy, whether or not it
        technically lands on a coastline.

        If no ping track is supplied, the check is skipped (reported as a
        pass) rather than silently claiming everything is fine.
        """
        if not pings:
            return {
                'pass': True,
                'flagged': [],
                'details': 'No ping track supplied; off-track check skipped',
            }

        ping_map: Dict[int, PingMetadata] = {p.ping_number: p for p in pings}
        sorted_numbers = sorted(ping_map.keys())

        flagged = []
        for d in detections:
            start, end = d.ping_range[0], d.ping_range[1]
            candidates = [ping_map[n] for n in range(start, end + 1) if n in ping_map]

            if not candidates:
                # Ping range doesn't overlap the loaded track (e.g. detection
                # references pings outside this file) — can't verify, don't flag.
                continue

            nearest_dist = min(
                geographic_distance(
                    d.geolocation.longitude, d.geolocation.latitude,
                    p.longitude, p.latitude,
                )
                for p in candidates
            )
            if nearest_dist > self.max_offtrack_distance_m:
                flagged.append(d.detection_id)

        return {
            'pass': len(flagged) == 0,
            'flagged': flagged,
            'details': f"{len(flagged)} detections farther than {self.max_offtrack_distance_m}m from their sonar track"
        }
    
    def check_gps_jumps(self, pings: List[PingMetadata]) -> Dict:
        """Check for unrealistic GPS jumps between consecutive pings."""
        flagged_pings = []
        if len(pings) < 2:
            return {'pass': True, 'flagged': [], 'details': 'Not enough pings to check'}
            
        for i in range(1, len(pings)):
            p1 = pings[i-1]
            p2 = pings[i]
            
            dist = geographic_distance(p1.longitude, p1.latitude, p2.longitude, p2.latitude)
            dt = (p2.timestamp - p1.timestamp).total_seconds()
            
            if dt > 0:
                speed_mps = dist / dt
                speed_knots = speed_mps * 1.94384
                if speed_knots > self.max_speed_knots:
                    flagged_pings.append(p2.ping_number)
                    
        return {
            'pass': len(flagged_pings) == 0,
            'flagged': flagged_pings,
            'details': f"{len(flagged_pings)} pings exceeded max speed {self.max_speed_knots} kts"
        }
    
    def check_aspect_ratios(self, detections: List[GeolocatedDetection]) -> Dict:
        """Flag detections with extreme aspect ratios."""
        flagged = []
        for d in detections:
            l, w = d.dimensions_meters.length_m, d.dimensions_meters.width_m
            if w > 0:
                ar1 = l / w
            else:
                ar1 = float('inf')
            if l > 0:
                ar2 = w / l
            else:
                ar2 = float('inf')
                
            if max(ar1, ar2) > self.max_aspect_ratio:
                flagged.append(d.detection_id)
                
        return {
            'pass': len(flagged) == 0,
            'flagged': flagged,
            'details': f"{len(flagged)} detections have extreme aspect ratios"
        }
    
    def check_confidence_threshold(self, detections: List[GeolocatedDetection]) -> Dict:
        """Flag detections below minimum confidence."""
        flagged = [d.detection_id for d in detections if d.confidence_score < self.min_confidence]
        return {
            'pass': len(flagged) == 0,
            'flagged': flagged,
            'details': f"{len(flagged)} detections below {self.min_confidence} confidence"
        }
    
    def check_depth_consistency(self, pings: List[PingMetadata]) -> Dict:
        """Flag pings where depth varies more than tolerance from rolling average."""
        flagged = []
        window = 10
        for i in range(len(pings)):
            start = max(0, i - window // 2)
            end = min(len(pings), i + window // 2)
            if end > start:
                avg_depth = sum(p.depth_m for p in pings[start:end]) / (end - start)
                if avg_depth > 0:
                    diff = abs(pings[i].depth_m - avg_depth) / avg_depth
                    if diff > self.depth_tolerance:
                        flagged.append(pings[i].ping_number)
                        
        return {
            'pass': len(flagged) == 0,
            'flagged': flagged,
            'details': f"{len(flagged)} pings have inconsistent depth"
        }
    
    def check_positional_uncertainty(self, detections: List[GeolocatedDetection]) -> Dict:
        """Flag detections with uncertainty > max_uncertainty_m."""
        flagged = [d.detection_id for d in detections if d.geolocation.uncertainty_meters > self.max_uncertainty_m]
        return {
            'pass': len(flagged) == 0,
            'flagged': flagged,
            'details': f"{len(flagged)} detections have high uncertainty"
        }
    

