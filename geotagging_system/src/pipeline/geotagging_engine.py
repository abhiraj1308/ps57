"""
Main orchestration engine for the marine debris geotagging pipeline.

Coordinates sonar metadata parsing, detection geotagging, overlap merging,
and report generation.
"""
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np

from src.models import (
    PingMetadata, Detection, GeolocatedDetection, Geolocation, Dimensions,
    QualityFlag, SonarSystemConfig, SonarImageInfo, MissionMetadata,
    SurveyReport, DetectionSummary,
)
from src.exceptions import PipelineError
from src.transforms.coordinate_transformer import CoordinateTransformer
from src.transforms.projection_utils import geographic_distance
from src.parsers import get_parser


class GeotaggingEngine:
    """
    Main orchestration engine for the marine debris geotagging pipeline.

    Pipeline steps:
        1. Load configuration and select sonar system profile.
        2. Parse sonar metadata from an input file.
        3. Build a time-indexed ping lookup table.
        4. Process AI detections → geolocated detections.
        5. Merge overlapping detections from adjacent swaths.
        6. Generate a :class:`SurveyReport`.

    Example::

        >>> engine = GeotaggingEngine(config_path='config.yaml')
        >>> engine.load_sonar_data('survey.csv')
        >>> results = engine.process_detections(detections, image_info)
        >>> report = engine.generate_report(mission_metadata)
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        sonar_system: str = "klein_3000",
    ):
        """Load config, initialise transformer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        self.sonar_system_name = sonar_system

        sonar_cfg = self.config.get("sonar_systems", {}).get(sonar_system, {})
        if sonar_cfg:
            self.sonar_config = SonarSystemConfig(**sonar_cfg)
        else:
            self.sonar_config = SonarSystemConfig(
                name="Default",
                frequency_khz=445,
                max_range_m=150,
                beam_width_deg=0.2,
            )

        self.transformer = CoordinateTransformer(self.sonar_config)
        self.pings: List[PingMetadata] = []
        self.geolocated_detections: List[GeolocatedDetection] = []
        self.merge_distance = (
            self.config.get("pipeline", {}).get("merge_distance_m", 5.0)
        )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config, return empty dict if not found."""
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            self.logger.warning("Config file not found: %s. Using defaults.", config_path)
        except Exception as e:
            self.logger.error("Error loading config: %s. Using defaults.", e)
        return {}

    # ------------------------------------------------------------------
    # Ping loading
    # ------------------------------------------------------------------

    def load_sonar_data(self, filepath: str) -> int:
        """
        Parse sonar metadata file and build ping index.

        Auto-detects format (XTF, CSV, JSON, NMEA) via file extension.

        Returns:
            Number of pings loaded.
        """
        try:
            parser = get_parser(filepath)
            self.pings = parser.parse(filepath)
            self._build_ping_index()
            return len(self.pings)
        except Exception as e:
            raise PipelineError(
                detail=f"Failed to load sonar data from {filepath}: {e}",
                stage="load_sonar_data",
            )

    def load_pings_directly(self, pings: List[PingMetadata]) -> None:
        """Load pre-parsed ping metadata directly."""
        self.pings = sorted(pings, key=lambda p: p.ping_number)
        self._build_ping_index()

    def _build_ping_index(self) -> None:
        """Build sorted index for fast ping lookup by number."""
        self.pings.sort(key=lambda p: p.ping_number)
        self._ping_map: Dict[int, PingMetadata] = {
            p.ping_number: p for p in self.pings
        }

    # ------------------------------------------------------------------
    # Ping access
    # ------------------------------------------------------------------

    def get_ping_at_index(self, index: int) -> PingMetadata:
        """Get ping metadata by list index with bounds clamping."""
        if not self.pings:
            raise PipelineError(detail="No pings loaded", stage="get_ping")
        index = max(0, min(index, len(self.pings) - 1))
        return self.pings[index]

    def get_interpolated_ping(self, ping_number: float) -> PingMetadata:
        """
        Get interpolated ping metadata for fractional ping numbers.

        Linearly interpolates lat, lon, heading, altitude, and depth
        between the floor and ceil ping indices.
        """
        if not self.pings:
            raise PipelineError(detail="No pings loaded", stage="interpolate_ping")

        floor_idx = int(np.floor(ping_number))
        ceil_idx = int(np.ceil(ping_number))

        min_ping = self.pings[0].ping_number
        max_ping = self.pings[-1].ping_number

        if floor_idx <= min_ping:
            return self._ping_map.get(min_ping, self.pings[0])
        if ceil_idx >= max_ping:
            return self._ping_map.get(max_ping, self.pings[-1])

        ping1 = self._ping_map.get(floor_idx)
        ping2 = self._ping_map.get(ceil_idx)

        if not ping1 or not ping2:
            # Fall back to nearest available ping
            idx = min(
                max(floor_idx - min_ping, 0), len(self.pings) - 1
            )
            return self.pings[idx]

        if floor_idx == ceil_idx:
            return ping1

        w2 = ping_number - floor_idx
        w1 = 1.0 - w2

        # Circular heading interpolation
        h1, h2 = ping1.heading_deg, ping2.heading_deg
        diff = h2 - h1
        if diff > 180:
            h1 += 360
        elif diff < -180:
            h2 += 360
        interp_heading = (h1 * w1 + h2 * w2) % 360

        interp_ts = ping1.timestamp + (ping2.timestamp - ping1.timestamp) * w2

        return PingMetadata(
            ping_number=int(ping_number),
            timestamp=interp_ts,
            latitude=ping1.latitude * w1 + ping2.latitude * w2,
            longitude=ping1.longitude * w1 + ping2.longitude * w2,
            heading_deg=interp_heading,
            speed_knots=ping1.speed_knots * w1 + ping2.speed_knots * w2,
            altitude_m=ping1.altitude_m * w1 + ping2.altitude_m * w2,
            depth_m=ping1.depth_m * w1 + ping2.depth_m * w2,
            slant_range_m=ping1.slant_range_m * w1 + ping2.slant_range_m * w2,
            swath_width_m=ping1.swath_width_m * w1 + ping2.swath_width_m * w2,
            cable_out_m=ping1.cable_out_m * w1 + ping2.cable_out_m * w2,
            roll_deg=ping1.roll_deg * w1 + ping2.roll_deg * w2,
            pitch_deg=ping1.pitch_deg * w1 + ping2.pitch_deg * w2,
            heave_m=ping1.heave_m * w1 + ping2.heave_m * w2,
            hdop=ping1.hdop if ping1.hdop is not None else ping2.hdop,
            num_samples=ping1.num_samples,
            quality_flags=list(
                set(ping1.quality_flags) | set(ping2.quality_flags)
            ),
        )

    # ------------------------------------------------------------------
    # Detection processing
    # ------------------------------------------------------------------

    def process_detections(
        self,
        detections: List[Detection],
        image_info: SonarImageInfo,
        merge_overlapping: bool = True,
    ) -> List[GeolocatedDetection]:
        """
        Process all AI detections through the geotagging pipeline.

        For each detection:
            1. Determine centre ping from ``ping_range`` or bbox y-coords.
            2. Retrieve interpolated ping metadata.
            3. Transform pixel coordinates to geographic.
            4. Calculate dimensions in metres.
            5. Assign quality flags.
            6. Build a :class:`GeolocatedDetection`.

        Args:
            detections: Raw detections from the AI model.
            image_info: Sonar image dimensions/resolution.
            merge_overlapping: Whether to merge nearby same-class detections.

        Returns:
            List of geolocated detections.
        """
        results: List[GeolocatedDetection] = []
        for det in detections:
            try:
                geolocated = self._process_single_detection(det, image_info)
                results.append(geolocated)
            except Exception as e:
                self.logger.error(
                    "Error processing detection %s: %s", det.detection_id, e
                )

        if merge_overlapping and results:
            results = self._merge_overlapping_detections(results)

        self.geolocated_detections = results
        return results

    def _process_single_detection(
        self,
        detection: Detection,
        image_info: SonarImageInfo,
    ) -> GeolocatedDetection:
        """Process one detection into a geolocated result."""
        bbox = detection.bbox_pixels  # [x_min, y_min, x_max, y_max]

        # 1. Determine ping range
        if detection.ping_range is not None:
            start_ping, end_ping = detection.ping_range
        else:
            start_ping = int(bbox[1])
            end_ping = int(bbox[3])

        center_ping = (start_ping + end_ping) / 2.0

        # 2. Get centre ping metadata
        ping_meta = self.get_interpolated_ping(center_ping)

        # 3. Transform pixel centre to geographic
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        target_lon, target_lat = self.transformer.transform_pixel_to_geographic(
            center_x, center_y, ping_meta, image_info
        )

        # 4. Dimensions
        dimensions = self.transformer.calculate_detection_dimensions(
            bbox, ping_meta, image_info
        )

        # 5. Uncertainty
        pixel_offset = abs(center_x - image_info.samples_per_ping / 2.0)
        slant_dist = (
            pixel_offset / (image_info.samples_per_ping / 2.0)
        ) * image_info.range_m
        ground_range = self.transformer.calculate_ground_range(
            slant_dist, ping_meta.altitude_m
        )
        uncertainty = self.transformer.calculate_uncertainty(
            ping_meta, ground_range
        )

        # 6. Quality flags
        quality_flags = list(ping_meta.quality_flags)
        if ping_meta.cable_out_m > 0:
            quality_flags.append(QualityFlag.LAYBACK_APPLIED)
        if ping_meta.altitude_m > 0 and ping_meta.slant_range_m > ping_meta.altitude_m:
            quality_flags.append(QualityFlag.SLANT_RANGE_CORRECTED)
        # Deduplicate
        quality_flags = list(set(quality_flags))

        return GeolocatedDetection(
            detection_id=detection.detection_id,
            timestamp=ping_meta.timestamp,
            classification=detection.class_label,
            confidence_score=detection.confidence,
            bounding_box_pixels=list(bbox),
            geolocation=Geolocation(
                latitude=target_lat,
                longitude=target_lon,
                depth_meters=ping_meta.depth_m + ping_meta.altitude_m,
                uncertainty_meters=uncertainty,
            ),
            dimensions_meters=dimensions,
            ping_range=[int(start_ping), int(end_ping)],
            quality_flags=quality_flags,
        )

    # ------------------------------------------------------------------
    # Merge overlapping detections
    # ------------------------------------------------------------------

    def _merge_overlapping_detections(
        self, detections: List[GeolocatedDetection]
    ) -> List[GeolocatedDetection]:
        """
        Merge detections of the same class within *merge_distance_m*.

        Uses greedy spatial clustering:
            * Group by class label.
            * Within each group, merge pairs closer than threshold.
            * Merged position = confidence-weighted average.
            * Merged confidence = max of constituents.
        """
        if not detections:
            return []

        merged: List[GeolocatedDetection] = []
        by_class: Dict[str, List[GeolocatedDetection]] = {}
        for d in detections:
            by_class.setdefault(d.classification, []).append(d)

        for cls, items in by_class.items():
            used = set()
            for i, d1 in enumerate(items):
                if i in used:
                    continue
                cluster = [d1]
                used.add(i)

                for j, d2 in enumerate(items):
                    if j in used:
                        continue
                    dist = geographic_distance(
                        d1.geolocation.longitude,
                        d1.geolocation.latitude,
                        d2.geolocation.longitude,
                        d2.geolocation.latitude,
                    )
                    if dist <= self.merge_distance:
                        cluster.append(d2)
                        used.add(j)

                if len(cluster) == 1:
                    merged.append(cluster[0])
                else:
                    total_conf = sum(d.confidence_score for d in cluster)
                    avg_lat = (
                        sum(
                            d.geolocation.latitude * d.confidence_score
                            for d in cluster
                        )
                        / total_conf
                    )
                    avg_lon = (
                        sum(
                            d.geolocation.longitude * d.confidence_score
                            for d in cluster
                        )
                        / total_conf
                    )
                    max_conf = max(d.confidence_score for d in cluster)
                    max_len = max(d.dimensions_meters.length_m for d in cluster)
                    max_wid = max(d.dimensions_meters.width_m for d in cluster)
                    base = cluster[0]

                    merged.append(
                        GeolocatedDetection(
                            detection_id=f"{base.detection_id}_merged",
                            timestamp=base.timestamp,
                            classification=cls,
                            confidence_score=max_conf,
                            bounding_box_pixels=base.bounding_box_pixels,
                            geolocation=Geolocation(
                                latitude=avg_lat,
                                longitude=avg_lon,
                                depth_meters=sum(
                                    d.geolocation.depth_meters for d in cluster
                                )
                                / len(cluster),
                                uncertainty_meters=max(
                                    d.geolocation.uncertainty_meters
                                    for d in cluster
                                ),
                            ),
                            dimensions_meters=Dimensions(
                                length_m=max_len,
                                width_m=max_wid,
                                area_sq_meters=max_len * max_wid,
                            ),
                            ping_range=[
                                min(d.ping_range[0] for d in cluster),
                                max(d.ping_range[1] for d in cluster),
                            ],
                            quality_flags=base.quality_flags,
                        )
                    )

        return merged

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self, mission_metadata: MissionMetadata
    ) -> SurveyReport:
        """Generate complete survey report from processed detections."""
        by_class: Dict[str, int] = {}
        total_conf = 0.0
        for d in self.geolocated_detections:
            by_class[d.classification] = by_class.get(d.classification, 0) + 1
            total_conf += d.confidence_score

        n = len(self.geolocated_detections)
        summary = DetectionSummary(
            total_detections=n,
            detections_by_class=by_class,
            avg_confidence=round(total_conf / n, 3) if n else 0.0,
            total_pings_processed=len(self.pings),
        )

        return SurveyReport(
            mission_metadata=mission_metadata,
            detections=self.geolocated_detections,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # GeoJSON export
    # ------------------------------------------------------------------

    def get_geojson(self) -> Dict[str, Any]:
        """Export geolocated detections as a GeoJSON FeatureCollection."""
        features = []
        for d in self.geolocated_detections:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            d.geolocation.longitude,
                            d.geolocation.latitude,
                        ],
                    },
                    "properties": {
                        "detection_id": d.detection_id,
                        "class": d.classification,
                        "confidence": d.confidence_score,
                        "depth_m": d.geolocation.depth_meters,
                        "uncertainty_m": d.geolocation.uncertainty_meters,
                        "length_m": d.dimensions_meters.length_m,
                        "width_m": d.dimensions_meters.width_m,
                        "ping_start": d.ping_range[0],
                        "ping_end": d.ping_range[1],
                        "quality_flags": [
                            f.value for f in d.quality_flags
                        ],
                        "timestamp": d.timestamp.isoformat(),
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}
