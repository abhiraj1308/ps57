"""
Core coordinate transformation engine for sonar geotagging.

Handles slant-range to ground-range correction, layback correction for towed
systems, pixel-to-world coordinate mapping, and UTM/WGS84 conversions.
"""
import math
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

from src.models import (
    PingMetadata, Detection, GeolocatedDetection, Geolocation,
    Dimensions, QualityFlag, SonarSystemConfig, SonarImageInfo,
)
from src.exceptions import CoordinateTransformError
from src.transforms.projection_utils import (
    auto_detect_utm_epsg, create_bidirectional_transformer,
    geographic_distance, project_point_along_bearing, bearing_between,
    handle_dateline_crossing, is_valid_coordinate,
)
from pyproj import Transformer, Geod


class CoordinateTransformer:
    """
    Core coordinate transformation engine for sonar geotagging.

    Handles:
    - Slant range to ground range conversion
    - Layback correction for towed systems
    - Pixel-to-world coordinate transformation
    - UTM/WGS84 conversions
    - Detection footprint calculation

    Example::

        >>> from src.models import SonarSystemConfig
        >>> config = SonarSystemConfig(
        ...     name="Klein 3000", frequency_khz=445,
        ...     max_range_m=150, beam_width_deg=0.2
        ... )
        >>> transformer = CoordinateTransformer(config)
        >>> gr = transformer.calculate_ground_range(100.0, 30.0)
        >>> round(gr, 2)
        95.39
    """

    def __init__(
        self,
        sonar_config: SonarSystemConfig,
        utm_epsg: Optional[int] = None,
    ):
        """
        Initialize transformer.

        Args:
            sonar_config: Sonar system configuration.
            utm_epsg: Optional fixed UTM EPSG code.  If ``None``,
                      auto-detected from the first coordinate processed.
        """
        self.sonar_config = sonar_config
        self.utm_epsg = utm_epsg
        self._wgs_to_utm: Optional[Transformer] = None
        self._utm_to_wgs: Optional[Transformer] = None
        self._geod = Geod(ellps="WGS84")
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_transformers(self, lon: float, lat: float) -> None:
        """Lazy-init UTM transformers from the first coordinate seen."""
        if not is_valid_coordinate(lat, lon):
            raise CoordinateTransformError(
                source_crs="WGS84",
                target_crs="UTM",
                coordinates=[lon, lat],
            )
        if self._wgs_to_utm is None:
            if self.utm_epsg is None:
                self.utm_epsg = auto_detect_utm_epsg(lon, lat)
            self._wgs_to_utm, self._utm_to_wgs = (
                create_bidirectional_transformer(4326, self.utm_epsg)
            )
            self.logger.info("UTM zone auto-detected: EPSG:%s", self.utm_epsg)

    # ------------------------------------------------------------------
    # Slant range → ground range
    # ------------------------------------------------------------------

    def calculate_ground_range(
        self, slant_range: float, altitude: float
    ) -> float:
        """
        Convert slant range to ground range using the Pythagorean theorem.

        .. math::

            R_g = \\sqrt{R_s^2 - H^2} \\quad (R_s > H)

        Returns ``0.0`` when *slant_range* ≤ *altitude* (blind zone).

        Args:
            slant_range: Distance from sonar to target along acoustic
                path (metres).
            altitude: Height of sonar above seafloor (metres).

        Returns:
            Horizontal ground range in metres.

        Raises:
            CoordinateTransformError: If *altitude* is negative.

        Example::

            >>> t.calculate_ground_range(100.0, 30.0)
            95.3939...
        """
        if altitude < 0:
            raise CoordinateTransformError(
                source_crs="slant",
                target_crs="ground",
                coordinates=[slant_range, altitude],
            )
        if slant_range <= altitude:
            return 0.0
        return math.sqrt(slant_range ** 2 - altitude ** 2)

    # ------------------------------------------------------------------
    # Layback correction
    # ------------------------------------------------------------------

    def apply_layback_correction(
        self,
        ship_lon: float,
        ship_lat: float,
        heading_deg: float,
        cable_out_m: float,
        fish_depth_m: float,
        towpoint_height_m: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Calculate true towfish position from vessel GPS position.

        Uses the catenary model::

            layback = k × √(cable² − (depth + tow_height)²)

        and projects backwards along the heading vector.

        Args:
            ship_lon: Vessel GPS longitude (degrees).
            ship_lat: Vessel GPS latitude (degrees).
            heading_deg: Vessel heading (degrees, 0 = North, CW).
            cable_out_m: Cable payout length (metres).
            fish_depth_m: Towfish depth below surface (metres).
            towpoint_height_m: Towpoint height above waterline (metres).

        Returns:
            ``(fish_lon, fish_lat)`` in WGS-84 degrees.
        """
        if cable_out_m <= 0:
            return ship_lon, ship_lat

        vertical_dist = fish_depth_m + towpoint_height_m
        if cable_out_m < vertical_dist:
            self.logger.warning(
                "Cable out (%.1f m) < vertical depth (%.1f m); zeroing layback.",
                cable_out_m,
                vertical_dist,
            )
            return ship_lon, ship_lat

        k = self.sonar_config.catenary_factor
        layback = k * math.sqrt(cable_out_m ** 2 - vertical_dist ** 2)

        # Project backwards along heading
        reverse_heading = (heading_deg + 180.0) % 360.0
        return project_point_along_bearing(
            ship_lon, ship_lat, reverse_heading, layback
        )

    # ------------------------------------------------------------------
    # Pixel → geographic
    # ------------------------------------------------------------------

    def transform_pixel_to_geographic(
        self,
        x_pixel: float,
        y_pixel: float,
        ping_metadata: PingMetadata,
        image_info: SonarImageInfo,
    ) -> Tuple[float, float]:
        """
        Transform sonar image pixel coordinates to geographic coordinates.

        Args:
            x_pixel: Across-track pixel.  0 = port edge,
                ``samples_per_ping`` = starboard edge,
                ``samples_per_ping / 2`` = nadir.
            y_pixel: Along-track pixel (≈ ping index).
            ping_metadata: Navigation/attitude data for the ping.
            image_info: Sonar image dimensions and resolution.

        Returns:
            ``(longitude, latitude)`` in WGS-84 degrees.
        """
        self._ensure_transformers(
            ping_metadata.longitude, ping_metadata.latitude
        )

        # Layback correction (towed systems)
        fish_lon, fish_lat = self.apply_layback_correction(
            ping_metadata.longitude,
            ping_metadata.latitude,
            ping_metadata.heading_deg,
            ping_metadata.cable_out_m,
            ping_metadata.depth_m,
        )

        # Across-track offset from nadir
        center_x = image_info.samples_per_ping / 2.0
        pixel_offset = abs(x_pixel - center_x)

        # Convert pixel offset → slant distance
        slant_distance = (pixel_offset / center_x) * image_info.range_m

        # Slant → ground range
        ground_range = self.calculate_ground_range(
            slant_distance, ping_metadata.altitude_m
        )

        # Determine port/starboard and compute bearing
        is_starboard = x_pixel > center_x
        bearing_offset = 90.0 if is_starboard else -90.0
        target_heading = (ping_metadata.heading_deg + bearing_offset) % 360.0

        target_lon, target_lat = project_point_along_bearing(
            fish_lon, fish_lat, target_heading, ground_range
        )
        return target_lon, target_lat

    # ------------------------------------------------------------------
    # Full detection → GeoJSON Feature
    # ------------------------------------------------------------------

    def ping_to_world_coordinates(
        self,
        detection: Detection,
        ping_metadata: PingMetadata,
        image_info: SonarImageInfo,
    ) -> Dict[str, Any]:
        """
        Convert a detection bounding box to a GeoJSON Feature.

        Process:
            1. Compute bbox centre in pixel space.
            2. Transform centre to geographic coordinates.
            3. Calculate detection dimensions in metres.
            4. Estimate positional uncertainty.
            5. Build and return a GeoJSON ``Feature`` dict.

        Returns:
            GeoJSON Feature dictionary.
        """
        bbox = detection.bbox_pixels
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0

        target_lon, target_lat = self.transform_pixel_to_geographic(
            center_x, center_y, ping_metadata, image_info
        )

        # Ground range for uncertainty estimation
        pixel_offset = abs(center_x - (image_info.samples_per_ping / 2.0))
        slant_distance = (
            pixel_offset / (image_info.samples_per_ping / 2.0)
        ) * image_info.range_m
        ground_range = self.calculate_ground_range(
            slant_distance, ping_metadata.altitude_m
        )

        dimensions = self.calculate_detection_dimensions(
            bbox, ping_metadata, image_info
        )
        uncertainty = self.calculate_uncertainty(ping_metadata, ground_range)

        feature: Dict[str, Any] = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [target_lon, target_lat],
            },
            "properties": {
                "detection_id": detection.detection_id,
                "class_label": detection.class_label,
                "confidence": detection.confidence,
                "dimensions": {
                    "length_m": dimensions.length_m,
                    "width_m": dimensions.width_m,
                    "area_sq_meters": dimensions.area_sq_meters,
                },
                "uncertainty_m": uncertainty,
                "depth_m": ping_metadata.depth_m,
                "timestamp": ping_metadata.timestamp.isoformat(),
            },
        }
        return feature

    # ------------------------------------------------------------------
    # Detection dimensions
    # ------------------------------------------------------------------

    def calculate_detection_dimensions(
        self,
        bbox_pixels: List[float],
        ping_metadata: PingMetadata,
        image_info: SonarImageInfo,
    ) -> Dimensions:
        """
        Calculate real-world dimensions of a detection.

        * **Width** (across-track): difference in ground range between
          the left and right bbox edges.
        * **Length** (along-track): number of ping lines × along-track
          spacing (estimated from vessel speed and ping interval).

        Args:
            bbox_pixels: ``[x_min, y_min, x_max, y_max]`` in pixels.
            ping_metadata: Navigation data for the detection's centre ping.
            image_info: Sonar image dimensions/resolution.

        Returns:
            :class:`Dimensions` with ``length_m``, ``width_m``, and
            ``area_sq_meters``.
        """
        # ---- Across-track width ----
        center_x = image_info.samples_per_ping / 2.0

        slant1 = (
            abs(bbox_pixels[0] - center_x) / center_x
        ) * image_info.range_m
        slant2 = (
            abs(bbox_pixels[2] - center_x) / center_x
        ) * image_info.range_m

        gr1 = self.calculate_ground_range(slant1, ping_metadata.altitude_m)
        gr2 = self.calculate_ground_range(slant2, ping_metadata.altitude_m)
        width_m = abs(gr2 - gr1)

        # ---- Along-track length ----
        length_pixels = bbox_pixels[3] - bbox_pixels[1]
        # Estimate ping spacing from speed and number of pings
        speed_ms = ping_metadata.speed_knots * 0.514444  # knots → m/s
        # Default ping rate ≈ 10 Hz; override with image_info if available
        if image_info.pixel_size_along_m is not None and image_info.pixel_size_along_m > 0:
            ping_spacing = image_info.pixel_size_along_m
        elif speed_ms > 0:
            # Assume ~10 pings/s as default ping rate
            ping_spacing = speed_ms / 10.0
        else:
            # Last resort: estimate from range / height_pixels
            ping_spacing = image_info.range_m / image_info.height_pixels

        length_m = abs(length_pixels) * ping_spacing

        area = length_m * width_m
        return Dimensions(
            length_m=round(length_m, 3),
            width_m=round(width_m, 3),
            area_sq_meters=round(area, 3),
        )

    # ------------------------------------------------------------------
    # Positional uncertainty
    # ------------------------------------------------------------------

    def calculate_uncertainty(
        self,
        ping_metadata: PingMetadata,
        ground_range: float,
    ) -> float:
        """
        Estimate positional uncertainty in metres.

        Factors:
        - Base GPS uncertainty scaled by HDOP (default 1.5 if missing).
        - Angular uncertainty proportional to ground range.
        - Layback uncertainty proportional to cable payout.
        - Extra penalty when GPS was interpolated.

        Args:
            ping_metadata: Navigation/quality data.
            ground_range: Horizontal ground range to target (metres).

        Returns:
            Estimated uncertainty radius in metres.
        """
        hdop = ping_metadata.hdop if ping_metadata.hdop is not None else 1.5
        uncertainty = 1.0 * hdop
        uncertainty += 0.02 * ground_range
        uncertainty += 0.1 * ping_metadata.cable_out_m

        if QualityFlag.GPS_INTERPOLATED in ping_metadata.quality_flags:
            uncertainty += 5.0

        return round(uncertainty, 2)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def transform_relative_to_absolute(
        self,
        x_pixel: float,
        y_pixel: float,
        sonar_params: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        High-level convenience: pixel → ``(lon, lat)`` using a dict.

        ``sonar_params`` must contain ``"ping_metadata"`` and
        ``"image_info"`` sub-dicts that can be unpacked into
        :class:`PingMetadata` and :class:`SonarImageInfo`.

        Returns:
            ``(longitude, latitude)`` in WGS-84 degrees.
        """
        ping_metadata = PingMetadata(**sonar_params["ping_metadata"])
        image_info = SonarImageInfo(**sonar_params["image_info"])
        return self.transform_pixel_to_geographic(
            x_pixel, y_pixel, ping_metadata, image_info
        )
