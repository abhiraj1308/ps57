"""
KML/KMZ export for Google Earth visualization.

Uses the ``simplekml`` library to create styled placemarks for each
detection with colour-coded icons and HTML description popups.
"""
import logging
from pathlib import Path
from typing import Dict

try:
    import simplekml
    HAS_SIMPLEKML = True
except ImportError:
    HAS_SIMPLEKML = False

from src.models import SurveyReport, GeolocatedDetection
from src.exceptions import ReportGenerationError


class KMLGenerator:
    """KML export for Google Earth visualization."""

    # Colour mapping for detection classes (KML AABBGGRR format)
    _FALLBACK_COLORS: Dict[str, str] = {
        "ghost_net": "ff0000ff",
        "pipe": "ffff0000",
        "wreck": "ff00a5ff",
        "tire": "ff00ff00",
        "debris": "ff00ffff",
        "default": "ffffffff",
    }

    def generate(self, report: SurveyReport, output_path: str) -> Path:
        """Generate a KML file from a survey report."""
        if not HAS_SIMPLEKML:
            raise ReportGenerationError(
                detail="simplekml library is required for KML generation",
                format="kml",
            )

        kml = simplekml.Kml()
        kml.document.name = (
            f"Survey Detections - {report.mission_metadata.survey_id}"
        )

        folder = kml.newfolder(name="Detections")
        for det in report.detections:
            self._create_placemark(folder, det)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kml.save(str(path))
        return path

    def _create_placemark(
        self, kml_doc: "simplekml.Folder", detection: GeolocatedDetection
    ) -> None:
        """Create a styled placemark for a single detection."""
        pnt = kml_doc.newpoint(
            name=f"{detection.classification} ({detection.confidence_score:.2f})"
        )
        pnt.coords = [
            (detection.geolocation.longitude, detection.geolocation.latitude)
        ]
        pnt.description = self._create_description(detection)

        color = self._get_color(detection.classification)
        pnt.style.iconstyle.icon.href = (
            "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        )
        pnt.style.iconstyle.color = color

    def _create_description(self, detection: GeolocatedDetection) -> str:
        """Create an HTML description popup."""
        flags = ", ".join(f.value for f in detection.quality_flags)
        return (
            f"<b>ID:</b> {detection.detection_id}<br>"
            f"<b>Class:</b> {detection.classification}<br>"
            f"<b>Confidence:</b> {detection.confidence_score:.2f}<br>"
            f"<b>Depth (m):</b> {detection.geolocation.depth_meters:.2f}<br>"
            f"<b>Dimensions (L×W m):</b> "
            f"{detection.dimensions_meters.length_m:.2f} × "
            f"{detection.dimensions_meters.width_m:.2f}<br>"
            f"<b>Timestamp:</b> {detection.timestamp.isoformat()}<br>"
            f"<b>Quality:</b> {flags}<br>"
        )

    def _get_color(self, class_label: str) -> str:
        """Get KML colour for a detection class."""
        if HAS_SIMPLEKML:
            kml_colors = {
                "ghost_net": simplekml.Color.red,
                "pipe": simplekml.Color.blue,
                "wreck": simplekml.Color.orange,
                "tire": simplekml.Color.green,
                "debris": simplekml.Color.yellow,
            }
            return kml_colors.get(
                class_label.lower(), simplekml.Color.white
            )
        return self._FALLBACK_COLORS.get(
            class_label.lower(), self._FALLBACK_COLORS["default"]
        )
