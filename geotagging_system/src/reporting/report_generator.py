"""
Multi-format report generator for survey results.

Produces JSON, CSV, GeoJSON, GeoPackage, KML, and basic PDF reports
from a :class:`SurveyReport`.
"""
import json
import csv
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from src.models import SurveyReport, GeolocatedDetection, MissionMetadata
from src.exceptions import ReportGenerationError
from src.reporting.kml_generator import KMLGenerator


class ReportGenerator:
    """
    Multi-format report generator for survey results.

    Supported formats: JSON, CSV, GeoJSON, GeoPackage, KML, PDF.

    Example::

        >>> gen = ReportGenerator(output_dir='./output')
        >>> gen.generate_json(report, 'survey_001.json')
        >>> paths = gen.generate_all(report, 'survey_001')
    """

    def __init__(self, output_dir: str = "./data/sample_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def generate_json(self, report: SurveyReport, filename: str) -> Path:
        """
        Generate JSON report matching the specified structure.

        Uses Pydantic ``model_dump(mode='json')`` for serialisation.
        """
        filepath = self.output_dir / filename
        try:
            data = report.model_dump(mode="json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info("JSON report written to %s", filepath)
            return filepath
        except Exception as e:
            raise ReportGenerationError(
                detail=f"Failed to generate JSON: {e}",
                format="json",
                filepath=str(filepath),
            )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def generate_csv(self, report: SurveyReport, filename: str) -> Path:
        """
        Generate CSV with columns: detection_id, timestamp, class,
        confidence, lat, lon, depth, length_m, width_m, ping_start,
        ping_end, quality_flag.
        """
        filepath = self.output_dir / filename
        try:
            df = self._detections_to_dataframe(report)
            df.to_csv(filepath, index=False)
            self.logger.info("CSV report written to %s", filepath)
            return filepath
        except Exception as e:
            raise ReportGenerationError(
                detail=f"Failed to generate CSV: {e}",
                format="csv",
                filepath=str(filepath),
            )

    # ------------------------------------------------------------------
    # GeoJSON
    # ------------------------------------------------------------------

    def generate_geojson(self, report: SurveyReport, filename: str) -> Path:
        """Generate GeoJSON FeatureCollection."""
        filepath = self.output_dir / filename
        try:
            features = []
            for d in report.detections:
                features.append({
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
                        "timestamp": d.timestamp.isoformat(),
                        "class": d.classification,
                        "confidence": d.confidence_score,
                        "depth_m": d.geolocation.depth_meters,
                        "uncertainty_m": d.geolocation.uncertainty_meters,
                        "length_m": d.dimensions_meters.length_m,
                        "width_m": d.dimensions_meters.width_m,
                        "area_sq_m": d.dimensions_meters.area_sq_meters,
                        "ping_start": d.ping_range[0],
                        "ping_end": d.ping_range[1],
                        "quality_flags": [f.value for f in d.quality_flags],
                    },
                })
            collection = {"type": "FeatureCollection", "features": features}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(collection, f, indent=2, default=str)
            self.logger.info("GeoJSON written to %s", filepath)
            return filepath
        except Exception as e:
            raise ReportGenerationError(
                detail=f"Failed to generate GeoJSON: {e}",
                format="geojson",
                filepath=str(filepath),
            )

    # ------------------------------------------------------------------
    # GeoPackage
    # ------------------------------------------------------------------

    def generate_geopackage(self, report: SurveyReport, filename: str) -> Path:
        """Generate GeoPackage (.gpkg) with a 'detections' layer."""
        filepath = self.output_dir / filename
        try:
            gdf = self._detections_to_geodataframe(report)
            if not gdf.empty:
                gdf.to_file(filepath, driver="GPKG", layer="detections")
            else:
                # Write empty GeoPackage
                empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
                empty.to_file(filepath, driver="GPKG", layer="detections")
            self.logger.info("GeoPackage written to %s", filepath)
            return filepath
        except Exception as e:
            raise ReportGenerationError(
                detail=f"Failed to generate GeoPackage: {e}",
                format="gpkg",
                filepath=str(filepath),
            )

    # ------------------------------------------------------------------
    # KML
    # ------------------------------------------------------------------

    def generate_kml(self, report: SurveyReport, filename: str) -> Path:
        """Generate KML using :class:`KMLGenerator`."""
        filepath = self.output_dir / filename
        try:
            generator = KMLGenerator()
            return generator.generate(report, str(filepath))
        except Exception as e:
            raise ReportGenerationError(
                detail=f"Failed to generate KML: {e}",
                format="kml",
                filepath=str(filepath),
            )

    # ------------------------------------------------------------------
    # PDF (optional)
    # ------------------------------------------------------------------

    def generate_pdf(self, report: SurveyReport, filename: str) -> Path:
        """
        Generate a basic PDF summary using ``reportlab``.

        Contains mission info, summary statistics, and a detection table.
        Gracefully skips if ``reportlab`` is not installed.
        """
        filepath = self.output_dir / filename
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(filepath), pagesize=letter)
            width, height = letter

            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, f"Survey Report: {report.mission_metadata.survey_id}")

            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Date: {report.mission_metadata.date}")
            c.drawString(50, height - 100, f"Vessel: {report.mission_metadata.vessel}")
            c.drawString(50, height - 120, f"Total Detections: {report.summary.total_detections}")
            c.drawString(50, height - 140, f"Avg Confidence: {report.summary.avg_confidence:.2f}")

            c.drawString(50, height - 170, "Detections:")
            y = height - 190
            for det in report.detections[:20]:
                line = (
                    f"  {det.detection_id} | {det.classification} | "
                    f"conf={det.confidence_score:.2f} | "
                    f"({det.geolocation.latitude:.5f}, {det.geolocation.longitude:.5f})"
                )
                c.drawString(60, y, line)
                y -= 18
                if y < 50:
                    c.showPage()
                    y = height - 50

            c.save()
            self.logger.info("PDF report written to %s", filepath)
            return filepath
        except ImportError:
            self.logger.warning("reportlab not installed — skipping PDF generation")
            return filepath
        except Exception as e:
            raise ReportGenerationError(
                detail=f"Failed to generate PDF: {e}",
                format="pdf",
                filepath=str(filepath),
            )

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    def generate_all(
        self,
        report: SurveyReport,
        base_name: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Generate all requested formats at once.

        Args:
            report: The survey report to export.
            base_name: Base filename (without extension).
            formats: List of format keys. Defaults to
                ``['json', 'csv', 'geojson', 'gpkg']``.

        Returns:
            Mapping of format → output path.
        """
        if formats is None:
            formats = ["json", "csv", "geojson", "gpkg"]

        dispatch = {
            "json": lambda: self.generate_json(report, f"{base_name}.json"),
            "csv": lambda: self.generate_csv(report, f"{base_name}.csv"),
            "geojson": lambda: self.generate_geojson(report, f"{base_name}.geojson"),
            "gpkg": lambda: self.generate_geopackage(report, f"{base_name}.gpkg"),
            "kml": lambda: self.generate_kml(report, f"{base_name}.kml"),
            "pdf": lambda: self.generate_pdf(report, f"{base_name}.pdf"),
        }

        results: Dict[str, Path] = {}
        for fmt in formats:
            if fmt in dispatch:
                results[fmt] = dispatch[fmt]()
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detections_to_dataframe(self, report: SurveyReport) -> pd.DataFrame:
        """Convert detections to a flat DataFrame."""
        rows = []
        for d in report.detections:
            rows.append(
                {
                    "detection_id": d.detection_id,
                    "timestamp": d.timestamp.isoformat(),
                    "class": d.classification,
                    "confidence": d.confidence_score,
                    "lat": d.geolocation.latitude,
                    "lon": d.geolocation.longitude,
                    "depth": d.geolocation.depth_meters,
                    "length_m": d.dimensions_meters.length_m,
                    "width_m": d.dimensions_meters.width_m,
                    "ping_start": d.ping_range[0],
                    "ping_end": d.ping_range[1],
                    "quality_flag": ",".join(f.value for f in d.quality_flags),
                }
            )
        return pd.DataFrame(rows)

    def _detections_to_geodataframe(
        self, report: SurveyReport
    ) -> gpd.GeoDataFrame:
        """Convert detections to GeoDataFrame with Point geometries."""
        df = self._detections_to_dataframe(report)
        if df.empty:
            return gpd.GeoDataFrame(columns=list(df.columns) + ["geometry"], crs="EPSG:4326")
        geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
