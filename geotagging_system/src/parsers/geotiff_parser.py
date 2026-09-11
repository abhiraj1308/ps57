from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from .base_parser import BaseSonarParser
from src.models import PingMetadata, QualityFlag
from src.exceptions import MetadataParseError

try:
    import rasterio
except ImportError:
    rasterio = None

class GeoTIFFParser(BaseSonarParser):
    """Extracts georeferencing from GeoTIFF sonar images."""
    
    def parse(self, filepath: str) -> List[PingMetadata]:
        """Extract affine transform and CRS, generate ping metadata from image rows."""
        if rasterio is None:
            raise MetadataParseError("rasterio is not installed. Please install it to parse GeoTIFF files.")
            
        try:
            with rasterio.open(filepath) as src:
                self.transform = src.transform
                height = src.height
                width = src.width
        except Exception as e:
            raise MetadataParseError(f"Failed to read GeoTIFF {filepath}: {e}")
            
        pings = []
        center_col = width // 2
        
        for row in range(height):
            lon, lat = self.pixel_to_geographic(center_col, row)
            ping = PingMetadata(
                ping_number=row,
                timestamp=datetime.now(timezone.utc),  # Cannot reliably extract time from Tiff image
                latitude=lat,
                longitude=lon,
                quality_flags=[QualityFlag.GPS_VALID],
            )
            pings.append(ping)
            
        return self.validate_and_interpolate(pings)

    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        if rasterio is None:
            raise MetadataParseError("rasterio is not installed.")
        try:
            with rasterio.open(filepath) as src:
                return {
                    "width": src.width,
                    "height": src.height,
                    "crs": src.crs.to_string() if src.crs else "None"
                }
        except Exception as e:
            raise MetadataParseError(str(e))
            
    def pixel_to_geographic(self, col: int, row: int) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic using affine transform."""
        if not hasattr(self, 'transform'):
            raise MetadataParseError("Transform not loaded.")
        # Apply affine transform
        x, y = self.transform * (col, row)
        return x, y
