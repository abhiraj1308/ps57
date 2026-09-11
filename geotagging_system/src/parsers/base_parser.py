from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import numpy as np
from scipy import interpolate
import logging
from src.models import PingMetadata, QualityFlag
from src.exceptions import MetadataParseError

class BaseSonarParser(ABC):
    """Abstract base class for sonar metadata parsers."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse(self, filepath: str) -> List[PingMetadata]:
        """Parse sonar file and return list of ping metadata."""
        pass
    
    @abstractmethod
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """Get file format info without full parsing."""
        pass
    
    def validate_and_interpolate(self, pings: List[PingMetadata]) -> List[PingMetadata]:
        """
        Validate navigation data and interpolate gaps.
        - Detects zero/null coordinates and interpolates from neighbors
        - Flags interpolated positions with GPS_INTERPOLATED quality flag
        Uses numpy vectorized operations for speed.
        """
        if not pings:
            return []

        lats = np.array([p.latitude for p in pings])
        lons = np.array([p.longitude for p in pings])
        headings = np.array([p.heading_deg for p in pings])
        altitudes = np.array([p.altitude_m for p in pings])
        depths = np.array([p.depth_m for p in pings])

        # Detect gaps (lat==0 AND lon==0 treated as dropouts)
        dropouts = (lats == 0.0) & (lons == 0.0)
        valid = ~dropouts

        if not np.any(valid):
            raise MetadataParseError("No valid positions found in the dataset.")

        # If no dropouts, return as-is
        if not np.any(dropouts):
            return pings

        indices = np.arange(len(pings))
        valid_indices = indices[valid]

        # Interpolate coordinates
        interp_lat = interpolate.interp1d(valid_indices, lats[valid], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_lon = interpolate.interp1d(valid_indices, lons[valid], kind='linear', bounds_error=False, fill_value="extrapolate")

        new_lats = interp_lat(indices)
        new_lons = interp_lon(indices)

        # Build corrected ping list
        result = []
        for i, p in enumerate(pings):
            if dropouts[i]:
                # Create a new PingMetadata with interpolated coords and flag
                flags = list(p.quality_flags)
                if QualityFlag.GPS_INTERPOLATED not in flags:
                    flags.append(QualityFlag.GPS_INTERPOLATED)
                new_p = p.model_copy(update={
                    'latitude': float(new_lats[i]),
                    'longitude': float(new_lons[i]),
                    'quality_flags': flags,
                })
                result.append(new_p)
            else:
                result.append(p)

        return result

    def _detect_dropouts(self, values: np.ndarray) -> np.ndarray:
        """Returns boolean mask of dropout positions (zeros or NaNs)."""
        return (values == 0.0) | np.isnan(values)
    
    def _normalize_timestamp(self, ts_str: str, formats: List[str] = None) -> datetime:
        """Try multiple timestamp formats, return UTC datetime."""
        # Try ISO format first (handles +00:00, Z suffix, etc.)
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass

        if formats is None:
            formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        raise MetadataParseError(f"Could not parse timestamp: {ts_str}")
    
    def _validate_coordinate(self, lat: float, lon: float) -> bool:
        """Basic coordinate range validation."""
        if np.isnan(lat) or np.isnan(lon): return False
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
