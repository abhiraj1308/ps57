import math
from typing import Tuple, Optional
from pyproj import CRS, Transformer, Geod
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import logging

logger = logging.getLogger(__name__)

def auto_detect_utm_epsg(lon: float, lat: float) -> int:
    """
    Auto-detect UTM zone EPSG code for a given lon/lat.
    Uses pyproj.database.query_utm_crs_info with AreaOfInterest.
    Falls back to mathematical computation: zone = floor((lon+180)/6)+1
    EPSG = 32600+zone (N) or 32700+zone (S)
    Handles polar regions: UPS EPSG:32661 (N) or EPSG:32761 (S) when |lat|>84
    """
    if lat > 84.0:
        return 32661
    elif lat < -80.0:
        return 32761
        
    try:
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon,
                south_lat_degree=lat,
                east_lon_degree=lon,
                north_lat_degree=lat,
            ),
        )
        if utm_crs_list:
            return int(utm_crs_list[0].code)
    except Exception as e:
        logger.warning(f"Failed to query UTM CRS info: {e}. Falling back to math computation.")

    zone = math.floor((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def create_transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    """Create pyproj Transformer with always_xy=True."""
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def create_bidirectional_transformer(src_epsg: int, dst_epsg: int) -> Tuple[Transformer, Transformer]:
    """Create forward and inverse transformers."""
    forward = create_transformer(src_epsg, dst_epsg)
    inverse = create_transformer(dst_epsg, src_epsg)
    return forward, inverse


def geographic_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate geodesic distance in meters between two WGS84 points using pyproj.Geod."""
    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist


def bearing_between(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate forward azimuth (bearing) in degrees from point 1 to point 2."""
    geod = Geod(ellps="WGS84")
    fwd_az, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    return (fwd_az + 360.0) % 360.0


def project_point_along_bearing(lon: float, lat: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """Project a point along a bearing for a given distance. Returns (lon, lat). Uses Geod.fwd()."""
    geod = Geod(ellps="WGS84")
    lon2, lat2, _ = geod.fwd(lon, lat, bearing_deg, distance_m)
    return lon2, lat2


def handle_dateline_crossing(lon1: float, lon2: float) -> Tuple[float, float]:
    """Normalize longitudes to handle dateline crossing.
    If abs(lon2 - lon1) > 180, adjust lon2."""
    diff = lon2 - lon1
    if diff > 180:
        lon2 -= 360
    elif diff < -180:
        lon2 += 360
    return lon1, lon2


def is_valid_coordinate(lat: float, lon: float) -> bool:
    """Check if lat/lon are within valid ranges."""
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
