import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timezone
import re
from .base_parser import BaseSonarParser
from src.models import PingMetadata
from src.exceptions import MetadataParseError

class CSVParser(BaseSonarParser):
    # Common column name mappings
    COLUMN_MAPPINGS = {
        'ping_number': ['ping_number', 'ping_num', 'ping', 'ping_id', 'ping_no'],
        'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'utc_time'],
        'latitude': ['latitude', 'lat', 'y', 'sensor_y', 'fish_lat'],
        'longitude': ['longitude', 'lon', 'lng', 'x', 'sensor_x', 'fish_lon'],
        'heading': ['heading', 'heading_deg', 'hdg', 'course', 'cog', 'sensor_heading'],
        'speed': ['speed', 'speed_knots', 'sog', 'velocity'],
        'altitude': ['altitude', 'alt', 'altitude_m', 'sensor_altitude', 'height_above_bottom'],
        'depth': ['depth', 'depth_m', 'sensor_depth', 'fish_depth'],
        'slant_range': ['slant_range', 'range', 'range_m', 'max_range'],
        'cable_out': ['cable_out', 'cable_out_m', 'layback', 'cable_payout'],
    }
    
    def parse(self, filepath: str) -> List[PingMetadata]:
        """Parse CSV with auto-detected column mapping.
        Uses pandas for vectorized loading.
        Handles: missing columns (use defaults), extra columns (ignore),
        multiple timestamp formats, DMS coordinates.
        """
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise MetadataParseError(f"Failed to read CSV {filepath}: {str(e)}")

        if df.empty:
            raise MetadataParseError(f"CSV file contains no data rows: {filepath}")

        col_map = self._auto_map_columns(list(df.columns))
        
        pings = []
        for i, row in df.iterrows():
            ping_no = row.get(col_map.get('ping_number', ''), i)
            
            ts_val = row.get(col_map.get('timestamp', ''), None)
            if ts_val and pd.notnull(ts_val):
                ts = self._normalize_timestamp(str(ts_val))
            else:
                ts = datetime.now(timezone.utc)
                
            lat_val = row.get(col_map.get('latitude', ''), 0.0)
            lon_val = row.get(col_map.get('longitude', ''), 0.0)
            
            if isinstance(lat_val, str) and '°' in lat_val:
                lat = self._parse_dms_coordinate(lat_val)
            else:
                lat = float(lat_val) if pd.notnull(lat_val) else 0.0
                
            if isinstance(lon_val, str) and '°' in lon_val:
                lon = self._parse_dms_coordinate(lon_val)
            else:
                lon = float(lon_val) if pd.notnull(lon_val) else 0.0
                
            def get_val(key):
                if key not in col_map:
                    return None
                v = row.get(col_map[key], None)
                return float(v) if v is not None and pd.notnull(v) else None
                
            ping = PingMetadata(
                ping_number=int(ping_no) if pd.notnull(ping_no) else i,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                heading_deg=get_val('heading') or 0.0,
                speed_knots=get_val('speed') or 0.0,
                altitude_m=get_val('altitude') or 0.0,
                depth_m=get_val('depth') or 0.0,
                slant_range_m=get_val('slant_range') or 0.0,
                cable_out_m=get_val('cable_out') or 0.0,
            )
            pings.append(ping)

        if not pings:
            raise MetadataParseError(f"No valid pings parsed from CSV: {filepath}")

        return self.validate_and_interpolate(pings)
    
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(filepath, nrows=5)
            return {"columns": list(df.columns), "num_rows": len(df)}
        except Exception as e:
            raise MetadataParseError(f"Failed to get info for {filepath}: {str(e)}")
            
    def _auto_map_columns(self, df_columns: List[str]) -> Dict[str, str]:
        """Map CSV columns to standard field names using COLUMN_MAPPINGS."""
        mapping = {}
        df_cols_lower = {c.lower(): c for c in df_columns}
        for std_col, alternatives in self.COLUMN_MAPPINGS.items():
            for alt in alternatives:
                if alt in df_cols_lower:
                    mapping[std_col] = df_cols_lower[alt]
                    break
        return mapping
    
    def _parse_dms_coordinate(self, value: str) -> float:
        """Convert DMS string like '27°50'44.16"N' to decimal degrees."""
        match = re.search(r"(\d+)°(\d+)'?([\d.]+)\"?([NSEW])", value.replace(' ', ''))
        if match:
            deg, min, sec, dir = match.groups()
            dec = float(deg) + float(min)/60 + float(sec)/3600
            if dir in ['S', 'W']:
                dec = -dec
            return dec
        return 0.0
