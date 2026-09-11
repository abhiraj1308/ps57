import json
from typing import List, Dict, Any
from datetime import datetime, timezone
from .base_parser import BaseSonarParser
from src.models import PingMetadata
from src.exceptions import MetadataParseError

class JSONParser(BaseSonarParser):
    def parse(self, filepath: str) -> List[PingMetadata]:
        """Parse JSON metadata file.
        Supports formats:
        1. Array of ping objects: [{"ping_number": 0, "latitude": ..., ...}, ...]
        2. Nested: {"pings": [...], "metadata": {...}}
        3. AUV format: {"navigation": [{"time": ..., "lat": ..., ...}], "sonar": {...}}
        Auto-detects format and normalizes to PingMetadata list.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise MetadataParseError(f"Failed to read JSON {filepath}: {str(e)}")
            
        ping_dicts = self._flatten_nested(data)
        
        pings = []
        for i, pd_ in enumerate(ping_dicts):
            ts_str = pd_.get('timestamp') or pd_.get('time')
            ts = self._normalize_timestamp(str(ts_str)) if ts_str else datetime.now(timezone.utc)
            
            ping = PingMetadata(
                ping_number=pd_.get('ping_number', pd_.get('ping', i)),
                timestamp=ts,
                latitude=float(pd_.get('latitude', pd_.get('lat', 0.0))),
                longitude=float(pd_.get('longitude', pd_.get('lon', pd_.get('lng', 0.0)))),
                heading_deg=float(pd_.get('heading', pd_.get('heading_deg', 0.0)) or 0.0),
                speed_knots=float(pd_.get('speed', pd_.get('speed_knots', 0.0)) or 0.0),
                altitude_m=float(pd_.get('altitude', pd_.get('altitude_m', pd_.get('alt', 0.0))) or 0.0),
                depth_m=float(pd_.get('depth', pd_.get('depth_m', 0.0)) or 0.0),
                slant_range_m=float(pd_.get('slant_range', pd_.get('slant_range_m', pd_.get('range', 0.0))) or 0.0),
                cable_out_m=float(pd_.get('cable_out', pd_.get('cable_out_m', pd_.get('layback', 0.0))) or 0.0),
            )
            pings.append(ping)
            
        return self.validate_and_interpolate(pings)
        
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"type": type(data).__name__, "length": len(data) if isinstance(data, list) else len(data.keys())}
        except Exception as e:
            raise MetadataParseError(str(e))
            
    def _flatten_nested(self, data: dict) -> List[dict]:
        """Find the ping array in nested JSON structures."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if 'pings' in data and isinstance(data['pings'], list):
                return data['pings']
            if 'navigation' in data and isinstance(data['navigation'], list):
                return data['navigation']
        return [data]
