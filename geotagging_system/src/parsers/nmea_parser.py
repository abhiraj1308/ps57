from typing import List, Dict, Any
from datetime import datetime, timezone
import re
from .base_parser import BaseSonarParser
from src.models import PingMetadata, QualityFlag
from src.exceptions import MetadataParseError

class NMEAParser(BaseSonarParser):
    """
    Parses NMEA 0183 sentences for navigation data.
    Supported sentences:
    - $GPGGA: position, fix quality, HDOP, altitude
    - $GPRMC: position, speed, course, date/time
    - $GPVTG: track and speed over ground
    - $SDDBT/$SDDBS: depth below transducer/surface
    """
    def parse(self, filepath: str) -> List[PingMetadata]:
        """Read NMEA log file line by line, parse sentences, merge by timestamp."""
        pings_data = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            raise MetadataParseError(f"Failed to read NMEA file: {e}")
            
        for line in lines:
            line = line.strip()
            if not line or not self._validate_checksum(line):
                continue
                
            parts = line.split('*')[0].split(',')
            sentence_type = parts[0]
            
            try:
                data = {}
                if sentence_type.endswith('GGA'):
                    data = self._parse_gga(parts)
                elif sentence_type.endswith('RMC'):
                    data = self._parse_rmc(parts)
                elif sentence_type.endswith('VTG'):
                    data = self._parse_vtg(parts)
                elif sentence_type.endswith('DBT') or sentence_type.endswith('DBS'):
                    data = self._parse_depth(parts)
                
                if 'timestamp' in data:
                    ts = data['timestamp']
                    if ts not in pings_data:
                        pings_data[ts] = {}
                    pings_data[ts].update(data)
            except Exception as e:
                self.logger.warning(f"Error parsing NMEA sentence {line}: {e}")
                
        pings = []
        for i, (ts, data) in enumerate(sorted(pings_data.items())):
            ping = PingMetadata(
                ping_number=i,
                timestamp=ts,
                latitude=data.get('latitude', 0.0),
                longitude=data.get('longitude', 0.0),
                heading_deg=data.get('heading') or 0.0,
                speed_knots=data.get('speed') or 0.0,
                altitude_m=max(0.0, data.get('altitude') or 0.0),
                depth_m=max(0.0, data.get('depth') or 0.0),
                quality_flags=[QualityFlag.GPS_VALID],
            )
            pings.append(ping)
            
        return self.validate_and_interpolate(pings)

    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        return {"type": "NMEA 0183"}

    def _parse_gga(self, parts: List[str]) -> dict:
        """Parse $GPGGA sentence."""
        if len(parts) < 10 or not parts[2]: return {}
        return {
            'latitude': self._nmea_lat_to_decimal(parts[2], parts[3]),
            'longitude': self._nmea_lon_to_decimal(parts[4], parts[5]),
            'altitude': float(parts[9]) if parts[9] else None
        }

    def _parse_rmc(self, parts: List[str]) -> dict:
        """Parse $GPRMC sentence."""
        if len(parts) < 10 or parts[2] != 'A': return {}
        
        time_part = parts[1]
        date_part = parts[9]
        if len(time_part) >= 6 and len(date_part) == 6:
            ts_str = f"20{date_part[4:6]}-{date_part[2:4]}-{date_part[0:2]}T{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}Z"
            ts = self._normalize_timestamp(ts_str)
        else:
            ts = datetime.now(timezone.utc)
            
        return {
            'timestamp': ts,
            'latitude': self._nmea_lat_to_decimal(parts[3], parts[4]),
            'longitude': self._nmea_lon_to_decimal(parts[5], parts[6]),
            'speed': float(parts[7]) if parts[7] else None,
            'heading': float(parts[8]) if parts[8] else None
        }

    def _parse_vtg(self, parts: List[str]) -> dict:
        """Parse $GPVTG sentence."""
        if len(parts) < 8: return {}
        return {
            'heading': float(parts[1]) if parts[1] else None,
            'speed': float(parts[7]) if parts[7] else None
        }

    def _parse_depth(self, parts: List[str]) -> dict:
        """Parse $SDDBT or $SDDBS sentence."""
        if len(parts) < 4: return {}
        return {'depth': float(parts[3]) if parts[3] else None}

    def _validate_checksum(self, sentence: str) -> bool:
        """Validate NMEA XOR checksum."""
        if '*' not in sentence or not sentence.startswith('$'):
            return False
        content, checksum = sentence[1:].split('*')
        calc_checksum = 0
        for char in content:
            calc_checksum ^= ord(char)
        return f"{calc_checksum:02X}" == checksum.upper()

    def _nmea_lat_to_decimal(self, value: str, direction: str) -> float:
        """Convert NMEA lat (DDMM.MMMM,N/S) to decimal degrees."""
        if not value: return 0.0
        try:
            deg = int(value[:2])
            min = float(value[2:])
            dec = deg + min / 60.0
            return -dec if direction == 'S' else dec
        except Exception:
            return 0.0

    def _nmea_lon_to_decimal(self, value: str, direction: str) -> float:
        """Convert NMEA lon (DDDMM.MMMM,E/W) to decimal degrees."""
        if not value: return 0.0
        try:
            deg = int(value[:3])
            min = float(value[3:])
            dec = deg + min / 60.0
            return -dec if direction == 'W' else dec
        except Exception:
            return 0.0
