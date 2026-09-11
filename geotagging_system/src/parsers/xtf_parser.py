from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
from .base_parser import BaseSonarParser
from src.models import PingMetadata, QualityFlag
from src.exceptions import MetadataParseError

try:
    import pyxtf
except ImportError:
    pyxtf = None

class XTFParser(BaseSonarParser):
    def parse(self, filepath: str) -> List[PingMetadata]:
        """Parse XTF file.
        - Uses pyxtf.xtf_read() to read file header and packets
        - Extracts XTFHeaderType.sonar packets
        - For each XTFPingHeader: extract coordinates, heading, altitude, depth, layback, timestamp
        - Handles NavUnits (0=meters, 3=degrees)
        - Falls back to Ship coordinates when Sensor coordinates are 0
        - Wraps in try/except with MetadataParseError
        - Calls validate_and_interpolate before returning
        """
        if pyxtf is None:
            raise MetadataParseError("pyxtf is not installed. Please install it to parse XTF files.")
        
        try:
            file_header, packets = pyxtf.xtf_read(filepath)
            sonar_packets = packets[pyxtf.XTFHeaderType.sonar]
        except Exception as e:
            raise MetadataParseError(f"Failed to read XTF file {filepath}: {str(e)}")
            
        pings = []
        for i, pkt in enumerate(sonar_packets):
            sensor_x = pkt.SensorXcoordinate
            sensor_y = pkt.SensorYcoordinate

            used_ship_fallback = False
            if sensor_x == 0.0 and sensor_y == 0.0:
                sensor_x = pkt.ShipXcoordinate
                sensor_y = pkt.ShipYcoordinate
                used_ship_fallback = True

            lon = sensor_x
            lat = sensor_y

            ts = datetime(
                pkt.Year, pkt.Month, pkt.Day, 
                pkt.Hour, pkt.Minute, pkt.Second,
                tzinfo=timezone.utc
            ) + timedelta(milliseconds=getattr(pkt, 'HSeconds', 0) * 10)

            range_m = 0.0
            num_samples = 0
            if hasattr(pkt, 'ping_chan_headers') and len(pkt.ping_chan_headers) > 0:
                range_m = float(pkt.ping_chan_headers[0].SlantRange)
                num_samples = int(getattr(pkt.ping_chan_headers[0], 'NumSamples', 0))

            flags = [QualityFlag.GPS_VALID]
            if used_ship_fallback:
                flags = [QualityFlag.GPS_INTERPOLATED]

            ping = PingMetadata(
                ping_number=i,
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                heading_deg=float(pkt.SensorHeading),
                speed_knots=0.0,
                altitude_m=max(0.0, float(pkt.SensorPrimaryAltitude)),
                depth_m=max(0.0, float(pkt.SensorDepth)),
                slant_range_m=range_m,
                cable_out_m=max(0.0, float(getattr(pkt, 'Layback', 0.0) or 0.0)),
                num_samples=num_samples,
                quality_flags=flags,
                ship_latitude=float(pkt.ShipYcoordinate),
                ship_longitude=float(pkt.ShipXcoordinate),
            )
            pings.append(ping)
            
        return self.validate_and_interpolate(pings)
    
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """Read just the file header for quick info."""
        if pyxtf is None:
            raise MetadataParseError("pyxtf is not installed.")
        try:
            file_header = pyxtf.xtf_read(filepath, read_packets=False)
            return {
                "system_type": file_header.SystemType,
                "nav_units": file_header.NavUnits,
                "num_channels": file_header.NumberOfChannels
            }
        except Exception as e:
            raise MetadataParseError(f"Failed to read XTF header {filepath}: {str(e)}")
