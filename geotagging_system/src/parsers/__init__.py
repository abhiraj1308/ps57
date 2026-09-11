import os
from .base_parser import BaseSonarParser
from .xtf_parser import XTFParser
from .csv_parser import CSVParser
from .json_parser import JSONParser
from .nmea_parser import NMEAParser
from .geotiff_parser import GeoTIFFParser
from src.exceptions import MetadataParseError

def get_parser(filepath: str) -> BaseSonarParser:
    """Auto-detect parser based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.xtf':
        return XTFParser()
    elif ext == '.csv':
        return CSVParser()
    elif ext == '.json':
        return JSONParser()
    elif ext in ['.txt', '.nmea', '.log']:
        return NMEAParser()
    elif ext in ['.tif', '.tiff']:
        return GeoTIFFParser()
    else:
        raise MetadataParseError(f"No parser found for file extension: {ext}")

__all__ = ['BaseSonarParser', 'XTFParser', 'CSVParser', 'JSONParser', 'NMEAParser', 'GeoTIFFParser', 'get_parser']
