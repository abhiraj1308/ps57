"""Custom exception hierarchy for the geotagging system."""


class GeotaggingError(Exception):
    """Base exception for all geotagging errors."""
    pass


class MetadataParseError(GeotaggingError):
    """Raised by parsers on corrupt/missing data."""
    def __init__(self, detail: str, filepath: str = "", line_number: int = 0):
        self.filepath = filepath
        self.line_number = line_number
        self.detail = detail
        loc = f" at {filepath}:{line_number}" if filepath else ""
        super().__init__(f"MetadataParseError{loc} - {detail}")


class CoordinateTransformError(GeotaggingError):
    """Raised on invalid transforms."""
    def __init__(self, source_crs: str = "", target_crs: str = "", coordinates: list = None):
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.coordinates = coordinates or []
        super().__init__(f"CoordinateTransformError: {source_crs} -> {target_crs} failed for {self.coordinates}")


class ReportGenerationError(GeotaggingError):
    """Raised on output failures."""
    def __init__(self, detail: str, format: str = "", filepath: str = ""):
        self.format = format
        self.filepath = filepath
        self.detail = detail
        super().__init__(f"ReportGenerationError: {detail}")


class ValidationError(GeotaggingError):
    """Raised by QA validator."""
    def __init__(self, rule_name: str, detection_id: str = "", detail: str = ""):
        self.rule_name = rule_name
        self.detection_id = detection_id
        self.detail = detail
        super().__init__(f"ValidationError ({rule_name}) for detection {detection_id}: {detail}")


class PipelineError(GeotaggingError):
    """Raised by batch processor."""
    def __init__(self, detail: str, stage: str = ""):
        self.stage = stage
        self.detail = detail
        super().__init__(f"PipelineError at stage '{stage}': {detail}")
