from dataclasses import dataclass


@dataclass
class GeoPoint:
    """
    Geographic position expressed as latitude and longitude.
    """

    latitude: float
    longitude: float


@dataclass
class LocalPoint:
    """
    Local position relative to a reference point.

    x = east/west displacement in metres
    y = north/south displacement in metres
    """

    x: float
    y: float