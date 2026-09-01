import math

from .coordinates import GeoPoint, LocalPoint


EARTH_RADIUS_METRES = 6_378_137.0


def local_to_geo(
    reference: GeoPoint,
    local_position: LocalPoint,
) -> GeoPoint:
    """
    Converts a local X/Y position in metres into
    latitude/longitude relative to a reference point.

    Coordinate convention:

        +X = East
        -X = West
        +Y = North
        -Y = South

    This approximation is suitable for the relatively
    small operating area of a prototype.

    Parameters
    ----------
    reference:
        GPS position representing the origin.

    local_position:
        Local displacement from that origin in metres.

    Returns
    -------
    GeoPoint:
        Calculated latitude and longitude.
    """

    latitude_radians = math.radians(reference.latitude)

    delta_latitude = (
        local_position.y / EARTH_RADIUS_METRES
    )

    delta_longitude = (
        local_position.x
        / (
            EARTH_RADIUS_METRES
            * math.cos(latitude_radians)
        )
    )

    latitude = (
        reference.latitude
        + math.degrees(delta_latitude)
    )

    longitude = (
        reference.longitude
        + math.degrees(delta_longitude)
    )

    return GeoPoint(
        latitude=latitude,
        longitude=longitude,
    )


def distance_metres(
    point_a: GeoPoint,
    point_b: GeoPoint,
) -> float:
    """
    Calculates the approximate distance between
    two geographic points using the Haversine formula.
    """

    lat1 = math.radians(point_a.latitude)
    lat2 = math.radians(point_b.latitude)

    delta_lat = math.radians(
        point_b.latitude - point_a.latitude
    )

    delta_lon = math.radians(
        point_b.longitude - point_a.longitude
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return EARTH_RADIUS_METRES * c