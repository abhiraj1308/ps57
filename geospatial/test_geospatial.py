from geospatial.coordinates import (
    GeoPoint,
    LocalPoint,
)

from geospatial.transforms import (
    distance_metres,
    local_to_geo,
)


def main():
    print("=" * 60)
    print("PS57 GEOSPATIAL TEST")
    print("=" * 60)

    reference = GeoPoint(
        latitude=19.0760,
        longitude=72.8777,
    )

    print("\nREFERENCE POSITION")
    print("-" * 60)
    print(f"Latitude  : {reference.latitude}")
    print(f"Longitude : {reference.longitude}")

    test_positions = [
        LocalPoint(x=0, y=0),
        LocalPoint(x=10, y=0),
        LocalPoint(x=0, y=10),
        LocalPoint(x=10, y=10),
        LocalPoint(x=-10, y=-10),
    ]

    print("\nTRANSFORMED POSITIONS")
    print("-" * 60)

    for position in test_positions:
        geographic = local_to_geo(
            reference=reference,
            local_position=position,
        )

        distance = distance_metres(
            reference,
            geographic,
        )

        print(
            f"Local X/Y : "
            f"({position.x:6.1f} m, {position.y:6.1f} m)"
        )

        print(
            f"GPS       : "
            f"{geographic.latitude:.7f}, "
            f"{geographic.longitude:.7f}"
        )

        print(
            f"Distance  : "
            f"{distance:.2f} m"
        )

        print("-" * 60)

    print("\nGEOSPATIAL TEST COMPLETE")


if __name__ == "__main__":
    main()