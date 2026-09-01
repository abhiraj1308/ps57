from detection_intelligence.schemas import (
    VisionDetection,
    SonarDetection,
)

from detection_intelligence.fusion import fuse_detections


def main():
    print("=" * 60)
    print("PS57 DETECTION INTELLIGENCE TEST")
    print("=" * 60)

    vision = [
        VisionDetection(
            class_name="debris",
            confidence=0.94,
            latitude=19.0760,
            longitude=72.8777,
            width=12.5,
            height=4.2,
        )
    ]

    sonar = [
        SonarDetection(
            range_index=50,
            intensity=0.82,
            span_start=48,
            span_end=52,
        ),
        SonarDetection(
            range_index=140,
            intensity=0.61,
            span_start=138,
            span_end=142,
        ),
    ]

    results = fuse_detections(
        vision_detections=vision,
        sonar_detections=sonar,
    )

    print("\nFUSED RESULTS")
    print("-" * 60)

    for result in results:
        print(f"Source       : {result.source}")
        print(f"Class        : {result.class_name}")
        print(f"Confidence   : {result.confidence}")
        print(f"Priority     : {result.priority}")
        print(f"Latitude     : {result.latitude}")
        print(f"Longitude    : {result.longitude}")
        print(f"Sonar Range  : {result.sonar_range_index}")
        print(f"Sonar Signal : {result.sonar_intensity}")
        print(f"Fusion       : {result.fusion_status}")
        print("-" * 60)


if __name__ == "__main__":
    main()