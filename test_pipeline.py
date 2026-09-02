from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ps57_pipeline import (
    PipelineDependencies,
    analyze_sss,
    persist,
)


# ============================================================
# MOCK DETECTION OBJECT
# ============================================================

@dataclass
class MockDetection:
    class_name: str = "man_made_anomaly"
    confidence: float = 0.91

    latitude: float | None = 19.076243
    longitude: float | None = 72.877931

    severity: str = "high"
    decision: str = "accept"

    position_uncertainty_m: float | None = 3.5

    bbox: Any = None

    status: str = "new"


# ============================================================
# 1. MOCK DEYASINI PREPROCESSING
# ============================================================

def mock_preprocess_sss(
    image_path: Path,
) -> Any:

    print("    ✓ Mock preprocessing received:")
    print(f"      {image_path}")

    return {
        "processed": True,
        "source": str(image_path),
        "format": "mock_sss",
    }


# ============================================================
# 2. MOCK FAIZAN AI DETECTION
# ============================================================

def mock_detect_anomalies(
    processed_image: Any,
) -> list[Any]:

    print("    ✓ Mock AI detection running")

    detection = MockDetection()

    print(
        "      Detected: "
        f"{detection.class_name}"
    )

    print(
        "      Confidence: "
        f"{detection.confidence}"
    )

    return [detection]


# ============================================================
# 3. MOCK SHREYASY DETECTION INTELLIGENCE
# ============================================================

def mock_evaluate_detections(
    detections: list[Any],
) -> list[Any]:

    print("    ✓ Mock detection intelligence running")

    accepted = [
        detection
        for detection in detections
        if detection.confidence >= 0.50
    ]

    print(
        f"      Accepted: "
        f"{len(accepted)}/{len(detections)}"
    )

    return accepted


# ============================================================
# 4. MOCK SURAJ GEOLOCATION
# ============================================================

def mock_geolocate(
    detection: Any,
    metadata: dict[str, Any],
) -> Any:

    print("    ✓ Mock geolocation running")

    print(
        "      Latitude : "
        f"{detection.latitude}"
    )

    print(
        "      Longitude: "
        f"{detection.longitude}"
    )

    return detection


# ============================================================
# CREATE TEST IMAGE
# ============================================================

def create_test_image() -> Path:

    test_dir = Path("test_data")

    test_dir.mkdir(
        exist_ok=True
    )

    image_path = (
        test_dir /
        "test_sss_image.png"
    )

    # Minimal valid PNG file.
    png_bytes = bytes.fromhex(
        "89504E470D0A1A0A"
        "0000000D49484452"
        "0000000100000001"
        "08060000001F15C489"
        "0000000A49444154"
        "789C636000000002"
        "0001E221BC33"
        "0000000049454E44"
        "AE426082"
    )

    image_path.write_bytes(
        png_bytes
    )

    return image_path


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PS57 PIPELINE + POSTGRESQL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create test image
    # --------------------------------------------------------

    image_path = create_test_image()

    print()
    print(
        f"Test image: {image_path}"
    )

    # --------------------------------------------------------
    # Test metadata
    # --------------------------------------------------------

    metadata = {
        "source": "pipeline_postgresql_test",
        "latitude": 19.076243,
        "longitude": 72.877931,
        "heading": 90.0,
    }

    # --------------------------------------------------------
    # Connect pipeline dependencies
    # --------------------------------------------------------

    dependencies = PipelineDependencies(
        preprocess_sss=mock_preprocess_sss,

        detect_anomalies=mock_detect_anomalies,

        evaluate_detections=(
            mock_evaluate_detections
        ),

        geolocate=mock_geolocate,

        # IMPORTANT:
        # This is your REAL PostgreSQL persistence
        # function from ps57_pipeline.py.
        persist=persist,
    )

    # --------------------------------------------------------
    # Run complete pipeline
    # --------------------------------------------------------

    result = analyze_sss(
        image_path=image_path,

        metadata=metadata,

        dependencies=dependencies,
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE + DATABASE RESULT")
    print("=" * 70)

    print(
        f"Image      : "
        f"{result.image_path}"
    )

    print(
        f"Detections : "
        f"{result.detection_count}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if result.detection_count == 1:

        print()
        print("=" * 70)
        print(
            "PS57 PIPELINE + POSTGRESQL TEST PASSED"
        )
        print("=" * 70)
        print()

    else:

        print()
        print("=" * 70)
        print(
            "PS57 PIPELINE + POSTGRESQL TEST FAILED"
        )
        print("=" * 70)
        print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()