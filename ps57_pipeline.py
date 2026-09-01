from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import sys

from sqlalchemy.exc import SQLAlchemyError

from detection_intelligence.schemas import (
    PS57Detection,
    SSSAnomalyDetection,
)


# ============================================================
# PROJECT / BACKEND PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BACKEND_DIR = PROJECT_ROOT / "backend"

# Allow the root-level pipeline to import backend modules.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from db import SessionLocal
from models.detection import Detection


# ============================================================
# PS57 MASTER PIPELINE
# ============================================================
#
# FINAL FLOW:
#
# SSS IMAGE
#     ↓
# preprocess_sss()
#     ↓
# detect_anomalies()
#     ↓
# evaluate_detections()
#     ↓
# geolocate()
#     ↓
# persist()
#     ↓
# FINAL PS57 RESULTS
#
# This file is the integration/orchestration layer.
# It does NOT implement the algorithms owned by teammates.
# ============================================================


# ============================================================
# PIPELINE DEPENDENCIES
# ============================================================

@dataclass
class PipelineDependencies:
    """
    Functions supplied by the different PS57 modules.

    Deyasini:
        preprocess_sss()

    Faizan:
        detect_anomalies()

    Shreyasy:
        evaluate_detections()

    Suraj:
        geolocate()

    Abhiraj:
        persist()
    """

    preprocess_sss: Callable[
        [Path],
        Any,
    ]

    detect_anomalies: Callable[
        [Any],
        list[SSSAnomalyDetection],
    ]

    evaluate_detections: Callable[
        [list[SSSAnomalyDetection]],
        list[PS57Detection],
    ]

    geolocate: Callable[
        [PS57Detection, dict[str, Any]],
        PS57Detection,
    ]

    persist: Callable[
        [list[PS57Detection]],
        None,
    ]


# ============================================================
# PIPELINE RESULT
# ============================================================

@dataclass
class PipelineResult:
    """
    Final result returned by the PS57 analysis pipeline.
    """

    image_path: str

    detections: list[PS57Detection]

    detection_count: int


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_image(
    image_path: str | Path,
) -> Path:
    """
    Validates the input side-scan sonar image.

    Supported formats:
        PNG
        JPG
        JPEG
        TIF
        TIFF
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"SSS image not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"SSS input is not a file:\n{path}"
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            "Unsupported SSS image format: "
            f"{path.suffix}"
        )

    return path


# ============================================================
# DATABASE PERSISTENCE
# ============================================================

def persist(
    detections: list[PS57Detection],
) -> None:
    """
    Saves final PS57 detections to the existing SQLAlchemy
    database.

    Mapping to the existing Detection table:

        class_name
            -> class_name

        confidence
            -> confidence

        bbox.width
            -> width

        bbox.height
            -> height

        latitude
            -> latitude

        longitude
            -> longitude

        status
            -> status

        severity
            -> priority
    """

    if not detections:
        print(
            "No detections to persist."
        )
        return

    db = SessionLocal()

    try:

        for detection in detections:

            record = Detection(
                class_name=detection.class_name,

                confidence=float(
                    detection.confidence
                ),

                latitude=detection.latitude,

                longitude=detection.longitude,

                width=(
                    float(
                        detection.bbox.width
                    )
                    if detection.bbox is not None
                    else None
                ),

                height=(
                    float(
                        detection.bbox.height
                    )
                    if detection.bbox is not None
                    else None
                ),

                status=(
                    detection.status
                    or "new"
                ),

                priority=(
                    detection.severity
                    or "medium"
                ),
            )

            db.add(record)

        db.commit()

        print(
            "Database persistence complete: "
            f"{len(detections)} detection(s)"
        )

    except SQLAlchemyError as error:

        db.rollback()

        print(
            "DATABASE PERSISTENCE ERROR"
        )

        print(error)

        raise

    finally:

        db.close()


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_sss(
    image_path: str | Path,
    metadata: dict[str, Any],
    dependencies: PipelineDependencies,
) -> PipelineResult:
    """
    Executes the complete PS57 SSS analysis pipeline.

    Steps:

        1. Validate input
        2. Preprocess SSS image
        3. Run AI anomaly detection
        4. Evaluate/filter detections
        5. Geolocate detections
        6. Persist final results
        7. Return final results
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    image = validate_image(
        image_path
    )

    print("\n" + "=" * 70)
    print(
        "PS57 SIDE-SCAN SONAR ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Input image : {image}"
    )

    # ========================================================
    # 2. SSS PREPROCESSING
    # ========================================================

    print(
        "\n[1/5] SSS PREPROCESSING"
    )

    processed_image = (
        dependencies.preprocess_sss(
            image
        )
    )

    print(
        "Preprocessing complete."
    )

    # ========================================================
    # 3. AI ANOMALY DETECTION
    # ========================================================

    print(
        "\n[2/5] AI ANOMALY DETECTION"
    )

    raw_detections = (
        dependencies.detect_anomalies(
            processed_image
        )
    )

    if raw_detections is None:
        raw_detections = []

    print(
        f"Raw AI detections : "
        f"{len(raw_detections)}"
    )

    # ========================================================
    # 4. DETECTION INTELLIGENCE
    # ========================================================

    print(
        "\n[3/5] DETECTION INTELLIGENCE"
    )

    validated_detections = (
        dependencies.evaluate_detections(
            raw_detections
        )
    )

    if validated_detections is None:
        validated_detections = []

    print(
        f"Validated detections : "
        f"{len(validated_detections)}"
    )

    # ========================================================
    # 5. GEOLOCATION
    # ========================================================

    print(
        "\n[4/5] GEOLOCATION"
    )

    located_detections: list[
        PS57Detection
    ] = []

    for detection in validated_detections:

        located = dependencies.geolocate(
            detection,
            metadata,
        )

        located_detections.append(
            located
        )

    print(
        f"Geolocated detections : "
        f"{len(located_detections)}"
    )

    # ========================================================
    # 6. DATABASE PERSISTENCE
    # ========================================================

    print(
        "\n[5/5] DATABASE PERSISTENCE"
    )

    dependencies.persist(
        located_detections
    )

    print(
        f"Stored detections : "
        f"{len(located_detections)}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PS57 ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    return PipelineResult(
        image_path=str(image),

        detections=located_detections,

        detection_count=len(
            located_detections
        ),
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def print_result(
    result: PipelineResult,
) -> None:
    """
    Prints the final PS57 results.
    """

    print("\n")
    print("=" * 70)

    print(
        "PS57 FINAL RESULTS"
    )

    print("=" * 70)

    print(
        f"Input image : "
        f"{result.image_path}"
    )

    print(
        f"Detections  : "
        f"{result.detection_count}"
    )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if not result.detections:

        print(
            "\nNo anomalies detected."
        )

        print(
            "=" * 70
        )

        return

    # --------------------------------------------------------
    # Display detections
    # --------------------------------------------------------

    for index, detection in enumerate(
        result.detections,
        start=1,
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Detection #{index}"
        )

        print(
            "-" * 70
        )

        print(
            f"Class       : "
            f"{detection.class_name}"
        )

        print(
            f"Confidence  : "
            f"{detection.confidence:.3f}"
        )

        print(
            f"Severity    : "
            f"{detection.severity}"
        )

        print(
            f"Decision    : "
            f"{detection.decision}"
        )

        print(
            f"Latitude    : "
            f"{detection.latitude}"
        )

        print(
            f"Longitude   : "
            f"{detection.longitude}"
        )

        print(
            f"Uncertainty : "
            f"{detection.position_uncertainty_m}"
        )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# NO AUTOMATIC EXECUTION
# ============================================================
#
# We intentionally do not run analyze_sss() automatically.
#
# The actual teammate implementations still need to be
# connected to PipelineDependencies.
# ============================================================