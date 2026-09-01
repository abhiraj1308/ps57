from typing import Any

from .schemas import (
    FusedDetection,
    SonarDetection,
    VisionDetection,
)

from .scoring import (
    calculate_fused_confidence,
    calculate_priority,
)


# ============================================================
# PS57 DETECTION FUSION
# ============================================================

def fuse_detections(
    vision_detections: list[VisionDetection],
    sonar_detections: list[SonarDetection],
) -> list[FusedDetection]:
    """
    PS57 sensor-fusion layer.

    Combines observations from:

        Vision / YOLO
              +
        Sonar processing
              ↓
        Unified detections

    Current prototype behaviour:

    1. If both vision and sonar are available:
       - Vision detections are associated with the strongest
         available sonar target.
       - Confidence is calculated from both signals.
       - Priority is recalculated.
       - Fusion status becomes "fused".

    2. If only vision is available:
       - The vision observation is preserved.
       - Fusion status becomes "vision_only".

    3. If only sonar is available:
       - Every sonar target is preserved.
       - Class is reported as "unknown_target".
       - Fusion status becomes "sonar_only".

    This is intentionally a prototype association strategy.
    Later, real range/position/time matching can replace the
    strongest-sonar association.
    """

    fused_results: list[FusedDetection] = []

    # --------------------------------------------------------
    # CASE 1:
    # Vision + Sonar
    # --------------------------------------------------------

    if vision_detections and sonar_detections:

        strongest_sonar = _get_strongest_sonar(
            sonar_detections
        )

        for vision in vision_detections:

            sonar_intensity = strongest_sonar.intensity

            # -----------------------------------------------
            # Combined confidence
            # -----------------------------------------------

            confidence = calculate_fused_confidence(
                vision_confidence=vision.confidence,
                sonar_intensity=sonar_intensity,
            )

            # -----------------------------------------------
            # Combined priority
            # -----------------------------------------------

            priority = calculate_priority(
                confidence=confidence,
                sonar_intensity=sonar_intensity,
            )

            # -----------------------------------------------
            # Unified detection
            # -----------------------------------------------

            fused_results.append(
                FusedDetection(
                    source="vision+sonar",

                    class_name=vision.class_name,

                    confidence=round(
                        confidence,
                        3,
                    ),

                    priority=priority,

                    latitude=vision.latitude,
                    longitude=vision.longitude,

                    width=vision.width,
                    height=vision.height,

                    sonar_range_index=(
                        strongest_sonar.range_index
                    ),

                    sonar_intensity=round(
                        strongest_sonar.intensity,
                        3,
                    ),

                    fusion_status="fused",
                )
            )

        return fused_results

    # --------------------------------------------------------
    # CASE 2:
    # Vision only
    # --------------------------------------------------------

    if vision_detections:

        for vision in vision_detections:

            confidence = calculate_fused_confidence(
                vision_confidence=vision.confidence,
            )

            priority = calculate_priority(
                confidence=confidence,
            )

            fused_results.append(
                FusedDetection(
                    source="vision",

                    class_name=vision.class_name,

                    confidence=round(
                        confidence,
                        3,
                    ),

                    priority=priority,

                    latitude=vision.latitude,
                    longitude=vision.longitude,

                    width=vision.width,
                    height=vision.height,

                    sonar_range_index=None,
                    sonar_intensity=None,

                    fusion_status="vision_only",
                )
            )

        return fused_results

    # --------------------------------------------------------
    # CASE 3:
    # Sonar only
    # --------------------------------------------------------

    if sonar_detections:

        for sonar in sonar_detections:

            confidence = calculate_fused_confidence(
                vision_confidence=0.0,
                sonar_intensity=sonar.intensity,
            )

            priority = calculate_priority(
                confidence=confidence,
                sonar_intensity=sonar.intensity,
            )

            fused_results.append(
                FusedDetection(
                    source="sonar",

                    class_name="unknown_target",

                    confidence=round(
                        confidence,
                        3,
                    ),

                    priority=priority,

                    latitude=None,
                    longitude=None,

                    width=None,
                    height=None,

                    sonar_range_index=(
                        sonar.range_index
                    ),

                    sonar_intensity=round(
                        sonar.intensity,
                        3,
                    ),

                    fusion_status="sonar_only",
                )
            )

        return fused_results

    # --------------------------------------------------------
    # CASE 4:
    # No sensor detections
    # --------------------------------------------------------

    return []


# ============================================================
# SONAR ASSOCIATION
# ============================================================

def _get_strongest_sonar(
    sonar_detections: list[SonarDetection],
) -> SonarDetection | None:
    """
    Returns the sonar target with the highest signal intensity.

    Current prototype association strategy.

    Later this can be replaced by a proper matching algorithm
    using:

        - range
        - timestamp
        - GPS position
        - platform heading
        - sensor orientation
        - spatial distance
    """

    if not sonar_detections:
        return None

    return max(
        sonar_detections,
        key=lambda target: target.intensity,
    )


# ============================================================
# SERIALIZATION
# ============================================================

def fused_detection_to_dict(
    detection: FusedDetection,
) -> dict[str, Any]:
    """
    Converts a FusedDetection object into a standard dictionary.

    Useful for:

        - API communication
        - logging
        - JSON serialization
        - debugging
        - future message queues
    """

    return {
        "source": detection.source,

        "class_name": detection.class_name,

        "confidence": detection.confidence,

        "priority": detection.priority,

        "latitude": detection.latitude,

        "longitude": detection.longitude,

        "width": detection.width,

        "height": detection.height,

        "sonar_range_index": (
            detection.sonar_range_index
        ),

        "sonar_intensity": (
            detection.sonar_intensity
        ),

        "fusion_status": (
            detection.fusion_status
        ),
    }