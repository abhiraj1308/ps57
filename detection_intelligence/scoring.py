def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Keeps a value inside a defined range.
    """

    return max(minimum, min(value, maximum))


def calculate_fused_confidence(
    vision_confidence: float,
    sonar_intensity: float | None = None,
) -> float:
    """
    Calculates a combined confidence score.

    Vision provides the primary confidence.
    Sonar can increase confidence when a corresponding
    sonar target is present.

    This is intentionally conservative because the
    current sonar pipeline is synthetic.
    """

    vision_confidence = clamp(vision_confidence)

    if sonar_intensity is None:
        return round(vision_confidence, 3)

    sonar_strength = clamp(sonar_intensity)

    # Vision remains the dominant signal.
    fused = (
        vision_confidence * 0.70
        + sonar_strength * 0.30
    )

    return round(clamp(fused), 3)


def calculate_priority(
    confidence: float,
    sonar_intensity: float | None = None,
) -> str:
    """
    Converts confidence and sonar strength into
    a PS57 priority level.
    """

    confidence = clamp(confidence)

    sonar_strength = (
        clamp(sonar_intensity)
        if sonar_intensity is not None
        else 0.0
    )

    if confidence >= 0.85 or sonar_strength >= 0.80:
        return "high"

    if confidence >= 0.60 or sonar_strength >= 0.50:
        return "medium"

    return "low"