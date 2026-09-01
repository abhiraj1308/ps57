import os


# ============================================================
# PS57 DATABASE PERSISTENCE TEST
# ============================================================
#
# This test runs from the Windows terminal.
#
# Our Docker PostgreSQL database is exposed at:
#
#     localhost:5432
#
# The environment variable below tells SQLAlchemy to use
# PostgreSQL instead of the SQLite fallback in backend/db.py.
# ============================================================


os.environ["DATABASE_URL"] = (
    "postgresql://ps57:ps57pass@localhost:5432/ps57"
)


from detection_intelligence.schemas import (
    BoundingBox,
    PS57Detection,
)

from ps57_pipeline import (
    persist,
)


# ============================================================
# CREATE TEST DETECTION
# ============================================================

test_detection = PS57Detection(
    class_name="man_made_anomaly",

    confidence=0.87,

    bbox=BoundingBox(
        x=438.0,
        y=201.0,
        width=96.0,
        height=71.0,
    ),

    severity="high",

    status="new",

    decision="accept",

    latitude=19.076243,

    longitude=72.877931,

    position_uncertainty_m=2.1,

    image_id="integration_test_001",

    timestamp=None,
)


# ============================================================
# DISPLAY TEST DATA
# ============================================================

print("=" * 70)

print(
    "PS57 DATABASE PERSISTENCE TEST"
)

print("=" * 70)

print("\nTest detection:")
print("-" * 70)

print(
    f"Class       : "
    f"{test_detection.class_name}"
)

print(
    f"Confidence  : "
    f"{test_detection.confidence}"
)

print(
    f"Severity    : "
    f"{test_detection.severity}"
)

print(
    f"Decision    : "
    f"{test_detection.decision}"
)

print(
    f"Latitude    : "
    f"{test_detection.latitude}"
)

print(
    f"Longitude   : "
    f"{test_detection.longitude}"
)

print(
    f"Image ID    : "
    f"{test_detection.image_id}"
)


# ============================================================
# PERSIST TO DATABASE
# ============================================================

print(
    "\nSaving detection to PostgreSQL..."
)

persist(
    [test_detection]
)


# ============================================================
# SUCCESS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "PS57 DATABASE PERSISTENCE TEST PASSED"
)

print(
    "=" * 70
)
