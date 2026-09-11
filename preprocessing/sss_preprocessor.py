
"""
Side-Scan Sonar preprocessing module.

Supports:
- PNG/JPG/JPEG/TIF/TIFF sonar images
- XTF files through pyxtf

Main public function:
    preprocess_sss(input_path, output_dir="data/processed")

The module is intentionally conservative:
- it does not delete dark sonar regions or acoustic shadows
- it records missing/invalid data instead of treating every dark region as noise
- XTF navigation fields are preserved in metadata when present
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pyxtf
except ImportError:
    pyxtf = None


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_INPUTS = SUPPORTED_IMAGES | {".xtf"}


def _safe_float(value: Any):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _read_image(path: Path) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint16:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image.astype(np.float32), None, 0, 255,
                              cv2.NORM_MINMAX).astype(np.uint8)

    return image, {"source_type": "image"}


def _channel_array(channel: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(channel)
    except Exception:
        return None

    if arr.size == 0:
        return None

    # pyxtf normally exposes sonar channel samples as a 1-D numpy array.
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
        return arr.astype(np.uint8)

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _read_xtf(path: Path, max_pings: int | None = None) -> tuple[np.ndarray, dict]:
    if pyxtf is None:
        raise ImportError("pyxtf is required for XTF input. Install it with: pip install pyxtf")

    file_header, packets = pyxtf.xtf_read(str(path))

    sonar_type = pyxtf.XTFHeaderType.sonar
    pings = packets.get(sonar_type, [])

    if not pings:
        raise ValueError("No sonar packets/pings were found in the XTF file.")

    if max_pings is not None:
        pings = pings[:max_pings]

    rows = []
    nav = []

    for ping in pings:
        channels = []
        for ch in getattr(ping, "data", []):
            arr = _channel_array(ch)
            if arr is not None and arr.size > 8:
                channels.append(arr)

        if not channels:
            continue

        # Prefer two-channel side-scan layout when available.
        if len(channels) >= 2:
            port = channels[0][::-1]
            starboard = channels[1]
            width = max(len(port), len(starboard))
            port = cv2.resize(port.reshape(1, -1), (width, 1),
                               interpolation=cv2.INTER_LINEAR).ravel()
            starboard = cv2.resize(starboard.reshape(1, -1), (width, 1),
                                   interpolation=cv2.INTER_LINEAR).ravel()
            row = np.concatenate([port, starboard])
        else:
            row = channels[0]

        rows.append(row)

        nav.append({
            "timestamp": _ping_timestamp(ping),
            "latitude_or_y": _safe_float(getattr(ping, "SensorYcoordinate", None)),
            "longitude_or_x": _safe_float(getattr(ping, "SensorXcoordinate", None)),
            "ship_x": _safe_float(getattr(ping, "ShipXcoordinate", None)),
            "heading_deg": _safe_float(getattr(ping, "SensorHeading", None)),
            "pitch_deg": _safe_float(getattr(ping, "SensorPitch", None)),
            "roll_deg": _safe_float(getattr(ping, "SensorRoll", None)),
            "heave": _safe_float(getattr(ping, "Heave", None)),
            "sensor_depth": _safe_float(getattr(ping, "SensorDepth", None)),
            "sensor_altitude": _safe_float(getattr(ping, "SensorPrimaryAltitude", None)),
            "range_to_fish": _safe_float(getattr(ping, "RangeToFish", None)),
            "ping_number": int(getattr(ping, "PingNumber", len(nav))),
        })

    if not rows:
        raise ValueError("XTF contained sonar packets, but no usable channel samples.")

    target_width = max(len(r) for r in rows)
    waterfall = np.zeros((len(rows), target_width), dtype=np.uint8)

    for i, row in enumerate(rows):
        if len(row) != target_width:
            row = cv2.resize(row.reshape(1, -1), (target_width, 1),
                             interpolation=cv2.INTER_LINEAR).ravel()
        waterfall[i] = row

    return waterfall, {
        "source_type": "xtf",
        "xtf_ping_count": len(rows),
        "navigation_samples": nav,
        "xtf_file_header_type": type(file_header).__name__,
    }


def _ping_timestamp(ping: Any) -> str | None:
    try:
        t = ping.time
        return str(t)
    except Exception:
        pass

    fields = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
    if all(hasattr(ping, f) for f in fields):
        return (
            f"{int(ping.Year):04d}-{int(ping.Month):02d}-{int(ping.Day):02d}"
            f"T{int(ping.Hour):02d}:{int(ping.Minute):02d}:{int(ping.Second):02d}"
        )
    return None


def _validate(image: np.ndarray) -> dict:
    finite = np.isfinite(image)
    valid = image[finite]
    if valid.size == 0:
        raise ValueError("Image contains no finite pixels.")

    mean = float(np.mean(valid))
    std = float(np.std(valid))
    min_v = int(np.min(valid))
    max_v = int(np.max(valid))

    blank_pct = float(np.mean(image == 0) * 100)
    dark_pct = float(np.mean(image <= 5) * 100)
    bright_pct = float(np.mean(image >= 250) * 100)

    issues = []
    if image.shape[0] < 16 or image.shape[1] < 16:
        issues.append("image_too_small")
    if std < 2.0:
        issues.append("very_low_variance")
    if blank_pct > 95:
        issues.append("mostly_blank")
    if bright_pct > 95:
        issues.append("mostly_saturated")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "blank_percent": blank_pct,
        "dark_percent": dark_pct,
        "bright_percent": bright_pct,
    }


def _noise_estimate(image: np.ndarray) -> float:
    # Robust local-noise proxy based on median-filter residual.
    med = cv2.medianBlur(image, 3)
    residual = image.astype(np.float32) - med.astype(np.float32)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    return float(np.clip(mad / 64.0, 0, 1))


def _denoise(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(image, 3)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(image, 7, 35, 35)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(image, None, 7, 7, 21)
    if method == "none":
        return image.copy()
    raise ValueError(f"Unknown denoise method: {method}")


def _clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def _normalize(image: np.ndarray, low=1.0, high=99.0) -> np.ndarray:
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    out = (image.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _quality_score(
    original: np.ndarray,
    processed: np.ndarray,
    validation: dict,
) -> tuple[int, dict]:
    noise = _noise_estimate(original)
    contrast = float(np.clip(np.std(processed) / 64.0, 0, 1))
    valid_data = float(np.clip(1.0 - validation["blank_percent"] / 100.0, 0, 1))
    sharp = float(np.clip(math.log1p(_sharpness(processed)) / math.log1p(10000), 0, 1))
    dynamic = float(np.clip((validation["max"] - validation["min"]) / 255.0, 0, 1))

    score = 100.0 * (
        0.30 * (1.0 - noise)
        + 0.25 * contrast
        + 0.20 * valid_data
        + 0.15 * sharp
        + 0.10 * dynamic
    )

    details = {
        "noise_score": round(1.0 - noise, 4),
        "contrast_score": round(contrast, 4),
        "valid_data_score": round(valid_data, 4),
        "sharpness_score": round(sharp, 4),
        "dynamic_range_score": round(dynamic, 4),
        "formula": "0.30 noise + 0.25 contrast + 0.20 valid_data + 0.15 sharpness + 0.10 dynamic_range",
    }
    return int(np.clip(round(score), 0, 100)), details


def preprocess_sss(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
    *,
    denoise_method: str = "bilateral",
    use_clahe: bool = True,
    normalize: bool = True,
    clahe_clip: float = 2.0,
    save_metadata: bool = True,
    max_pings: int | None = None,
) -> dict:
    """
    Main Side-Scan Sonar preprocessing function.

    Returns a dictionary containing:
        image_id
        processed_image
        quality_score
        validation
        quality_metrics
        metadata
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError(
            f"Unsupported input: {input_path.suffix}. "
            f"Supported: {sorted(SUPPORTED_INPUTS)}"
        )

    if input_path.suffix.lower() == ".xtf":
        image, metadata = _read_xtf(input_path, max_pings=max_pings)
    else:
        image, metadata = _read_image(input_path)

    validation = _validate(image)
    if not validation["valid"]:
        raise ValueError(f"Input failed validation: {validation['issues']}")

    original = image.copy()

    # Conservative processing: preserve coherent dark regions/shadows.
    processed = _denoise(image, denoise_method)

    if use_clahe:
        processed = _clahe(processed, clip_limit=clahe_clip)

    if normalize:
        processed = _normalize(processed)

    # Ensure a deterministic 8-bit grayscale output.
    processed = np.ascontiguousarray(processed, dtype=np.uint8)

    image_id = input_path.stem
    output_path = output_dir / f"{image_id}_processed.png"

    ok = cv2.imwrite(str(output_path), processed)
    if not ok:
        raise IOError(f"Failed to write {output_path}")

    quality_score, quality_metrics = _quality_score(
        original, processed, validation
    )

    result = {
        "image_id": image_id,
        "processed_image": str(output_path),
        "quality_score": quality_score,
        "validation": validation,
        "quality_metrics": quality_metrics,
        "metadata": metadata,
        "processing": {
            "denoise_method": denoise_method,
            "clahe": use_clahe,
            "clahe_clip": clahe_clip,
            "percentile_normalization": normalize,
            "output_dtype": "uint8",
            "output_channels": 1,
        },
    }

    if save_metadata:
        metadata_path = output_dir / f"{image_id}_metadata.json"
        metadata_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        result["metadata_file"] = str(metadata_path)

    return result
