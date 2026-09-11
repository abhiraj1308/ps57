import argparse
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from shapely import wkb


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 128
DEFAULT_MIN_BOX_SIZE = 2
DEFAULT_DUPLICATE_IOU = 0.95
DEFAULT_MIN_VISIBLE_FRACTION = 0.50


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="General-purpose sonar annotation to YOLO dataset generator"
)

parser.add_argument(
    "--raster_dir",
    type=Path,
    required=True,
    help="Folder containing sonar GeoTIFF files"
)

parser.add_argument(
    "--annotation_db",
    type=Path,
    required=True,
    help="SQLite database containing annotations"
)

parser.add_argument(
    "--table",
    default="training_stones",
    help="SQLite annotation table"
)

parser.add_argument(
    "--geometry_column",
    default="GEOMETRY",
    help="Geometry column containing WKB"
)

parser.add_argument(
    "--output",
    type=Path,
    default=Path("dataset_yolo_v3")
)

parser.add_argument(
    "--tile_size",
    type=int,
    default=DEFAULT_TILE_SIZE
)

parser.add_argument(
    "--overlap",
    type=int,
    default=DEFAULT_OVERLAP
)

parser.add_argument(
    "--min_box_size",
    type=float,
    default=DEFAULT_MIN_BOX_SIZE
)

parser.add_argument(
    "--duplicate_iou",
    type=float,
    default=DEFAULT_DUPLICATE_IOU
)

parser.add_argument(
    "--min_visible_fraction",
    type=float,
    default=DEFAULT_MIN_VISIBLE_FRACTION
)

args = parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

if args.tile_size <= 0:
    raise ValueError("tile_size must be greater than zero")

if args.overlap < 0 or args.overlap >= args.tile_size:
    raise ValueError("overlap must be >= 0 and smaller than tile_size")

if args.min_box_size <= 0:
    raise ValueError("min_box_size must be greater than zero")

if not 0 < args.duplicate_iou <= 1:
    raise ValueError("duplicate_iou must be between 0 and 1")

if not 0 < args.min_visible_fraction <= 1:
    raise ValueError("min_visible_fraction must be between 0 and 1")


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

image_dir = args.output / "images" / "all"
label_dir = args.output / "labels" / "all"

image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("=" * 70)
print("GENERAL BOULDER / ROCK DATASET PREPARATION V3")
print("=" * 70)

print("\nAnnotation database:")
print(args.annotation_db)

conn = sqlite3.connect(args.annotation_db)

query = f"""
SELECT ogc_fid, {args.geometry_column}
FROM {args.table}
ORDER BY ogc_fid
"""

rows = conn.execute(query).fetchall()
conn.close()

annotations = []

for fid, blob in rows:

    try:
        geometry = wkb.loads(blob)

        if geometry.is_empty:
            continue

        annotations.append(
            {
                "fid": fid,
                "geometry": geometry,
                "bounds": geometry.bounds,
            }
        )

    except Exception as exc:
        print(f"Geometry error for FID {fid}: {exc}")


print("Loaded annotations:", len(annotations))


# ============================================================
# FIND RASTER FILES
# ============================================================

raster_files = sorted(
    list(args.raster_dir.glob("*.tif"))
    + list(args.raster_dir.glob("*.tiff"))
)

if not raster_files:
    raise FileNotFoundError(
        f"No .tif or .tiff files found in {args.raster_dir}"
    )

print("\nRaster files found:", len(raster_files))

for path in raster_files:
    print(" -", path.name)


# ============================================================
# BOUNDING BOX IOU
# ============================================================

def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (
        (ix2 - ix1) *
        (iy2 - iy1)
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# REMOVE DUPLICATE BOXES
# ============================================================

def deduplicate_boxes(boxes, threshold):

    result = []

    for box in boxes:

        is_duplicate = False

        for existing in result:

            if bbox_iou(box, existing) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(box)

    return result


# ============================================================
# VALID PIXEL RATIO
# ============================================================

def valid_pixel_ratio(src, x1, y1, x2, y2):

    x1 = max(0, min(src.width - 1, int(x1)))
    x2 = max(0, min(src.width - 1, int(x2)))
    y1 = max(0, min(src.height - 1, int(y1)))
    y2 = max(0, min(src.height - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    window = rasterio.windows.Window(
        x1,
        y1,
        width,
        height
    )

    data = src.read(
        1,
        window=window
    )

    if src.nodata is None:
        return 1.0

    return float(
        np.mean(data != src.nodata)
    )


# ============================================================
# PROCESS RASTERS
# ============================================================

step = args.tile_size - args.overlap

total_tiles = 0
total_boxes = 0
total_duplicates = 0


for raster_path in raster_files:

    print("\n" + "=" * 70)
    print("PROCESSING:", raster_path.name)
    print("=" * 70)

    with rasterio.open(raster_path) as src:

        print("Size:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)

        # ----------------------------------------------------
        # MATCH ANNOTATIONS TO THIS RASTER
        # ----------------------------------------------------

        raster_boxes = []

        for annotation in annotations:

            minx, miny, maxx, maxy = annotation["bounds"]

            if (
                maxx < src.bounds.left
                or minx > src.bounds.right
                or maxy < src.bounds.bottom
                or miny > src.bounds.top
            ):
                continue

            try:

                left, top = src.index(
                    minx,
                    maxy
                )

                right, bottom = src.index(
                    maxx,
                    miny
                )

                x1 = min(left, right)
                x2 = max(left, right)
                y1 = min(top, bottom)
                y2 = max(top, bottom)

                x1 = max(0, min(src.width - 1, x1))
                x2 = max(0, min(src.width - 1, x2))
                y1 = max(0, min(src.height - 1, y1))
                y2 = max(0, min(src.height - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                ratio = valid_pixel_ratio(
                    src,
                    x1,
                    y1,
                    x2,
                    y2
                )

                if ratio <= 0:
                    continue

                raster_boxes.append(
                    (
                        annotation["fid"],
                        x1,
                        y1,
                        x2,
                        y2,
                        ratio
                    )
                )

            except Exception:
                continue

        print(
            "Candidate annotations:",
            len(raster_boxes)
        )

        # ----------------------------------------------------
        # GEOMETRIC DEDUPLICATION
        # ----------------------------------------------------

        unique = []

        for item in raster_boxes:

            current_box = item[1:5]

            duplicate = False

            for existing in unique:

                existing_box = existing[1:5]

                if bbox_iou(
                    current_box,
                    existing_box
                ) >= args.duplicate_iou:

                    duplicate = True
                    break

            if not duplicate:
                unique.append(item)

        total_duplicates += (
            len(raster_boxes) -
            len(unique)
        )

        raster_boxes = unique

        print(
            "Geometric duplicates removed:",
            len(raster_boxes) -
            len(unique)
        )

        print(
            "Unique annotations:",
            len(raster_boxes)
        )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = src.read(1)

        nodata = src.nodata

        if nodata is not None:
            valid_mask = image != nodata
        else:
            valid_mask = np.ones(
                image.shape,
                dtype=bool
            )

        valid_values = image[valid_mask]

        # ----------------------------------------------------
        # NORMALIZE IMAGE
        # ----------------------------------------------------

        if len(valid_values) > 0:

            low = np.percentile(
                valid_values,
                1
            )

            high = np.percentile(
                valid_values,
                99
            )

            if high > low:

                image = np.clip(
                    (
                        image.astype(np.float32)
                        - low
                    )
                    / (high - low)
                    * 255,
                    0,
                    255
                ).astype(np.uint8)

        # ----------------------------------------------------
        # CREATE TILES
        # ----------------------------------------------------

        tile_number = 0

        for y0 in range(
            0,
            src.height,
            step
        ):

            for x0 in range(
                0,
                src.width,
                step
            ):

                x_end = min(
                    x0 + args.tile_size,
                    src.width
                )

                y_end = min(
                    y0 + args.tile_size,
                    src.height
                )

                actual_width = x_end - x0
                actual_height = y_end - y0

                tile_boxes = []

                # --------------------------------------------
                # COLLECT OBJECTS FOR TILE
                # --------------------------------------------

                for (
                    fid,
                    bx1,
                    by1,
                    bx2,
                    by2,
                    ratio
                ) in raster_boxes:

                    if (
                        bx2 <= x0
                        or bx1 >= x_end
                        or by2 <= y0
                        or by1 >= y_end
                    ):
                        continue

                    original_width = bx2 - bx1
                    original_height = by2 - by1

                    if (
                        original_width <
                        args.min_box_size
                        or
                        original_height <
                        args.min_box_size
                    ):
                        continue

                    tx1 = max(
                        0,
                        bx1 - x0
                    )

                    ty1 = max(
                        0,
                        by1 - y0
                    )

                    tx2 = min(
                        actual_width,
                        bx2 - x0
                    )

                    ty2 = min(
                        actual_height,
                        by2 - y0
                    )

                    clipped_width = tx2 - tx1
                    clipped_height = ty2 - ty1

                    if (
                        clipped_width <
                        args.min_box_size
                        or
                        clipped_height <
                        args.min_box_size
                    ):
                        continue

                    visible_fraction = (
                        clipped_width *
                        clipped_height
                    ) / (
                        original_width *
                        original_height
                    )

                    if (
                        visible_fraction <
                        args.min_visible_fraction
                    ):
                        continue

                    tile_boxes.append(
                        (
                            tx1,
                            ty1,
                            tx2,
                            ty2
                        )
                    )

                # --------------------------------------------
                # REMOVE DUPLICATES INSIDE TILE
                # --------------------------------------------

                tile_boxes = deduplicate_boxes(
                    tile_boxes,
                    args.duplicate_iou
                )

                if not tile_boxes:
                    continue

                # --------------------------------------------
                # EXTRACT IMAGE TILE
                # --------------------------------------------

                tile = image[
                    y0:y_end,
                    x0:x_end
                ]

                padded = np.zeros(
                    (
                        args.tile_size,
                        args.tile_size
                    ),
                    dtype=np.uint8
                )

                padded[
                    :actual_height,
                    :actual_width
                ] = tile

                # --------------------------------------------
                # SAVE IMAGE
                # --------------------------------------------

                filename = (
                    f"{raster_path.stem}_"
                    f"{tile_number:05d}.png"
                )

                image_path = (
                    image_dir /
                    filename
                )

                label_path = (
                    label_dir /
                    filename.replace(
                        ".png",
                        ".txt"
                    )
                )

                Image.fromarray(
                    padded
                ).save(image_path)

                # --------------------------------------------
                # SAVE YOLO LABELS
                # --------------------------------------------

                with open(
                    label_path,
                    "w"
                ) as f:

                    for (
                        tx1,
                        ty1,
                        tx2,
                        ty2
                    ) in tile_boxes:

                        xc = (
                            (tx1 + tx2) / 2
                        ) / args.tile_size

                        yc = (
                            (ty1 + ty2) / 2
                        ) / args.tile_size

                        bw = (
                            tx2 - tx1
                        ) / args.tile_size

                        bh = (
                            ty2 - ty1
                        ) / args.tile_size

                        f.write(
                            f"0 "
                            f"{xc:.6f} "
                            f"{yc:.6f} "
                            f"{bw:.6f} "
                            f"{bh:.6f}\n"
                        )

                total_tiles += 1
                total_boxes += len(tile_boxes)

                tile_number += 1

        print(
            "Positive tiles created:",
            tile_number
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V3 DATASET CREATION COMPLETE")
print("=" * 70)

print("Total tiles:", total_tiles)
print("Total boxes:", total_boxes)
print(
    "Geometric duplicates removed:",
    total_duplicates
)

print("\nOutput:")
print(args.output)

print("\nImages:")
print(image_dir)

print("\nLabels:")
print(label_dir)