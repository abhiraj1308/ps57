#!/usr/bin/env python3
"""
prepare_yolo_dataset.py

Converts a Hugging Face-style object detection dataset (images + metadata.jsonl)
into a standard Ultralytics YOLO object-detection dataset (images/, labels/, data.yaml).

Usage:
    python prepare_yolo_dataset.py
    python prepare_yolo_dataset.py --source datasets --output yolo_dataset --force
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face object-detection dataset to standard Ultralytics YOLO format."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets",
        help="Path to source dataset root directory containing train/, valid/, and test/ (default: datasets).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="yolo_dataset",
        help="Path to destination directory for the converted YOLO dataset (default: yolo_dataset).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination directory if it already exists.",
    )
    parser.add_argument(
        "--clip-tolerance",
        type=float,
        default=2.0,
        help="Maximum pixel boundary overflow tolerated for safe clipping (default: 2.0 px).",
    )
    return parser.parse_args()


def extract_bboxes_and_categories(
    objects_data: Dict[str, Any]
) -> Tuple[List[List[float]], List[str], Optional[str]]:
    """
    Extracts bboxes and categories from objects dictionary, robustly handling:
    - Empty: bbox: [], category: []
    - Single bbox: bbox: [x, y, w, h], category: ["Cat"] or "Cat"
    - Multi bboxes: bbox: [[x1, y1, w1, h1], ...], category: ["Cat1", "Cat2", ...]
    
    Returns:
        (bboxes_list, categories_list, error_or_warning_message)
    """
    if not isinstance(objects_data, dict):
        return [], [], f"objects field is not a dictionary: {type(objects_data)}"

    raw_bbox = objects_data.get("bbox", [])
    raw_cat = objects_data.get("category", [])

    # Normalize category to list of strings
    if isinstance(raw_cat, str):
        categories = [raw_cat]
    elif isinstance(raw_cat, list):
        categories = [str(c) for c in raw_cat]
    else:
        categories = []

    # Normalize bbox to list of [x, y, w, h] lists
    bboxes: List[List[float]] = []
    if len(raw_bbox) == 0:
        bboxes = []
    elif isinstance(raw_bbox[0], (int, float)):
        # Flat list [x, y, w, h]
        if len(raw_bbox) == 4:
            bboxes = [[float(v) for v in raw_bbox]]
        else:
            return [], categories, f"Invalid flat bbox length: {len(raw_bbox)}"
    elif isinstance(raw_bbox[0], list):
        # Nested list [[x, y, w, h], ...]
        for b in raw_bbox:
            if isinstance(b, list) and len(b) == 4:
                bboxes.append([float(v) for v in b])
            else:
                return [], categories, f"Invalid nested bbox element: {b}"
    else:
        return [], categories, f"Unrecognized bbox structure: {raw_bbox}"

    # Check alignment between bboxes and categories
    if len(bboxes) != len(categories):
        return bboxes, categories, (
            f"Object count mismatch: {len(bboxes)} bboxes vs {len(categories)} categories"
        )

    return bboxes, categories, None


def discover_classes(
    source_dir: Path, splits: Dict[str, str]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Scans metadata files across all splits to discover all unique category names dynamically.
    Returns:
        class_to_id: {class_name: id}
        id_to_class: {id: class_name}
    """
    all_classes: Set[str] = set()

    for split_key in splits.keys():
        meta_file = source_dir / split_key / "metadata.jsonl"
        if not meta_file.is_file():
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    objects = record.get("objects", {})
                    _, categories, _ = extract_bboxes_and_categories(objects)
                    all_classes.update(categories)
                except Exception as e:
                    print(f"[WARN] Error reading class in {meta_file}:{line_idx}: {e}")

    # Deterministic alphabetical sorting for reproducible class mapping
    sorted_classes = sorted(list(all_classes))
    class_to_id = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    id_to_class = {i: cls_name for i, cls_name in enumerate(sorted_classes)}

    return class_to_id, id_to_class


def convert_bbox_to_yolo(
    bbox: List[float],
    img_width: int,
    img_height: int,
    clip_tolerance_px: float = 2.0,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[str], bool]:
    """
    Converts [x_min, y_min, width, height] in pixel coordinates to YOLO
    [center_x, center_y, width, height] normalized to [0.0, 1.0].

    Returns:
        (yolo_coords, error_msg, was_clipped)
    """
    x, y, w, h = bbox

    # Check for non-positive dimensions
    if w <= 0 or h <= 0:
        return None, f"Non-positive dimensions (w={w}, h={h})", False

    x_min = float(x)
    y_min = float(y)
    x_max = x_min + float(w)
    y_max = y_min + float(h)

    # Check if box starts significantly outside image
    if x_min < -clip_tolerance_px or y_min < -clip_tolerance_px:
        return None, f"Box starts outside image (< -{clip_tolerance_px}px): x={x_min}, y={y_min}", False

    # Check if box ends significantly outside image
    if x_max > (img_width + clip_tolerance_px) or y_max > (img_height + clip_tolerance_px):
        return None, (
            f"Box exceeds image bounds by >{clip_tolerance_px}px: "
            f"x_max={x_max} (img_w={img_width}), y_max={y_max} (img_h={img_height})"
        ), False

    # Perform safe clipping for sub-pixel boundary overflows
    orig_x_min, orig_y_min, orig_x_max, orig_y_max = x_min, y_min, x_max, y_max
    x_min = max(0.0, min(x_min, float(img_width)))
    y_min = max(0.0, min(y_min, float(img_height)))
    x_max = max(x_min, min(x_max, float(img_width)))
    y_max = max(y_min, min(y_max, float(img_height)))

    was_clipped = (
        (x_min != orig_x_min)
        or (y_min != orig_y_min)
        or (x_max != orig_x_max)
        or (y_max != orig_y_max)
    )

    clipped_w = x_max - x_min
    clipped_h = y_max - y_min

    if clipped_w <= 0 or clipped_h <= 0:
        return None, f"Box collapsed after clipping: w={clipped_w}, h={clipped_h}", False

    center_x = (x_min + clipped_w / 2.0) / float(img_width)
    center_y = (y_min + clipped_h / 2.0) / float(img_height)
    norm_w = clipped_w / float(img_width)
    norm_h = clipped_h / float(img_height)

    # Check normalized coordinate ranges [0.0, 1.0]
    if not (
        0.0 <= center_x <= 1.0
        and 0.0 <= center_y <= 1.0
        and 0.0 < norm_w <= 1.0
        and 0.0 < norm_h <= 1.0
    ):
        return (
            None,
            f"Normalized coordinates out of bounds: ({center_x}, {center_y}, {norm_w}, {norm_h})",
            was_clipped,
        )

    return (center_x, center_y, norm_w, norm_h), None, was_clipped


def process_split(
    split_source_name: str,
    split_target_name: str,
    source_dir: Path,
    output_dir: Path,
    class_to_id: Dict[str, int],
    clip_tolerance: float = 2.0,
) -> Dict[str, Any]:
    """
    Processes a single split:
    - Reads metadata.jsonl
    - Checks matching image and inspects actual image dimensions via Pillow
    - Copies image files to output/images/{split_target_name}/
    - Converts bboxes to YOLO format and writes output/labels/{split_target_name}/*.txt
    - Tracks comprehensive metrics.
    """
    source_split_dir = source_dir / split_source_name
    meta_file = source_split_dir / "metadata.jsonl"

    target_img_dir = output_dir / "images" / split_target_name
    target_lbl_dir = output_dir / "labels" / split_target_name

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "split_source": split_source_name,
        "split_target": split_target_name,
        "metadata_records": 0,
        "images_in_folder": 0,
        "images_processed": 0,
        "images_with_objects": 0,
        "background_images": 0,
        "images_with_invalid_annotations": 0,
        "total_objects": 0,
        "objects_per_class": Counter(),
        "missing_images": [],
        "duplicate_records": [],
        "invalid_annotations": [],
        "clipped_boxes_count": 0,
        "label_write_failures": [],
    }

    # Count image files in source split folder
    img_extensions = {".jpg", ".jpeg", ".png"}
    source_image_files = {
        p.name
        for p in source_split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in img_extensions
    }
    stats["images_in_folder"] = len(source_image_files)

    if not meta_file.is_file():
        print(f"[ERROR] Metadata file not found: {meta_file}")
        return stats

    seen_files: Set[str] = set()

    with open(meta_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            stats["metadata_records"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                stats["invalid_annotations"].append(f"Line {line_num}: JSON decode error: {e}")
                continue

            file_name = record.get("file_name")
            if not file_name:
                stats["invalid_annotations"].append(f"Line {line_num}: Missing 'file_name'")
                continue

            if file_name in seen_files:
                stats["duplicate_records"].append(file_name)
            seen_files.add(file_name)

            src_img_path = source_split_dir / file_name
            if not src_img_path.is_file():
                stats["missing_images"].append(file_name)
                continue

            # Read actual image dimensions via Pillow
            try:
                with Image.open(src_img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                stats["invalid_annotations"].append(f"{file_name}: Cannot read image dimensions ({e})")
                continue

            objects = record.get("objects", {})
            bboxes, categories, parse_err = extract_bboxes_and_categories(objects)
            if parse_err:
                stats["invalid_annotations"].append(f"{file_name}: {parse_err}")
                if len(objects.get("bbox", [])) > 0 or len(objects.get("category", [])) > 0:
                    stats["images_with_invalid_annotations"] += 1
                continue

            metadata_has_objects = len(bboxes) > 0

            # Generate YOLO annotation lines
            yolo_lines: List[str] = []
            for b_idx, (b, cat) in enumerate(zip(bboxes, categories)):
                if cat not in class_to_id:
                    stats["invalid_annotations"].append(f"{file_name}: Unknown class '{cat}'")
                    continue

                class_id = class_to_id[cat]
                yolo_coords, err_msg, was_clipped = convert_bbox_to_yolo(
                    b, img_w, img_h, clip_tolerance_px=clip_tolerance
                )

                if err_msg:
                    stats["invalid_annotations"].append(f"{file_name} bbox #{b_idx}: {err_msg}")
                    continue

                if was_clipped:
                    stats["clipped_boxes_count"] += 1

                cx, cy, nw, nh = yolo_coords
                # Format to 6 decimal places for high precision
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats["total_objects"] += 1
                stats["objects_per_class"][cat] += 1

            # Case 1: Genuine background image (metadata explicitly contains zero objects)
            if not metadata_has_objects:
                stats["background_images"] += 1

            # Case 2: Metadata contains objects, but ALL annotations failed validation
            elif len(yolo_lines) == 0:
                stats["images_with_invalid_annotations"] += 1
                stats["invalid_annotations"].append(
                    f"{file_name}: Contained {len(bboxes)} object(s), but all failed validation; excluded from dataset."
                )
                continue

            # Case 3: Metadata contains objects, and at least one annotation is valid
            # (Any individual invalid object was already logged to stats['invalid_annotations'])
            else:
                stats["images_with_objects"] += 1

            # 1. Copy image file (preserves original source file)
            dst_img_path = target_img_dir / file_name
            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                stats["label_write_failures"].append(f"Image copy failed for {file_name}: {e}")
                continue

            # 2. Write YOLO label file (.txt with same stem)
            label_name = src_img_path.stem + ".txt"
            dst_lbl_path = target_lbl_dir / label_name
            try:
                with open(dst_lbl_path, "w", encoding="utf-8") as lf:
                    if yolo_lines:
                        lf.write("\n".join(yolo_lines) + "\n")
                    else:
                        # Empty .txt file for negative/background images
                        pass
            except Exception as e:
                stats["label_write_failures"].append(f"Label write failed for {label_name}: {e}")
                continue

            stats["images_processed"] += 1

            # Progress output every 1000 images
            if stats["images_processed"] % 1000 == 0:
                print(f"  [{split_target_name}] Processed {stats['images_processed']} / {stats['images_in_folder']} images...")

    return stats


def create_yaml(
    output_dir: Path,
    splits: Dict[str, str],
    id_to_class: Dict[int, str],
) -> Path:
    """
    Creates data.yaml formatted for Ultralytics YOLO.
    Uses forward slashes for Windows path compatibility.
    """
    yaml_path = output_dir / "data.yaml"
    abs_output_path = output_dir.resolve().as_posix()

    lines = [
        f"# YOLO Object Detection Dataset Configuration",
        f"# Generated by prepare_yolo_dataset.py",
        f"",
        f"path: {abs_output_path}",
        f"train: images/{splits['train']}",
        f"val: images/{splits['valid']}",
        f"test: images/{splits['test']}",
        f"",
        f"nc: {len(id_to_class)}",
        f"",
        f"names:",
    ]

    for class_id in sorted(id_to_class.keys()):
        lines.append(f"  {class_id}: {id_to_class[class_id]}")

    lines.append("")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return yaml_path


def validate_generated_dataset(
    output_dir: Path,
    splits: Dict[str, str],
    all_stats: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Validates the generated YOLO dataset:
    - Checks data.yaml existence and readability
    - Verifies 1:1 image and label counts per split
    - Verifies every image has an exact matching .txt label file
    - Verifies annotation line syntax and range checks [0, 1]
    - Verifies background images have 0-byte label files
    """
    issues = []

    yaml_file = output_dir / "data.yaml"
    if not yaml_file.is_file():
        issues.append("data.yaml does not exist.")

    for split_source, split_target in splits.items():
        img_dir = output_dir / "images" / split_target
        lbl_dir = output_dir / "labels" / split_target

        if not img_dir.is_dir():
            issues.append(f"Missing images directory: {img_dir}")
            continue
        if not lbl_dir.is_dir():
            issues.append(f"Missing labels directory: {lbl_dir}")
            continue

        images = list(img_dir.iterdir())
        labels = list(lbl_dir.iterdir())

        expected_count = all_stats[split_source]["images_processed"]
        if len(images) != expected_count:
            issues.append(f"[{split_target}] Image count mismatch: expected {expected_count}, found {len(images)}")
        if len(labels) != expected_count:
            issues.append(f"[{split_target}] Label count mismatch: expected {expected_count}, found {len(labels)}")

        # Verify 1:1 matching and line format
        sample_checked = 0
        for img_path in images:
            expected_lbl = lbl_dir / (img_path.stem + ".txt")
            if not expected_lbl.is_file():
                issues.append(f"[{split_target}] Missing label file for image: {img_path.name}")
                break

            # Spot-check label file contents
            if sample_checked < 200:
                with open(expected_lbl, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                    for line_idx, line in enumerate(lines, start=1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(
                                f"[{split_target}] Invalid format in {expected_lbl.name}:{line_idx} - expected 5 items, got {len(parts)}"
                            )
                            break
                        try:
                            cid = int(parts[0])
                            coords = [float(p) for p in parts[1:]]
                            for c in coords:
                                if not (0.0 <= c <= 1.0):
                                    issues.append(
                                        f"[{split_target}] Coordinate out of range [0, 1] in {expected_lbl.name}: {c}"
                                    )
                                    break
                        except ValueError as ve:
                            issues.append(f"[{split_target}] Non-numeric value in {expected_lbl.name}: {ve}")
                            break
                sample_checked += 1

    return len(issues) == 0, issues


def print_summary(
    all_stats: Dict[str, Dict[str, Any]],
    id_to_class: Dict[int, str],
    output_dir: Path,
    yaml_created: bool,
    validation_passed: bool,
    validation_issues: List[str],
) -> None:
    """
    Prints a clean, comprehensive summary matching required specifications.
    """
    print("\n" + "=" * 65)
    print("        YOLO DATASET PREPARATION & VALIDATION SUMMARY")
    print("=" * 65)

    print("\nClasses:")
    for class_id in sorted(id_to_class.keys()):
        print(f"  {class_id}: {id_to_class[class_id]}")

    total_images_all = 0
    total_objects_all = 0
    total_missing_all = 0
    total_duplicates_all = 0
    total_invalid_annots_all = 0
    total_invalid_images_all = 0
    total_clipped_boxes = 0
    per_class_total = Counter()

    for split_source, stats in all_stats.items():
        split_display = stats["split_target"].upper()
        if split_display == "VAL":
            split_display = "VALIDATION (val)"

        print(f"\n{split_display}:")
        print(f"  Images: {stats['images_processed']}")
        print(f"  Images with objects: {stats['images_with_objects']}")
        print(f"  Background images: {stats['background_images']}")
        if stats.get("images_with_invalid_annotations", 0) > 0:
            print(f"  Images with invalid annotations: {stats['images_with_invalid_annotations']}")
        print(f"  Objects: {stats['total_objects']}")
        if stats["clipped_boxes_count"] > 0:
            print(f"  Sub-pixel boundary clipped boxes: {stats['clipped_boxes_count']}")

        total_images_all += stats["images_processed"]
        total_objects_all += stats["total_objects"]
        total_missing_all += len(stats["missing_images"])
        total_duplicates_all += len(stats["duplicate_records"])
        total_invalid_annots_all += len(stats["invalid_annotations"])
        total_invalid_images_all += stats.get("images_with_invalid_annotations", 0)
        total_clipped_boxes += stats["clipped_boxes_count"]

        for cat, cnt in stats["objects_per_class"].items():
            per_class_total[cat] += cnt

    print("\n" + "-" * 65)
    print(f"Total images: {total_images_all}")
    print(f"Total objects: {total_objects_all}")
    print(f"Total sub-pixel clipped boxes: {total_clipped_boxes}")

    print("\nPer-Class Object Counts:")
    for class_id in sorted(id_to_class.keys()):
        cat_name = id_to_class[class_id]
        print(f"  Class {class_id} ({cat_name}): {per_class_total[cat_name]}")

    print(f"\nMissing images: {total_missing_all}")
    print(f"Duplicate references: {total_duplicates_all}")
    print(f"Images with invalid annotations: {total_invalid_images_all}")
    print(f"Invalid annotations: {total_invalid_annots_all}")
    if total_invalid_annots_all > 0:
        print("  Sample invalid annotations:")
        for s in all_stats.values():
            for err in s["invalid_annotations"][:5]:
                print(f"    {err}")

    print(f"\nOutput:\n{output_dir.resolve()}")
    print(f"\ndata.yaml created: {'YES' if yaml_created else 'NO'}")
    print(f"Final Dataset Validation: {'PASSED' if validation_passed else 'FAILED'}")

    if not validation_passed:
        print("\nValidation Issues:")
        for issue in validation_issues:
            print(f"  - {issue}")
    print("=" * 65 + "\n")


def main() -> int:
    args = parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    print("=== YOLO Dataset Preparation Tool ===")
    print(f"Source directory: {source_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    # 1. Check source existence
    if not source_dir.is_dir():
        print(f"[ERROR] Source dataset directory does not exist: {source_dir.resolve()}")
        return 1

    # Define splits mapping (source_name -> yolo_target_name)
    splits = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    # Verify each split exists in source
    for s in splits.keys():
        if not (source_dir / s).is_dir():
            print(f"[ERROR] Source split directory not found: {(source_dir / s).resolve()}")
            return 1

    # 2. Safety check: Protect existing output directory
    if output_dir.exists():
        if not args.force:
            print(f"\n[SAFETY WARNING] Destination directory already exists: {output_dir.resolve()}")
            print("To prevent accidental data overwrite, please supply the --force flag to proceed:")
            print(f"    python prepare_yolo_dataset.py --force\n")
            return 1
        else:
            print(f"[INFO] --force specified. Cleaning existing output directory '{output_dir.resolve()}'...")
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Discover unique classes dynamically
    print("\nDiscovering unique classes across all metadata files...")
    class_to_id, id_to_class = discover_classes(source_dir, splits)

    if not class_to_id:
        print("[ERROR] No classes discovered. Please check metadata.jsonl files.")
        return 1

    print(f"Discovered {len(class_to_id)} classes:")
    for cid, cname in id_to_class.items():
        print(f"  {cid}: {cname}")

    # 4. Process each split
    all_stats: Dict[str, Dict[str, Any]] = {}
    for split_source, split_target in splits.items():
        print(f"\nProcessing '{split_source}' -> '{split_target}'...")
        stats = process_split(
            split_source_name=split_source,
            split_target_name=split_target,
            source_dir=source_dir,
            output_dir=output_dir,
            class_to_id=class_to_id,
            clip_tolerance=args.clip_tolerance,
        )
        all_stats[split_source] = stats
        print(f"  Completed '{split_target}': {stats['images_processed']} images processed.")

    # 5. Create data.yaml
    print("\nGenerating data.yaml...")
    yaml_path = create_yaml(output_dir, splits, id_to_class)
    yaml_created = yaml_path.is_file()

    # 6. Validate generated dataset
    print("Validating generated dataset structure and labels...")
    validation_passed, validation_issues = validate_generated_dataset(output_dir, splits, all_stats)

    # 7. Print summary
    print_summary(
        all_stats=all_stats,
        id_to_class=id_to_class,
        output_dir=output_dir,
        yaml_created=yaml_created,
        validation_passed=validation_passed,
        validation_issues=validation_issues,
    )

    return 0 if validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
