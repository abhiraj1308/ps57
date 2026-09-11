"""
Underwater Marine Debris and Anomaly Detection System
Stage-2 Integrated Multi-Model Perception Visualizer CLI

PURPOSE:
    Provides the command-line interface to visualize Stage-2 multi-model outputs.
    Reads an integrated evaluation JSON produced by run_stage2.py and the corresponding
    source sonar images, then renders:
    1. Detector bounding boxes (Crab-Pot, Cylinder, Pipeline) labeled with class name
       and raw AI confidence, preserving zero detections without inventing boxes.
    2. Seafloor TorchScript U-Net semantic segmentation as a separate visualization
       decoded from lossless RLE (or polygons), with class legend and area coverage,
       strictly without fabricating bounding boxes or confidence scores.

USAGE:
    python visualize_stage2.py --json results/integrated_demo.json --source demo_inputs --output results/stage2_visualizations
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from sonar_ai.visualizer import Stage2Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VisualizeStage2")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 Integrated Visualizer: Renders detector bounding boxes and "
            "seafloor U-Net semantic segmentation from consolidated Stage-2 JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to the integrated Stage-2 JSON file (e.g. results/integrated_demo.json).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source sonar image directory or single image file (e.g. demo_inputs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/stage2_visualizations",
        help="Target directory to store annotated visualization images.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "jpeg"],
        help="Image format extension for generated visualizations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Alpha opacity blend factor for U-Net seafloor segmentation overlay (0.0 to 1.0).",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=None,
        help="Optional explicit bounding box line thickness in pixels.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=None,
        help="Optional explicit font scale factor.",
    )
    parser.add_argument(
        "--skip-detections",
        action="store_true",
        help="If set, skips generating detection bounding box images.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="If set, skips generating U-Net semantic segmentation images.",
    )

    return parser


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print clean summary banner of generated Stage-2 visualizations.
    """
    banner = "=" * 65
    print("\n" + banner)
    print("STAGE-2 INTEGRATED VISUALIZATION GENERATION COMPLETE")
    print(banner)
    print(f"Integrated JSON Input        : {summary.get('json_path')}")
    print(f"Source Image Path            : {summary.get('source_path')}")
    print(f"Output Directory             : {summary.get('output_directory')}")
    print(f"Total Frames in JSON         : {summary.get('total_frames')}")
    print(f"Frames Successfully Processed: {summary.get('frames_visualized')}")
    print(f"Missing Image Frames         : {summary.get('missing_frames_count')}")

    if summary.get("missing_frames"):
        print("\nMissing Frames Details:")
        for m_id in summary["missing_frames"][:10]:
            print(f"    - {m_id}")
        if len(summary["missing_frames"]) > 10:
            print(f"    ... and {len(summary['missing_frames']) - 10} more.")

    print("\nGenerated Visualizations:")
    print(f"    - Detection Images       : {summary.get('detection_images_generated', 0)} (<frame_id>_detections.{summary.get('format', 'jpg')})")
    print(f"    - Seafloor Segmentations : {summary.get('segmentation_images_generated', 0)} (<frame_id>_segmentation.{summary.get('format', 'jpg')})")
    print("    " + "-" * 57)
    print(f"    Total Visualizations     : {summary.get('total_images_generated', 0)} images saved")
    print(f"\nTarget Output Directory      : {summary.get('output_directory')}")
    print(banner + "\n")


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.is_file():
        sys.stderr.write(
            f"ERROR: Stage-2 integrated JSON file not found: {json_path.resolve()}\n"
            "Please provide a valid JSON path via --json.\n"
        )
        return 1

    source_path = Path(args.source)
    if not source_path.exists():
        sys.stderr.write(
            f"ERROR: Input source does not exist: {source_path.resolve()}\n"
            "Please provide a valid image file or directory path via --source.\n"
        )
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        visualizer = Stage2Visualizer(
            alpha=args.alpha,
            line_thickness=args.line_thickness,
            font_scale=args.font_scale,
        )

        summary = visualizer.visualize_all(
            json_path=json_path,
            source_path=source_path,
            output_dir=output_dir,
            img_format=args.format,
            generate_detections=not args.skip_detections,
            generate_segmentation=not args.skip_segmentation,
        )
        summary["format"] = args.format

        print_summary(summary)
        return 0

    except Exception as exc:
        logger.critical(f"Stage-2 visualization generation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
