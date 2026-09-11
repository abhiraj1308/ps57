"""
Underwater Marine Debris and Anomaly Detection System
Stage-2 Integrated Multi-Model Sonar Perception CLI Entry Point

PURPOSE:
    Provides the integrated CLI entry point to run all four Stage-2 perception models
    across a single sonar image or a batch image directory:
    1. Crab-Pot YOLO11s Detector (models/crab_pot_yolo11s_832_best.pt, conf=0.15)
    2. Cylinder Anomaly YOLO Detector (models/Cylinder.pt, conf=0.25)
    3. Pipeline YOLO Detector (models/pipeline_yolo26n.pt, conf=0.25)
    4. Seafloor Natural TorchScript U-Net Segmentor (models/PS57_Natural_Seafloor_UNet_best.pt)

USAGE:
    python run_stage2.py --source "AI4Shipwrecks/test/images" --output "results/integrated_stage2_eval.json" --device cpu
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from sonar_ai.runner import DEFAULT_STAGE2_MODELS, UnifiedSonarRunner

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RunStage2")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 Integrated Evaluation Runner: Executes Crab-Pot, Cylinder Anomaly, "
            "Pipeline, and Natural Seafloor U-Net models and compiles a single consolidated JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="AI4Shipwrecks/test/images",
        help="Path to a single sonar image file or a directory containing sonar images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/integrated_stage2_eval.json",
        help="Path for the output integrated evaluation JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device ('cpu' or '0' for CUDA GPU).",
    )
    # Optional model path overrides
    parser.add_argument(
        "--crab-pot-model",
        type=str,
        default=None,
        help="Optional path override for Crab-Pot YOLO11s model (.pt).",
    )
    parser.add_argument(
        "--cylinder-model",
        type=str,
        default=None,
        help="Optional path override for Cylinder Anomaly YOLO model (.pt).",
    )
    parser.add_argument(
        "--pipeline-model",
        type=str,
        default=None,
        help="Optional path override for Pipeline YOLO model (.pt).",
    )
    parser.add_argument(
        "--seafloor-model",
        type=str,
        default=None,
        help="Optional path override for Natural Seafloor TorchScript U-Net model (.pt).",
    )
    parser.add_argument(
        "--save-per-frame",
        action="store_true",
        help="If set, also saves individual <frame_id>_integrated.json files alongside the main output.",
    )

    return parser


def print_summary(stats: Dict[str, Any]) -> None:
    """
    Print clean summary banner of the integrated Stage-2 run.
    """
    banner = "=" * 65
    print("\n" + banner)
    print("STAGE-2 INTEGRATED MULTI-MODEL EVALUATION RUN COMPLETE")
    print(banner)
    print(f"Pipeline Version     : {stats.get('pipeline_version')}")
    print(f"Source               : {stats.get('source')}")
    print(f"Device               : {stats.get('device')}")
    print(f"Total Images Found   : {stats.get('total_images')}")
    print(f"Images Processed     : {stats.get('images_processed')}")
    print(f"Failed Images Count  : {stats.get('failed_images')}")
    print("\nDetections / Outputs per Model:")
    dets = stats.get("detections_per_model", {})
    for m_key, count in dets.items():
        print(f"    - {m_key:<16}: {count} candidate detection(s)")
    print(f"    - {'seafloor':<16}: {stats.get('images_processed')} segmentation mask(s)")
    print(f"\nIntegrated Output JSON: {stats.get('output_path')}")
    print(banner + "\n")


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        sys.stderr.write(
            f"ERROR: Input source does not exist: {source_path.resolve()}\n"
            "Please provide a valid image file or directory path via --source.\n"
        )
        return 1

    # Apply optional model path overrides if specified
    models_config_overrides: Dict[str, Dict[str, Any]] = {}
    if args.crab_pot_model:
        models_config_overrides["crab_pot"] = {"model_path": args.crab_pot_model}
    if args.cylinder_model:
        models_config_overrides["cylinder"] = {"model_path": args.cylinder_model}
    if args.pipeline_model:
        models_config_overrides["pipeline"] = {"model_path": args.pipeline_model}
    if args.seafloor_model:
        models_config_overrides["seafloor"] = {"model_path": args.seafloor_model}

    try:
        runner = UnifiedSonarRunner(
            device=args.device,
            models_config=models_config_overrides if models_config_overrides else None,
        )

        _, stats = runner.run(
            source_path=source_path,
            output_path=args.output,
            save_per_frame=args.save_per_frame,
        )

        print_summary(stats)
        return 0

    except Exception as exc:
        logger.critical(f"Integrated Stage-2 evaluation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
