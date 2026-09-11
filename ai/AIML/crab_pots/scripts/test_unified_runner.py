"""
Unit and Integration Test Suite for Stage-2 Integrated UnifiedSonarRunner.

Validates:
1. Integrated execution across all 4 models on a single image.
2. Batch execution across multiple images.
3. Strict preservation of zero YOLO detections as detections: [].
4. Strict preservation of U-Net segmentation output without fake bboxes/confidence.
5. Fault isolation: one model failure on a frame does not stop other models or the run.
6. Corrupt image fault isolation into failed_images.
7. Output JSON generation and schema compliance.
8. Stage-2 Contract Rules: no fabricated Stage-3 fields, raw confidence only, normalized coordinates in [0, 1].
9. Backward compatibility for legacy single-model mode.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from run_stage2 import build_cli_parser
from sonar_ai.runner import DEFAULT_STAGE2_MODELS, UnifiedSonarRunner


class TestUnifiedSonarRunner(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create valid test images
        self.img1_path = self.test_path / "sonar_ping_001.png"
        img1 = Image.new("L", (640, 512), color=128)
        img1.save(self.img1_path)

        self.img2_path = self.test_path / "sonar_ping_002.png"
        img2 = Image.new("RGB", (800, 600), color=(50, 100, 150))
        img2.save(self.img2_path)

        # Create mock outputs for each of the 4 models conforming to Stage-2 schema
        self.mock_crab_pot_output = {
            "frame_id": "sonar_ping_001",
            "model_version": "yolo11s_crab_pot_832_v1",
            "image_width": 640,
            "image_height": 512,
            "inference_latency_ms": 14.2,
            "detections": [
                {
                    "detection_id": "sonar_ping_001_det_001",
                    "class_id": 0,
                    "class_name": "Crab-Pot",
                    "ai_confidence": 0.88,
                    "bbox_xywh_normalized": [0.5, 0.5, 0.1, 0.2],
                    "bbox_xyxy_pixel": [288.0, 204.8, 352.0, 307.2],
                }
            ],
        }

        self.mock_cylinder_output = {
            "frame_id": "sonar_ping_001",
            "model_version": "cylinder_yolo_v1",
            "image_width": 640,
            "image_height": 512,
            "inference_latency_ms": 11.5,
            "detections": [],  # Zero detection
        }

        self.mock_pipeline_output = {
            "frame_id": "sonar_ping_001",
            "model_version": "pipeline_yolo26n_v1",
            "image_width": 640,
            "image_height": 512,
            "inference_latency_ms": 10.8,
            "detections": [
                {
                    "detection_id": "sonar_ping_001_det_001",
                    "class_id": 0,
                    "class_name": "Pipeline",
                    "ai_confidence": 0.72,
                    "bbox_xywh_normalized": [0.4, 0.6, 0.3, 0.05],
                    "bbox_xyxy_pixel": [160.0, 294.4, 352.0, 320.0],
                }
            ],
        }

        self.mock_seafloor_output = {
            "frame_id": "sonar_ping_001",
            "model_version": "seafloor_unet_ts_v1",
            "image_width": 640,
            "image_height": 512,
            "inference_latency_ms": 16.5,
            "segmentation": {
                "model_input_shape": [1, 1, 64, 256],
                "mask_shape": [64, 256],
                "num_classes": 3,
                "class_names": {"0": "rock", "1": "sand", "2": "others"},
                "class_distribution": {
                    "rock": {"class_id": 0, "pixel_count": 13000, "area_percentage": 79.35, "mean_softmax_probability": 0.89},
                    "sand": {"class_id": 1, "pixel_count": 3300, "area_percentage": 20.14, "mean_softmax_probability": 0.78},
                    "others": {"class_id": 2, "pixel_count": 84, "area_percentage": 0.51, "mean_softmax_probability": 0.52},
                },
                "mask_rle": {"rock": "1 13000", "sand": "13001 3300", "others": "16301 84"},
                "polygons": [
                    {
                        "class_id": 0,
                        "class_name": "rock",
                        "area_pixels": 13000.0,
                        "contour_points_count": 12,
                        "polygon_normalized": [[0.0, 0.0], [0.8, 0.0], [0.8, 1.0], [0.0, 1.0]],
                    }
                ],
            },
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_adapters(self):
        """Helper to create 4 mock adapters returning standard Stage-2 outputs."""
        cp_adapter = MagicMock()
        cp_adapter.model_version = "yolo11s_crab_pot_832_v1"
        cp_adapter.predict_frame.return_value = self.mock_crab_pot_output

        cyl_adapter = MagicMock()
        cyl_adapter.model_version = "cylinder_yolo_v1"
        cyl_adapter.predict_frame.return_value = self.mock_cylinder_output

        pip_adapter = MagicMock()
        pip_adapter.model_version = "pipeline_yolo26n_v1"
        pip_adapter.predict_frame.return_value = self.mock_pipeline_output

        sea_adapter = MagicMock()
        sea_adapter.model_version = "seafloor_unet_ts_v1"
        sea_adapter.predict_frame.return_value = self.mock_seafloor_output

        return {
            "crab_pot": cp_adapter,
            "cylinder": cyl_adapter,
            "pipeline": pip_adapter,
            "seafloor": sea_adapter,
        }

    # -----------------------------------------------------------------------
    # 1. Single Image Integrated Test
    # -----------------------------------------------------------------------
    def test_single_image_integrated_run(self):
        adapters = self._create_mock_adapters()
        runner = UnifiedSonarRunner(adapters=adapters)

        out_json = self.test_path / "results" / "integrated_single.json"
        result, stats = runner.run(source_path=self.img1_path, output_path=out_json)

        # Output JSON file on disk exists
        self.assertTrue(out_json.exists())
        with open(out_json, "r", encoding="utf-8") as f:
            disk_data = json.load(f)

        self.assertEqual(disk_data["pipeline_version"], "stage2_integrated_v1")
        self.assertEqual(disk_data["metadata"]["total_images"], 1)
        self.assertEqual(disk_data["metadata"]["images_processed"], 1)
        self.assertEqual(disk_data["metadata"]["failed_images_count"], 0)
        self.assertEqual(len(disk_data["frames"]), 1)

        frame = disk_data["frames"][0]
        self.assertEqual(frame["frame_id"], "sonar_ping_001")
        self.assertEqual(frame["image_width"], 640)
        self.assertEqual(frame["image_height"], 512)
        self.assertIn("models", frame)
        self.assertIn("crab_pot", frame["models"])
        self.assertIn("cylinder", frame["models"])
        self.assertIn("pipeline", frame["models"])
        self.assertIn("seafloor", frame["models"])

        # Check model latency telemetry preserved
        self.assertEqual(frame["models"]["crab_pot"]["inference_latency_ms"], 14.2)
        self.assertEqual(frame["models"]["cylinder"]["inference_latency_ms"], 11.5)
        self.assertEqual(frame["models"]["pipeline"]["inference_latency_ms"], 10.8)
        self.assertEqual(frame["models"]["seafloor"]["inference_latency_ms"], 16.5)

    # -----------------------------------------------------------------------
    # 2. Multiple Images Batch Test
    # -----------------------------------------------------------------------
    def test_multiple_images_batch_run(self):
        adapters = self._create_mock_adapters()
        runner = UnifiedSonarRunner(adapters=adapters)

        out_json = self.test_path / "results" / "integrated_batch.json"
        result, stats = runner.run(source_path=self.test_path, output_path=out_json)

        self.assertEqual(stats["total_images"], 2)
        self.assertEqual(stats["images_processed"], 2)
        self.assertEqual(stats["failed_images"], 0)
        self.assertEqual(len(result["frames"]), 2)

        frame_ids = [f["frame_id"] for f in result["frames"]]
        self.assertIn("sonar_ping_001", frame_ids)
        self.assertIn("sonar_ping_002", frame_ids)

    # -----------------------------------------------------------------------
    # 3. Zero YOLO Detections Preservation Test
    # -----------------------------------------------------------------------
    def test_zero_yolo_detections_preserved(self):
        adapters = self._create_mock_adapters()
        runner = UnifiedSonarRunner(adapters=adapters)

        out_json = self.test_path / "zero_det.json"
        result, _ = runner.run(source_path=self.img1_path, output_path=out_json)

        cylinder_output = result["frames"][0]["models"]["cylinder"]
        self.assertEqual(cylinder_output["detections"], [])
        self.assertIsInstance(cylinder_output["detections"], list)

    # -----------------------------------------------------------------------
    # 4. U-Net Segmentation Schema Preservation Test
    # -----------------------------------------------------------------------
    def test_unet_segmentation_preserved(self):
        adapters = self._create_mock_adapters()
        runner = UnifiedSonarRunner(adapters=adapters)

        out_json = self.test_path / "unet_seg.json"
        result, _ = runner.run(source_path=self.img1_path, output_path=out_json)

        seafloor_output = result["frames"][0]["models"]["seafloor"]
        self.assertNotIn("detections", seafloor_output)  # No fake bboxes
        self.assertIn("segmentation", seafloor_output)

        seg = seafloor_output["segmentation"]
        self.assertEqual(seg["mask_shape"], [64, 256])
        self.assertEqual(seg["num_classes"], 3)
        self.assertIn("rock", seg["class_distribution"])
        self.assertIn("sand", seg["class_distribution"])
        self.assertIn("others", seg["class_distribution"])
        self.assertIn("mask_rle", seg)
        self.assertIn("polygons", seg)

    # -----------------------------------------------------------------------
    # 5. Fault Isolation: One Model Fails on Frame
    # -----------------------------------------------------------------------
    def test_one_model_failure_does_not_stop_others(self):
        adapters = self._create_mock_adapters()
        # Make cylinder detector fail with an error
        adapters["cylinder"].predict_frame.side_effect = RuntimeError("CUDA OOM in cylinder inference")

        runner = UnifiedSonarRunner(adapters=adapters)
        out_json = self.test_path / "model_fail.json"
        result, stats = runner.run(source_path=self.img1_path, output_path=out_json)

        # Run still succeeded and processed frame
        self.assertEqual(stats["images_processed"], 1)
        frame = result["frames"][0]

        # Cylinder model output has error details
        self.assertTrue(frame["models"]["cylinder"]["failed"])
        self.assertIn("CUDA OOM", frame["models"]["cylinder"]["error"])

        # Other three models succeeded completely
        self.assertNotIn("failed", frame["models"]["crab_pot"])
        self.assertNotIn("failed", frame["models"]["pipeline"])
        self.assertNotIn("failed", frame["models"]["seafloor"])
        self.assertEqual(len(frame["models"]["crab_pot"]["detections"]), 1)
        self.assertIn("segmentation", frame["models"]["seafloor"])

    # -----------------------------------------------------------------------
    # 6. Corrupt Image Fault Isolation Test
    # -----------------------------------------------------------------------
    def test_corrupt_image_fault_isolation(self):
        corrupt_img = self.test_path / "corrupt_sonar.png"
        corrupt_img.write_bytes(b"NON_IMAGE_DATA_CORRUPT")

        adapters = self._create_mock_adapters()
        runner = UnifiedSonarRunner(adapters=adapters)

        out_json = self.test_path / "corrupt_test.json"
        result, stats = runner.run(source_path=self.test_path, output_path=out_json)

        self.assertEqual(stats["total_images"], 3)
        self.assertEqual(stats["images_processed"], 2)
        self.assertEqual(stats["failed_images"], 1)
        self.assertEqual(len(result["failed_images"]), 1)
        self.assertEqual(result["failed_images"][0]["file_name"], "corrupt_sonar.png")

    # -----------------------------------------------------------------------
    # 7. Strict Stage-2 Contract Rules Test
    # -----------------------------------------------------------------------
    def test_stage2_contract_rules(self):
        adapters = self._create_mock_adapters()
        runner = UnifiedSonarRunner(adapters=adapters)

        out_json = self.test_path / "contract_test.json"
        result, _ = runner.run(source_path=self.img1_path, output_path=out_json)

        # Disallowed Stage-3 fabricated fields must not exist anywhere
        forbidden_fields = [
            "shadow_alignment_score",
            "geological_score",
            "temporal_continuity",
            "final_confidence",
            "priority",
            "status",
            "validation",
        ]

        def assert_no_forbidden(d, path=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    self.assertNotIn(
                        k,
                        forbidden_fields,
                        f"Forbidden Stage-3 field '{k}' found at path '{path}.{k}'",
                    )
                    assert_no_forbidden(v, f"{path}.{k}")
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    assert_no_forbidden(v, f"{path}[{i}]")

        assert_no_forbidden(result)

        # Check normalized coords are in [0, 1]
        for frame in result["frames"]:
            for m_key in ["crab_pot", "cylinder", "pipeline"]:
                for det in frame["models"][m_key].get("detections", []):
                    bbox_norm = det["bbox_xywh_normalized"]
                    for coord in bbox_norm:
                        self.assertTrue(0.0 <= coord <= 1.0)
                    self.assertIn("ai_confidence", det)

    # -----------------------------------------------------------------------
    # 8. Backward Compatibility: Single-Model Mode Test
    # -----------------------------------------------------------------------
    @patch("sonar_ai.runner.load_engine")
    def test_backward_compatibility_single_model_mode(self, mock_load_engine):
        mock_engine = MagicMock()
        mock_engine.model_version = "single_model_v1"
        mock_engine.device = "cpu"
        mock_engine.predict_frame.return_value = {
            "frame_id": "sonar_ping_001",
            "model_version": "single_model_v1",
            "image_width": 640,
            "image_height": 512,
            "inference_latency_ms": 10.0,
            "detections": [],
        }
        mock_load_engine.return_value = mock_engine

        dummy_model = self.test_path / "single_test_model.pt"
        dummy_model.write_bytes(b"dummy")

        runner = UnifiedSonarRunner(
            model_path=dummy_model,
            conf_threshold=0.3,
            device="cpu",
            model_version="single_model_v1",
        )
        self.assertTrue(runner.is_single_model)

        out_json = self.test_path / "single_mode_out.json"
        result, stats = runner.run(source_path=self.img1_path, output_path=out_json)

        self.assertIn("detections", result)
        self.assertEqual(stats["images_processed"], 1)

    # -----------------------------------------------------------------------
    # 9. CLI Argument Parser Test
    # -----------------------------------------------------------------------
    def test_cli_parser_defaults_and_overrides(self):
        parser = build_cli_parser()
        args = parser.parse_args([])
        self.assertEqual(args.source, "AI4Shipwrecks/test/images")
        self.assertEqual(args.output, "results/integrated_stage2_eval.json")
        self.assertEqual(args.device, "cpu")

        args = parser.parse_args([
            "--source", "custom/path",
            "--output", "custom_out.json",
            "--device", "0",
            "--crab-pot-model", "models/custom_crab.pt",
            "--save-per-frame",
        ])
        self.assertEqual(args.source, "custom/path")
        self.assertEqual(args.output, "custom_out.json")
        self.assertEqual(args.device, "0")
        self.assertEqual(args.crab_pot_model, "models/custom_crab.pt")
        self.assertTrue(args.save_per_frame)


if __name__ == "__main__":
    unittest.main()
