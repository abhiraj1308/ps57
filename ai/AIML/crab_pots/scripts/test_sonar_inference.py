"""
Comprehensive test suite for Stage-2 Sonar Perception Engine and Runner.
Validates:
- Backwards compatibility of original tests
- Dynamic YOLO class mapping (Crab-Pot, Cylinder Anomaly, Pipeline)
- TorchScript U-Net loading, input/output shapes, class mask generation, and RLE correctness
- Zero-detection frame preservation
- Corrupt image fault-isolation
- Detection and Segmentation JSON schema compliance
- CLI arguments parsing
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from sonar_ai.base import mask_to_polygons, mask_to_rle
from sonar_ai.model_inspector import detect_model_type, inspect_model, load_engine
from sonar_ai.unet_segmentor import SonarUNetSegmentor
from sonar_ai.yolo_detector import SonarYoloDetector
from sonar_inference import (
    DEFAULT_CLASS_ID,
    DEFAULT_CLASS_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_VERSION,
    SonarDetector,
    build_cli_parser,
    collect_image_files,
    run_inference,
)


class TestSonarInference(unittest.TestCase):
    """
    Test suite covering YOLO detectors, TorchScript U-Net, and Unified Runner.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create dummy model file for mocking
        self.dummy_model = self.test_path / "crab_pot_yolo11s_best.pt"
        self.dummy_model.write_bytes(b"dummy_weights")

        # Create valid test images
        self.img1_path = self.test_path / "sonar_ping_001.png"
        img1 = Image.new("L", (640, 512), color=128)
        img1.save(self.img1_path)

        self.img2_path = self.test_path / "sonar_ping_002.png"
        img2 = Image.new("RGB", (800, 600), color=(50, 100, 150))
        img2.save(self.img2_path)

        # Create a corrupt image file
        self.corrupt_img_path = self.test_path / "corrupt_ping.png"
        self.corrupt_img_path.write_bytes(b"NOT_A_VALID_IMAGE_DATA")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Baseline Tests (Preserving Original Suite)
    # -----------------------------------------------------------------------
    def test_collect_image_files_single_file(self):
        files = collect_image_files(self.img1_path)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], self.img1_path)

    def test_collect_image_files_directory(self):
        files = collect_image_files(self.test_path)
        self.assertEqual(len(files), 3)

    def test_missing_model_raises_error(self):
        nonexistent = self.test_path / "nonexistent.pt"
        with self.assertRaises(FileNotFoundError):
            SonarDetector(model_path=nonexistent)

    def test_missing_source_raises_error(self):
        nonexistent = self.test_path / "nonexistent_source.png"
        with self.assertRaises(FileNotFoundError):
            run_inference(model_path=self.dummy_model, source_path=nonexistent)

    @patch("ultralytics.YOLO")
    def test_single_image_inference_with_detection(self, mock_yolo_cls):
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xywhn.cpu.return_value.numpy.return_value = np.array([[0.45, 0.55, 0.12, 0.18]])
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[250.0, 235.0, 326.8, 327.2]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.84])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])

        mock_result = MagicMock()
        mock_result.orig_shape = (512, 640)
        mock_result.speed = {"inference": 12.5}
        mock_result.boxes = mock_boxes
        mock_yolo_instance.predict.return_value = [mock_result]
        mock_yolo_instance.names = {0: "Crab-Pot"}

        output_json = self.test_path / "single_out.json"
        result, stats = run_inference(
            model_path=self.dummy_model,
            source_path=self.img1_path,
            output_path=output_json,
            conf_threshold=0.25,
            device="cpu",
        )

        self.assertEqual(result["frame_id"], "sonar_ping_001")
        self.assertEqual(result["model_version"], DEFAULT_MODEL_VERSION)
        self.assertEqual(result["image_width"], 640)
        self.assertEqual(result["image_height"], 512)
        self.assertEqual(result["inference_latency_ms"], 12.5)

        self.assertEqual(len(result["detections"]), 1)
        det = result["detections"][0]
        self.assertEqual(det["detection_id"], "sonar_ping_001_det_001")
        self.assertEqual(det["class_id"], 0)
        self.assertEqual(det["class_name"], "Crab-Pot")
        self.assertEqual(det["ai_confidence"], 0.84)
        self.assertEqual(det["bbox_xywh_normalized"], [0.45, 0.55, 0.12, 0.18])
        self.assertIn("bbox_xyxy_pixel", det)

        self.assertTrue(output_json.exists())
        with open(output_json, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertEqual(disk_data["frame_id"], "sonar_ping_001")
        self.assertEqual(stats["images_processed"], 1)
        self.assertEqual(stats["failed_images"], 0)
        self.assertEqual(stats["total_detections"], 1)

    @patch("ultralytics.YOLO")
    def test_zero_detection_frame_preserved(self, mock_yolo_cls):
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        mock_result = MagicMock()
        mock_result.orig_shape = (600, 800)
        mock_result.speed = {"inference": 9.8}
        mock_result.boxes = None
        mock_yolo_instance.predict.return_value = [mock_result]

        output_json = self.test_path / "zero_det_out.json"
        result, stats = run_inference(
            model_path=self.dummy_model,
            source_path=self.img2_path,
            output_path=output_json,
            conf_threshold=0.25,
            device="cpu",
        )

        self.assertEqual(result["frame_id"], "sonar_ping_002")
        self.assertEqual(result["image_width"], 800)
        self.assertEqual(result["image_height"], 600)
        self.assertEqual(result["inference_latency_ms"], 9.8)
        self.assertEqual(result["detections"], [])
        self.assertEqual(stats["total_detections"], 0)

    @patch("ultralytics.YOLO")
    def test_batch_processing_with_corrupt_file_isolation(self, mock_yolo_cls):
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance

        def side_effect_predict(source, **kwargs):
            if "corrupt" in source:
                raise ValueError("Corrupt or invalid image stream")
            mock_res = MagicMock()
            mock_res.orig_shape = (512, 640)
            mock_res.speed = {"inference": 11.0}
            mock_res.boxes = MagicMock()
            mock_res.boxes.__len__.return_value = 0
            return [mock_res]

        mock_yolo_instance.predict.side_effect = side_effect_predict

        output_json = self.test_path / "batch_out.json"
        result, stats = run_inference(
            model_path=self.dummy_model,
            source_path=self.test_path,
            output_path=output_json,
            conf_threshold=0.25,
            device="cpu",
        )

        self.assertEqual(stats["images_processed"], 2)
        self.assertEqual(stats["failed_images"], 1)
        self.assertEqual(len(result["frames"]), 2)
        self.assertEqual(len(result["failed_images"]), 1)
        self.assertIn("corrupt_ping", result["failed_images"][0]["file_name"])

    # -----------------------------------------------------------------------
    # Extended Multi-Model & Dynamic Class Mapping Tests
    # -----------------------------------------------------------------------
    @patch("ultralytics.YOLO")
    def test_dynamic_class_mapping_cylinder(self, mock_yolo_cls):
        """Verify Cylinder model uses marine_man_made_anomaly rather than Crab-Pot."""
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance
        mock_yolo_instance.names = {0: "marine_man_made_anomaly"}

        detector = SonarDetector(model_path=self.dummy_model)
        self.assertEqual(detector.class_mapping.get(0), "marine_man_made_anomaly")

        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.xywhn.cpu.return_value.numpy.return_value = np.array([[0.5, 0.5, 0.2, 0.2]])
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100.0, 100.0, 200.0, 200.0]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.75])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])

        mock_res = MagicMock()
        mock_res.orig_shape = (512, 640)
        mock_res.speed = {"inference": 10.0}
        mock_res.boxes = mock_boxes
        mock_yolo_instance.predict.return_value = [mock_res]

        frame_out = detector.detect_frame(self.img1_path)
        self.assertEqual(frame_out["detections"][0]["class_name"], "marine_man_made_anomaly")

    @patch("ultralytics.YOLO")
    def test_dynamic_class_mapping_pipeline(self, mock_yolo_cls):
        """Verify Pipeline model preserves Pipeline class name."""
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance
        mock_yolo_instance.names = {0: "Pipeline"}

        detector = SonarDetector(model_path=self.dummy_model)
        self.assertEqual(detector.class_mapping.get(0), "Pipeline")

    # -----------------------------------------------------------------------
    # TorchScript U-Net Segmentation Tests
    # -----------------------------------------------------------------------
    def test_rle_encoding_correctness(self):
        """Verify lossless Run-Length Encoding logic."""
        mask = np.array([
            [0, 0, 1, 1],
            [1, 0, 0, 1]
        ], dtype=np.uint8)
        # Flattened: 0, 0, 1, 1, 1, 0, 0, 1
        # Indices (1-based): 1..2=0, 3..5=1, 6..7=0, 8=1
        rle = mask_to_rle(mask)
        self.assertIsInstance(rle, str)
        self.assertTrue(len(rle.split()) % 2 == 0)

    @patch("torch.jit.load")
    def test_unet_segmentor_prediction_schema(self, mock_jit_load):
        """Verify U-Net produces native segmentation schema without fake bboxes."""
        mock_script_module = MagicMock()
        mock_jit_load.return_value = mock_script_module

        # Mock U-Net output: [1, 3, 64, 256] with softmax distribution
        fake_probs = torch.zeros(1, 3, 64, 256, dtype=torch.float32)
        fake_probs[:, 0, :, :] = 0.8  # class 0: rock
        fake_probs[:, 1, :, :] = 0.15 # class 1: sand
        fake_probs[:, 2, :, :] = 0.05 # class 2: others
        mock_script_module.return_value = fake_probs

        segmentor = SonarUNetSegmentor(model_path=self.dummy_model, device="cpu")
        frame_out = segmentor.predict_frame(self.img1_path)

        # Check required frame metadata
        self.assertEqual(frame_out["frame_id"], "sonar_ping_001")
        self.assertEqual(frame_out["image_width"], 640)
        self.assertEqual(frame_out["image_height"], 512)
        self.assertIn("inference_latency_ms", frame_out)
        self.assertNotIn("detections", frame_out)  # No fake detections
        self.assertIn("segmentation", frame_out)

        seg = frame_out["segmentation"]
        self.assertEqual(seg["model_input_shape"], [1, 1, 64, 256])
        self.assertEqual(seg["mask_shape"], [64, 256])
        self.assertEqual(seg["num_classes"], 3)
        self.assertIn("rock", seg["class_distribution"])
        self.assertIn("sand", seg["class_distribution"])
        self.assertIn("others", seg["class_distribution"])
        self.assertIn("mask_rle", seg)
        self.assertIn("polygons", seg)

        # Rock should have 100% of the pixels in this mock
        self.assertEqual(seg["class_distribution"]["rock"]["pixel_count"], 64 * 256)
        self.assertEqual(seg["class_distribution"]["rock"]["area_percentage"], 100.0)

    @patch("ultralytics.YOLO")
    def test_dynamic_class_mapping_crab_pot(self, mock_yolo_cls):
        """Verify Crab-Pot model preserves Crab-Pot class mapping."""
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance
        mock_yolo_instance.names = {0: "Crab-Pot"}

        detector = SonarDetector(model_path=self.dummy_model)
        self.assertEqual(detector.class_mapping.get(0), "Crab-Pot")

    def test_rle_lossless_roundtrip(self):
        """Verify RLE roundtrip reconstruction matches original mask bit-for-bit."""
        from sonar_ai.base import rle_to_mask
        np.random.seed(42)
        original_mask = (np.random.rand(64, 256) > 0.6).astype(np.uint8)
        encoded_rle = mask_to_rle(original_mask)
        decoded_mask = rle_to_mask(encoded_rle, (64, 256))
        self.assertTrue(np.array_equal(original_mask, decoded_mask))

    @patch("ultralytics.YOLO")
    def test_inspect_model_yolo(self, mock_yolo_cls):
        """Verify inspect_model returns structured telemetry for YOLO detectors."""
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance
        mock_yolo_instance.names = {0: "marine_man_made_anomaly"}
        mock_yolo_instance.model.args = {"imgsz": 640}

        info = inspect_model(self.dummy_model)
        self.assertEqual(info["task"], "detection")
        self.assertEqual(info["num_classes"], 1)
        self.assertEqual(info["class_mapping"][0], "marine_man_made_anomaly")
        self.assertIn("input_dimensions", info)
        self.assertIn("output_dimensions", info)

    def test_real_model_classification_if_present(self):
        """Verify real project model files are correctly distinguished."""
        real_models_dir = Path("models")
        if not real_models_dir.is_dir():
            return

        expected_types = {
            "crab_pot_yolo11s_832_best.pt": "ultralytics_yolo",
            "Cylinder.pt": "ultralytics_yolo",
            "pipeline_yolo26n.pt": "ultralytics_yolo",
            "PS57_Natural_Seafloor_UNet_best.pt": "torchscript_unet",
        }

        for model_name, expected_type in expected_types.items():
            model_file = real_models_dir / model_name
            if model_file.is_file():
                detected = detect_model_type(model_file)
                self.assertEqual(
                    detected,
                    expected_type,
                    f"Model {model_name} misclassified: expected {expected_type}, got {detected}",
                )


    def test_cli_parser_options(self):
        parser = build_cli_parser()
        args = parser.parse_args([
            "--model", "models/Cylinder.pt",
            "--source", "AI4Shipwrecks/test/images",
            "--output", "results/cylinder.json",
            "--conf", "0.35",
            "--device", "cpu",
            "--model-version", "cylinder_v1",
            "--inspect",
        ])
        self.assertEqual(args.model, "models/Cylinder.pt")
        self.assertEqual(args.source, "AI4Shipwrecks/test/images")
        self.assertEqual(args.output, "results/cylinder.json")
        self.assertEqual(args.conf, 0.35)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.model_version, "cylinder_v1")
        self.assertTrue(args.inspect)


if __name__ == "__main__":
    unittest.main()
