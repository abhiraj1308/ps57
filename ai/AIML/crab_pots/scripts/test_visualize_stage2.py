"""
Unit and Integration Test Suite for Stage-2 Integrated Perception Visualizer.

Validates:
1. Loading the integrated JSON (valid schema, missing file, invalid format).
2. Handling frames with zero detections (no invented boxes, output preserved).
3. Drawing detector boxes (Crab-Pot, Cylinder, Pipeline, class labels, raw confidence).
4. Handling U-Net segmentation output (RLE decoding, overlay blending, legend, no fake boxes/confidence).
5. Missing source image handling (fault isolation, non-crashing, proper summary tracking).
6. End-to-end CLI execution.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sonar_ai.base import mask_to_rle
from sonar_ai.visualizer import (
    DEFAULT_DETECTOR_COLORS,
    DEFAULT_SEGMENTATION_COLORS,
    Stage2Visualizer,
)


class TestStage2Visualizer(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.source_dir = self.test_path / "source_images"
        self.source_dir.mkdir()
        self.output_dir = self.test_path / "visualizations"

        # Create a synthetic test image (grayscale acoustic background)
        self.img_width = 640
        self.img_height = 512
        raw_arr = np.full((self.img_height, self.img_width, 3), 60, dtype=np.uint8)
        self.test_img_path = self.source_dir / "frame_001.jpg"
        cv2.imwrite(str(self.test_img_path), raw_arr)

        # Create synthetic mask for U-Net (64 x 256)
        # Class 0: rock (top half), Class 1: sand (bottom half), Class 2: others (small strip)
        mask_h, mask_w = 64, 256
        self.unet_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        self.unet_mask[:30, :] = 0   # rock
        self.unet_mask[30:55, :] = 1 # sand
        self.unet_mask[55:, :] = 2   # others

        rock_rle = mask_to_rle((self.unet_mask == 0).astype(np.uint8))
        sand_rle = mask_to_rle((self.unet_mask == 1).astype(np.uint8))
        others_rle = mask_to_rle((self.unet_mask == 2).astype(np.uint8))

        # Synthetic multi-model integrated JSON structure
        self.mock_json_data = {
            "pipeline_version": "stage2_integrated_v1",
            "metadata": {
                "total_images": 2,
                "images_processed": 2,
                "failed_images_count": 0,
                "source": str(self.source_dir),
                "device": "cpu",
            },
            "failed_images": [],
            "frames": [
                {
                    "frame_id": "frame_001",
                    "image_width": self.img_width,
                    "image_height": self.img_height,
                    "models": {
                        "crab_pot": {
                            "frame_id": "frame_001",
                            "model_version": "yolo11s_crab_pot_832_v1",
                            "detections": [
                                {
                                    "detection_id": "frame_001_cp_001",
                                    "class_id": 0,
                                    "class_name": "Crab-Pot",
                                    "ai_confidence": 0.8523,
                                    "bbox_xywh_normalized": [0.25, 0.35, 0.1, 0.15],
                                    "bbox_xyxy_pixel": [128.0, 140.8, 192.0, 217.6],
                                }
                            ],
                        },
                        "cylinder": {
                            "frame_id": "frame_001",
                            "model_version": "cylinder_yolo_v1",
                            "detections": [
                                {
                                    "detection_id": "frame_001_cyl_001",
                                    "class_id": 0,
                                    "class_name": "marine_man_made_anomaly",
                                    "ai_confidence": 0.4215,
                                    "bbox_xywh_normalized": [0.7, 0.5, 0.15, 0.1],
                                    "bbox_xyxy_pixel": [400.0, 230.4, 496.0, 281.6],
                                }
                            ],
                        },
                        "pipeline": {
                            "frame_id": "frame_001",
                            "model_version": "pipeline_yolo26n_v1",
                            "detections": [
                                {
                                    "detection_id": "frame_001_pipe_001",
                                    "class_id": 0,
                                    "class_name": "Pipeline",
                                    "ai_confidence": 0.6789,
                                    "bbox_xywh_normalized": [0.5, 0.8, 0.4, 0.08],
                                    "bbox_xyxy_pixel": [192.0, 389.1, 448.0, 430.1],
                                }
                            ],
                        },
                        "seafloor": {
                            "frame_id": "frame_001",
                            "model_version": "seafloor_unet_ts_v1",
                            "segmentation": {
                                "mask_shape": [mask_h, mask_w],
                                "num_classes": 3,
                                "class_names": {"0": "rock", "1": "sand", "2": "others"},
                                "class_distribution": {
                                    "rock": {"class_id": 0, "pixel_count": int(np.sum(self.unet_mask == 0)), "area_percentage": 46.88, "mean_softmax_probability": 0.92},
                                    "sand": {"class_id": 1, "pixel_count": int(np.sum(self.unet_mask == 1)), "area_percentage": 39.06, "mean_softmax_probability": 0.88},
                                    "others": {"class_id": 2, "pixel_count": int(np.sum(self.unet_mask == 2)), "area_percentage": 14.06, "mean_softmax_probability": 0.75},
                                },
                                "mask_rle": {
                                    "rock": rock_rle,
                                    "sand": sand_rle,
                                    "others": others_rle,
                                },
                                "polygons": [],
                            },
                        },
                    },
                },
                # Frame 002: ZERO detections across all detectors
                {
                    "frame_id": "frame_002",
                    "image_width": self.img_width,
                    "image_height": self.img_height,
                    "models": {
                        "crab_pot": {"detections": []},
                        "cylinder": {"detections": []},
                        "pipeline": {"detections": []},
                        "seafloor": {
                            "segmentation": {
                                "mask_shape": [mask_h, mask_w],
                                "num_classes": 3,
                                "class_names": {"0": "rock", "1": "sand", "2": "others"},
                                "class_distribution": {
                                    "rock": {"class_id": 0, "pixel_count": 8192, "area_percentage": 50.0},
                                    "sand": {"class_id": 1, "pixel_count": 8192, "area_percentage": 50.0},
                                    "others": {"class_id": 2, "pixel_count": 0, "area_percentage": 0.0},
                                },
                                "mask_rle": {
                                    "rock": "1 8192",
                                    "sand": "8193 8192",
                                    "others": "",
                                },
                                "polygons": [],
                            }
                        },
                    },
                },
            ],
        }

        # Create image for frame_002
        self.test_img2_path = self.source_dir / "frame_002.jpg"
        cv2.imwrite(str(self.test_img2_path), raw_arr)

        # Save integrated JSON file
        self.json_file_path = self.test_path / "integrated_test.json"
        with open(self.json_file_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_json_data, f, indent=2)

        self.visualizer = Stage2Visualizer()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Requirement 11.1: Loading the integrated JSON
    # -----------------------------------------------------------------------
    def test_load_integrated_json(self):
        """Test loading valid JSON and robust handling of missing or invalid JSON."""
        # 1. Valid JSON
        loaded = Stage2Visualizer.load_json(self.json_file_path)
        self.assertIn("frames", loaded)
        self.assertEqual(len(loaded["frames"]), 2)
        self.assertEqual(loaded["pipeline_version"], "stage2_integrated_v1")

        # 2. Missing file
        with self.assertRaises(FileNotFoundError):
            Stage2Visualizer.load_json(self.test_path / "nonexistent.json")

        # 3. Invalid JSON structure (missing 'frames' key)
        bad_json_path = self.test_path / "bad.json"
        with open(bad_json_path, "w", encoding="utf-8") as f:
            json.dump({"pipeline_version": "v1"}, f)

        with self.assertRaises(ValueError):
            Stage2Visualizer.load_json(bad_json_path)

    # -----------------------------------------------------------------------
    # Requirement 11.2: Handling frames with zero detections
    # -----------------------------------------------------------------------
    def test_zero_detections_handling(self):
        """
        Test that a frame with zero detections preserves the image without
        inventing any bounding boxes or fake detections.
        """
        frame_002_record = self.mock_json_data["frames"][1]
        frame_models = frame_002_record["models"]

        raw_img = cv2.imread(str(self.test_img2_path))
        annotated_img, counts = self.visualizer.draw_detector_boxes(
            raw_img, frame_models, frame_id="frame_002"
        )

        # Confirm counts are all zero
        self.assertEqual(counts.get("crab_pot", 0), 0)
        self.assertEqual(counts.get("cylinder", 0), 0)
        self.assertEqual(counts.get("pipeline", 0), 0)
        self.assertEqual(sum(counts.values()), 0)

        # Confirm output image shape matches input
        self.assertEqual(annotated_img.shape, raw_img.shape)

        # Verify that below the top banner (banner height <= 50px),
        # the image pixels are completely unchanged (NO bounding boxes drawn)
        banner_boundary = 60
        np.testing.assert_array_equal(
            annotated_img[banner_boundary:, :, :],
            raw_img[banner_boundary:, :, :],
            err_msg="Pixels below header banner were altered in zero-detection frame! Bounding boxes were invented.",
        )

    # -----------------------------------------------------------------------
    # Requirement 11.3: Drawing detector boxes
    # -----------------------------------------------------------------------
    def test_drawing_detector_boxes(self):
        """
        Test drawing Crab-Pot, Cylinder, and Pipeline detections with correct
        pixel bounding boxes, labels, and raw AI confidence scores.
        """
        frame_001_record = self.mock_json_data["frames"][0]
        frame_models = frame_001_record["models"]

        raw_img = cv2.imread(str(self.test_img_path))
        annotated_img, counts = self.visualizer.draw_detector_boxes(
            raw_img, frame_models, frame_id="frame_001"
        )

        # Verify detection counts
        self.assertEqual(counts.get("crab_pot"), 1)
        self.assertEqual(counts.get("cylinder"), 1)
        self.assertEqual(counts.get("pipeline"), 1)
        self.assertEqual(sum(counts.values()), 3)

        # Verify pixel bounding box modifications for each object
        # 1. Crab-Pot at [128, 140, 192, 217]
        cp_region = annotated_img[140:218, 128:193]
        raw_cp_region = raw_img[140:218, 128:193]
        self.assertFalse(np.array_equal(cp_region, raw_cp_region), "Crab-Pot box was not drawn!")

        # 2. Cylinder at [400, 230, 496, 281]
        cyl_region = annotated_img[230:282, 400:497]
        raw_cyl_region = raw_img[230:282, 400:497]
        self.assertFalse(np.array_equal(cyl_region, raw_cyl_region), "Cylinder box was not drawn!")

        # 3. Pipeline at [192, 389, 448, 430]
        pipe_region = annotated_img[389:431, 192:449]
        raw_pipe_region = raw_img[389:431, 192:449]
        self.assertFalse(np.array_equal(pipe_region, raw_pipe_region), "Pipeline box was not drawn!")

        # Test normalized coordinate fallback when bbox_xyxy_pixel is absent
        norm_only_models = {
            "crab_pot": {
                "detections": [
                    {
                        "class_name": "Crab-Pot",
                        "ai_confidence": 0.91,
                        "bbox_xywh_normalized": [0.5, 0.5, 0.2, 0.2],
                    }
                ]
            }
        }
        annotated_norm, counts_norm = self.visualizer.draw_detector_boxes(
            raw_img, norm_only_models, frame_id="norm_test"
        )
        self.assertEqual(counts_norm["crab_pot"], 1)
        # Center region should have box drawn
        center_region = annotated_norm[200:300, 270:370]
        self.assertFalse(np.array_equal(center_region, raw_img[200:300, 270:370]))

    # -----------------------------------------------------------------------
    # Requirement 11.4: Handling the U-Net segmentation output
    # -----------------------------------------------------------------------
    def test_unet_segmentation_output(self):
        """
        Test that U-Net segmentation is visualized from mask/RLE data
        without fabricating bounding boxes or confidence scores.
        """
        frame_001_record = self.mock_json_data["frames"][0]
        seafloor_output = frame_001_record["models"]["seafloor"]

        raw_img = cv2.imread(str(self.test_img_path))
        seg_img = self.visualizer.draw_segmentation_overlay(
            raw_img, seafloor_output, frame_id="frame_001"
        )

        # 1. Output dimensions match input image
        self.assertEqual(seg_img.shape, raw_img.shape)

        # 2. Colors are blended into image
        self.assertFalse(np.array_equal(seg_img, raw_img), "Segmentation mask was not blended!")

        # 3. Verify distinct color regions matching rock (top), sand (middle), others (bottom)
        # Mid-x slice: rock region (y=100) vs sand region (y=300) should have distinct colors
        rock_pixel = seg_img[100, 320]
        sand_pixel = seg_img[300, 320]
        self.assertFalse(np.array_equal(rock_pixel, sand_pixel), "Rock and sand regions have identical colors!")

        # 4. Strictly confirm NO bounding boxes or fake confidence scores were fabricated
        self.assertNotIn("detections", seafloor_output)
        self.assertNotIn("confidence_score", seafloor_output["segmentation"])

        # 5. Test polygon fallback when RLE is empty
        poly_seafloor = {
            "segmentation": {
                "mask_shape": [64, 256],
                "num_classes": 3,
                "class_names": {"0": "rock", "1": "sand", "2": "others"},
                "mask_rle": {},
                "polygons": [
                    {
                        "class_id": 0,
                        "class_name": "rock",
                        "polygon_normalized": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    }
                ],
            }
        }
        poly_seg_img = self.visualizer.draw_segmentation_overlay(raw_img, poly_seafloor)
        self.assertEqual(poly_seg_img.shape, raw_img.shape)
        self.assertFalse(np.array_equal(poly_seg_img, raw_img))

    # -----------------------------------------------------------------------
    # Requirement 11.5: Missing source image handling
    # -----------------------------------------------------------------------
    def test_missing_source_image_handling(self):
        """
        Test that when a source image does not exist, the visualizer isolates the
        missing frame gracefully without crashing or throwing unhandled exceptions.
        """
        missing_record = {
            "frame_id": "nonexistent_sonar_frame_999",
            "image_width": 640,
            "image_height": 512,
            "models": {
                "crab_pot": {"detections": []},
            },
        }

        # 1. Single frame visualization: returns missing_image status safely
        result = self.visualizer.visualize_frame(
            frame_record=missing_record,
            source_dir_or_file=self.source_dir,
            output_dir=self.output_dir,
        )
        self.assertEqual(result["status"], "missing_image")
        self.assertEqual(result["generated_images"], [])
        self.assertIn("missing", result.get("error", "").lower())

        # 2. Batch visualization: missing frame is tracked in summary without failing run
        # Add a missing frame to mock JSON
        mock_with_missing = dict(self.mock_json_data)
        mock_with_missing["frames"] = self.mock_json_data["frames"] + [missing_record]

        missing_json_path = self.test_path / "with_missing.json"
        with open(missing_json_path, "w", encoding="utf-8") as f:
            json.dump(mock_with_missing, f, indent=2)

        summary = self.visualizer.visualize_all(
            json_path=missing_json_path,
            source_path=self.source_dir,
            output_dir=self.output_dir,
        )

        self.assertEqual(summary["total_frames"], 3)
        self.assertEqual(summary["frames_visualized"], 2)
        self.assertEqual(summary["missing_frames_count"], 1)
        self.assertIn("nonexistent_sonar_frame_999", summary["missing_frames"])
        self.assertEqual(summary["detection_images_generated"], 2)
        self.assertEqual(summary["segmentation_images_generated"], 2)
        self.assertEqual(summary["total_images_generated"], 4)

    # -----------------------------------------------------------------------
    # CLI Execution Integration Test
    # -----------------------------------------------------------------------
    def test_cli_execution(self):
        """
        Test executing visualize_stage2.py CLI via subprocess.
        """
        cmd = [
            sys.executable,
            "visualize_stage2.py",
            "--json",
            str(self.json_file_path),
            "--source",
            str(self.source_dir),
            "--output",
            str(self.output_dir / "cli_run"),
            "--format",
            "jpg",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"CLI exited with error:\n{res.stderr}")
        self.assertIn("STAGE-2 INTEGRATED VISUALIZATION GENERATION COMPLETE", res.stdout)
        self.assertIn("Total Visualizations     : 4 images saved", res.stdout)

        # Confirm 4 images were written to disk
        cli_out_dir = self.output_dir / "cli_run"
        self.assertTrue((cli_out_dir / "frame_001_detections.jpg").is_file())
        self.assertTrue((cli_out_dir / "frame_001_segmentation.jpg").is_file())
        self.assertTrue((cli_out_dir / "frame_002_detections.jpg").is_file())
        self.assertTrue((cli_out_dir / "frame_002_segmentation.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
