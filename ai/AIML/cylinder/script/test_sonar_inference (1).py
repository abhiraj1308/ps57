import numpy as np

from sonar_inference import safe_crop, sonar_features


def test_safe_crop():
    image = np.zeros((100, 200), dtype=np.uint8)
    crop = safe_crop(image, [20, 30, 60, 70], pad=10)
    assert crop.shape == (60, 60)


def test_safe_crop_clamps():
    image = np.zeros((100, 100), dtype=np.uint8)
    crop = safe_crop(image, [-20, -20, 30, 30], pad=10)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_features():
    image = np.full((20, 30), 100, dtype=np.uint8)
    features = sonar_features(image)

    assert features["width"] == 30
    assert features["height"] == 20
    assert features["mean"] == 100.0
    assert features["std"] == 0.0
