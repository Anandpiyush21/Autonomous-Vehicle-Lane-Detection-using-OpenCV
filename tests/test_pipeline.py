import os

import cv2
import numpy as np
import pytest

from lane_detection import LanePipeline, debug_mosaic
from lane_detection.cli import main
from lane_detection.thresholding import comb_result, gradient_combine, hls_combine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "data", "test_images")


@pytest.fixture(scope="module")
def pipeline():
    return LanePipeline()


def load(name):
    img = cv2.imread(os.path.join(IMAGES, name))
    assert img is not None, name
    return img


def test_straight_road(pipeline):
    pipeline.reset()
    result = pipeline.process(load("straight_lines1.jpg"))
    assert result.status.direction == "Straight"
    assert result.status.offset_pct < 10
    assert result.output.shape == (720, 1280, 3)


@pytest.mark.parametrize("name, direction", [("test5.jpg", "Right Curve"), ("test2.jpg", "Left Curve")])
def test_curves(pipeline, name, direction):
    pipeline.reset()
    assert pipeline.process(load(name)).status.direction == direction


def test_blank_frame_does_not_crash(pipeline):
    pipeline.reset()
    result = pipeline.process(np.zeros((720, 1280, 3), np.uint8))
    assert result.status is None
    assert result.output.shape == (720, 1280, 3)


def test_lost_lane_keeps_previous_fit(pipeline):
    pipeline.reset()
    pipeline.process(load("test1.jpg"))
    result = pipeline.process(np.zeros((720, 1280, 3), np.uint8))
    assert result.status is not None  # falls back on the last good fit


def test_any_input_size(pipeline):
    pipeline.reset()
    assert pipeline.process(load("Zebra.jpeg")).output.shape == (720, 1280, 3)


def test_thresholds_are_binary():
    roi = cv2.resize(load("test4.jpg"), (640, 360))[220:348]
    grad = gradient_combine(roi, (35, 100), (30, 255), (30, 255), (0.7, 1.3))
    hls = hls_combine(roi, (10, 100), (0, 60), (85, 255))
    assert set(np.unique(grad)) <= {0, 255}
    assert set(np.unique(comb_result(grad, hls))) <= {0, 100, 255}


def test_debug_mosaic(pipeline):
    pipeline.reset()
    assert debug_mosaic(pipeline.process(load("test3.jpg"))).shape == (720, 1920, 3)


def test_cli_image(tmp_path):
    out = tmp_path / "out.jpg"
    assert main(["image", os.path.join(IMAGES, "test1.jpg"), "--no-show", "-o", str(out)]) == 0
    assert cv2.imread(str(out)).shape == (720, 1280, 3)
