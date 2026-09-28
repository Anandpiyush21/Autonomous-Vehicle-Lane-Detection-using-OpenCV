"""End-to-end lane detection: undistort -> threshold -> warp -> search -> overlay."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import hud
from .calibration import load_or_calibrate, undistort
from .lanes import WARP_SIZE, LaneNotFound, Line, RoadStatus, draw_lane, find_LR_lines, road_info, warp_image
from .thresholding import comb_result, gradient_combine, hls_combine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CALIBRATION_DIR = os.path.join(REPO_ROOT, "data", "camera_cal")

FRAME_SIZE = (1280, 720)  # the camera the calibration images came from
WORK_SIZE = (640, 360)  # processing happens at half resolution
ROI_TOP, ROI_BOTTOM_MARGIN = 220, 12  # rows of the working image that contain the road


@dataclass
class Thresholds:
    sobel_x: tuple = (35, 100)
    sobel_y: tuple = (30, 255)
    magnitude: tuple = (30, 255)
    direction: tuple = (0.7, 1.3)
    hue: tuple = (10, 100)
    lightness: tuple = (0, 60)
    saturation: tuple = (85, 255)


@dataclass
class FrameResult:
    output: np.ndarray  # final annotated frame (FRAME_SIZE)
    status: Optional[RoadStatus]  # None when no lane was found
    stages: dict = field(default_factory=dict)  # intermediate images, keyed by stage name


def _perspective_points(rows: int, cols: int):
    src = np.float32([[110, rows], [cols / 2 - 24, 5], [cols / 2 + 24, 5], [cols - 110, rows]])
    dst = np.float32([(170, WARP_SIZE), (170, 0), (550, 0), (550, WARP_SIZE)])
    return src, dst


class LanePipeline:
    """Stateful lane detector. Feed it consecutive video frames, or call ``reset()`` between images."""

    def __init__(self, calibration_dir: str = DEFAULT_CALIBRATION_DIR, thresholds: Optional[Thresholds] = None,
                 draw_hud: bool = True):
        self.mtx, self.dist = load_or_calibrate(calibration_dir)
        self.th = thresholds or Thresholds()
        self.draw_hud = draw_hud
        self.reset()

    def reset(self) -> None:
        self.left, self.right = Line(), Line()
        self.status: Optional[RoadStatus] = None

    def process(self, frame: np.ndarray) -> FrameResult:
        if frame.shape[1::-1] != FRAME_SIZE:
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        undist = undistort(frame, self.mtx, self.dist)
        small = cv2.resize(undist, WORK_SIZE, interpolation=cv2.INTER_AREA)
        roi = small[ROI_TOP:WORK_SIZE[1] - ROI_BOTTOM_MARGIN]

        th = self.th
        gradient = gradient_combine(roi, th.sobel_x, th.sobel_y, th.magnitude, th.direction)
        color = hls_combine(roi, th.hue, th.lightness, th.saturation)
        binary = comb_result(gradient, color)

        src, dst = _perspective_points(*binary.shape[:2])
        warped, _, Minv = warp_image(binary, src, dst, (WARP_SIZE, WARP_SIZE))
        stages = {"undistorted": undist, "gradient": gradient, "color": color, "binary": binary, "warped": warped}

        try:
            stages["search"] = find_LR_lines(warped, self.left, self.right)
        except LaneNotFound:
            self.left.detected = self.right.detected = False
            if self.left.current_fit is None:  # nothing to fall back on
                self.status = None
                return FrameResult(self._annotate(undist.copy(), None), None, stages)
            stages["search"] = np.dstack([warped] * 3)  # reuse the last good fit this frame

        stages["lane_warped"], lane_warped = draw_lane(stages["search"], self.left, self.right)
        self.status = road_info(self.left, self.right, self.status)

        # Project the lane back onto the road and blend it with the full-resolution frame.
        lane_roi = cv2.warpPerspective(lane_warped, Minv, (roi.shape[1], roi.shape[0]))
        lane_small = np.zeros_like(small)
        lane_small[ROI_TOP:WORK_SIZE[1] - ROI_BOTTOM_MARGIN] = lane_roi
        lane_full = cv2.resize(lane_small, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
        output = cv2.addWeighted(undist, 1, lane_full, 0.4, 0)

        return FrameResult(self._annotate(output, self.status), self.status, stages)

    def _annotate(self, img: np.ndarray, status: Optional[RoadStatus]) -> np.ndarray:
        if not self.draw_hud:
            return img
        hud.draw_status_panel(img, status)
        if status is not None:
            hud.draw_road_map(img, self.left, self.right)
        return img


def debug_mosaic(result: FrameResult, tile: tuple = (640, 360)) -> np.ndarray:
    """2x3 grid of the pipeline stages with captions, handy for tuning thresholds."""
    def as_bgr(img):
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blank = np.zeros((tile[1], tile[0], 3), np.uint8)
    names = [("undistorted", "1. Undistorted"), ("gradient", "2. Sobel gradients"), ("color", "3. HLS colour"),
             ("warped", "4. Bird's-eye view"), ("lane_warped", "5. Sliding-window fit"), (None, "6. Result")]
    tiles = []
    for key, caption in names:
        img = result.output if key is None else result.stages.get(key)
        img = blank.copy() if img is None else cv2.resize(as_bgr(img), tile, interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (tile[0], 34), (20, 20, 20), -1)
        cv2.putText(img, caption, (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
