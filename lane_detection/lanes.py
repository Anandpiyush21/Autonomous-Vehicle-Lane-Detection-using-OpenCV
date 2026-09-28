"""Lane-line search on a bird's-eye binary image, polynomial fitting and road geometry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

WARP_SIZE = 720  # the bird's-eye image is WARP_SIZE x WARP_SIZE
NUM_WINDOWS = 9
MIN_PIXELS_TO_RECENTER = 50
MIN_PIXELS_TO_FIT = 3
HISTORY = 10  # frames averaged to smooth the fitted lines
MAX_WIDTH_STD = 80  # lane-width spread (px) above which we fall back to a blind search

LANE_WIDTH_M = 3.7  # standard lane width in India
YM_PER_PIX = 30 / WARP_SIZE


class LaneNotFound(Exception):
    """Raised when too few lane pixels are found to fit a line."""


class Line:
    """State for one lane line, carried across video frames."""

    def __init__(self, window_margin: int = 56):
        self.detected = False
        self.window_margin = window_margin
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY)
        self.current_fit: Optional[np.ndarray] = None
        self.radius_of_curvature: Optional[float] = None
        self.startx: Optional[float] = None  # x at the bottom of the warped image (nearest the car)
        self.endx: Optional[float] = None  # x at the top of the warped image
        self.allx: Optional[np.ndarray] = None
        self.ally: Optional[np.ndarray] = None

    def update(self, fit: np.ndarray, ploty: np.ndarray) -> None:
        """Add a new fit and smooth it with the recent history."""
        self.history.append(np.polyval(fit, ploty))
        avg_fit = np.polyfit(ploty, np.mean(self.history, axis=0), 2)
        self.current_fit = avg_fit
        self.allx, self.ally = np.polyval(avg_fit, ploty), ploty
        self.startx, self.endx = self.allx[-1], self.allx[0]


@dataclass
class RoadStatus:
    direction: str  # "Straight", "Left Curve", "Right Curve" or "Unknown"
    curvature_m: Optional[float]  # None on a straight road
    offset_side: str  # "Left", "Right" or "Center": where the car sits relative to lane centre
    offset_pct: float  # distance from lane centre as % of half the lane width


def warp_image(img, src, dst, size):
    """Perspective transform; also returns the forward and inverse matrices."""
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR), M, Minv


def _fit(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    if len(xs) < MIN_PIXELS_TO_FIT:
        raise LaneNotFound
    return np.polyfit(ys, xs, 2)


def _search_canvas(b_img: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*b_img.shape, 3), np.uint8)
    canvas[b_img > 0] = (255, 255, 255)
    return canvas


def blind_search(b_img, left_line: Line, right_line: Line) -> np.ndarray:
    """Histogram + sliding-window search, used on the first frame or after losing the lanes."""
    histogram = np.sum(b_img[b_img.shape[0] // 2:, :], axis=0)
    output = _search_canvas(b_img)

    midpoint = histogram.shape[0] // 2
    current_leftx = int(np.argmax(histogram[:midpoint]))
    current_rightx = int(np.argmax(histogram[midpoint:])) + midpoint

    window_height = b_img.shape[0] // NUM_WINDOWS
    nonzeroy, nonzerox = b_img.nonzero()
    margin = left_line.window_margin
    left_inds, right_inds = [], []

    for window in range(NUM_WINDOWS):
        y_low = b_img.shape[0] - (window + 1) * window_height
        y_high = b_img.shape[0] - window * window_height
        in_rows = (nonzeroy >= y_low) & (nonzeroy <= y_high)

        for cx, inds in ((current_leftx, left_inds), (current_rightx, right_inds)):
            cv2.rectangle(output, (cx - margin, y_low), (cx + margin, y_high), (0, 255, 0), 2)
            inds.append((in_rows & (nonzerox >= cx - margin) & (nonzerox <= cx + margin)).nonzero()[0])

        if len(left_inds[-1]) > MIN_PIXELS_TO_RECENTER:
            current_leftx = int(np.mean(nonzerox[left_inds[-1]]))
        if len(right_inds[-1]) > MIN_PIXELS_TO_RECENTER:
            current_rightx = int(np.mean(nonzerox[right_inds[-1]]))

    left_inds, right_inds = np.concatenate(left_inds), np.concatenate(right_inds)
    return _fit_and_update(b_img, output, nonzerox, nonzeroy, left_inds, right_inds, left_line, right_line)


def prev_window_refer(b_img, left_line: Line, right_line: Line) -> np.ndarray:
    """Search around the previous frame's fitted curves."""
    output = _search_canvas(b_img)
    nonzeroy, nonzerox = b_img.nonzero()
    margin = left_line.window_margin

    left_center = np.polyval(left_line.current_fit, nonzeroy)
    right_center = np.polyval(right_line.current_fit, nonzeroy)
    left_inds = (np.abs(nonzerox - left_center) <= margin).nonzero()[0]
    right_inds = (np.abs(nonzerox - right_center) <= margin).nonzero()[0]

    output = _fit_and_update(b_img, output, nonzerox, nonzeroy, left_inds, right_inds, left_line, right_line)

    # Lines that drift apart unevenly are probably wrong: search from scratch next frame.
    if np.std(right_line.allx - left_line.allx) > MAX_WIDTH_STD:
        left_line.detected = False
    return output


def _fit_and_update(b_img, output, nonzerox, nonzeroy, left_inds, right_inds, left_line, right_line):
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    output[lefty, leftx] = (255, 0, 0)
    output[righty, rightx] = (0, 0, 255)

    left_fit, right_fit = _fit(lefty, leftx), _fit(righty, rightx)

    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])
    left_line.update(left_fit, ploty)
    right_line.update(right_fit, ploty)
    left_line.detected = right_line.detected = True
    rad_of_curvature(left_line, right_line)
    return output


def find_LR_lines(binary_img, left_line: Line, right_line: Line) -> np.ndarray:
    """Locate both lane lines, reusing the previous frame's fit when we have one."""
    if left_line.detected and left_line.current_fit is not None:
        return prev_window_refer(binary_img, left_line, right_line)
    return blind_search(binary_img, left_line, right_line)


def rad_of_curvature(left_line: Line, right_line: Line) -> None:
    """Radius of curvature (metres) at the bottom of the image, for each line."""
    ploty = left_line.ally
    lane_px = max(abs(right_line.startx - left_line.startx), 1.0)
    xm_per_pix = LANE_WIDTH_M * (720 / 1280) / lane_px
    y_eval = np.max(ploty) * YM_PER_PIX

    for line in (left_line, right_line):
        a, b, _ = np.polyfit(ploty * YM_PER_PIX, line.allx[::-1] * xm_per_pix, 2)
        line.radius_of_curvature = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / max(abs(2 * a), 1e-9)


def road_info(left_line: Line, right_line: Line, previous: Optional[RoadStatus] = None) -> RoadStatus:
    """Classify the road as straight / left / right curve and measure the car's lane offset."""
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
    direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2

    if curvature > 1500 and abs(direction) < 100:
        road, curvature = "Straight", None
    elif curvature <= 1500 and direction < -50:
        road = "Left Curve"
    elif curvature <= 1050 and direction > 50:
        road = "Right Curve"
    elif previous is not None:
        road, curvature = previous.direction, previous.curvature_m
    else:
        road = "Unknown"

    lane_center = (right_line.startx + left_line.startx) / 2
    half_width = max((right_line.startx - left_line.startx) / 2, 1.0)
    car_center = WARP_SIZE / 2
    offset_pct = abs(lane_center - car_center) / half_width * 100
    if lane_center > car_center:
        side = "Left"
    elif lane_center < car_center:
        side = "Right"
    else:
        side = "Center"

    return RoadStatus(road, curvature, side, offset_pct)


def draw_lane(img, left_line: Line, right_line: Line, lane_color=(255, 0, 0), road_color=(0, 255, 0)):
    """Paint the lane lines and the drivable area between them (in warped space)."""
    window_img = np.zeros_like(img)
    half = left_line.window_margin / 5
    ploty = left_line.ally

    def band(xs_l, xs_r):
        left_edge = np.column_stack([xs_l, ploty])
        right_edge = np.flipud(np.column_stack([xs_r, ploty]))
        return np.int32([np.vstack([left_edge, right_edge])])

    cv2.fillPoly(window_img, band(left_line.allx - half, left_line.allx + half), lane_color)
    cv2.fillPoly(window_img, band(right_line.allx - half, right_line.allx + half), lane_color)
    cv2.fillPoly(window_img, band(left_line.allx + half, right_line.allx - half), road_color)

    return cv2.addWeighted(img, 1, window_img, 0.3, 0), window_img
