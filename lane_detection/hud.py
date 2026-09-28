"""Heads-up display drawn on top of the output frame: road status panel and a mini map."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from .lanes import WARP_SIZE, Line, RoadStatus

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
FONT = cv2.FONT_HERSHEY_DUPLEX

WHITE = (255, 255, 255)
MUTED = (200, 200, 200)
ACCENT = (80, 220, 120)  # green
WARN = (60, 170, 255)  # amber


@lru_cache(maxsize=1)
def _car_icon() -> Optional[np.ndarray]:
    return cv2.imread(os.path.join(ASSETS_DIR, "car_icon.png"), cv2.IMREAD_UNCHANGED)


def _translucent_box(img, x0, y0, x1, y1, alpha=0.55, color=(20, 20, 20)):
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def _paste_rgba(dst, src, x, y):
    """Alpha-blend a BGRA image onto a BGR image at (x, y), clipping at the borders."""
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    patch = src[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255
    dst[y0:y1, x0:x1] = (patch[:, :, :3] * alpha + dst[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)


def _guidance(status: RoadStatus) -> tuple[str, tuple[int, int, int]]:
    if status.direction == "Left Curve":
        return "<< Steer left", WARN
    if status.direction == "Right Curve":
        return "Steer right >>", WARN
    if status.direction == "Straight":
        return "Keep straight", ACCENT
    return "Hold lane", MUTED


def draw_status_panel(img: np.ndarray, status: Optional[RoadStatus]) -> np.ndarray:
    """Top-left panel with road direction, radius of curvature and lane offset."""
    scale = img.shape[1] / 1280
    s = lambda v: int(round(v * scale))  # noqa: E731

    _translucent_box(img, s(20), s(20), s(400), s(190))
    cv2.putText(img, "ROAD STATUS", (s(38), s(56)), FONT, 0.75 * scale, MUTED, max(1, s(1)), cv2.LINE_AA)

    if status is None:
        rows = [("Lane", "not detected")]
        guidance, color = "Searching...", MUTED
    else:
        curvature = "straight" if status.curvature_m is None else f"{status.curvature_m:,.0f} m"
        side = status.offset_side.lower()
        offset = "centered" if side == "center" else f"{status.offset_pct:.1f}% {side}"
        rows = [("Road", status.direction), ("Radius", curvature), ("Offset", offset)]
        guidance, color = _guidance(status)

    for i, (label, value) in enumerate(rows):
        y = s(92 + i * 28)
        cv2.putText(img, label, (s(38), y), FONT, 0.6 * scale, MUTED, max(1, s(1)), cv2.LINE_AA)
        cv2.putText(img, value, (s(140), y), FONT, 0.6 * scale, WHITE, max(1, s(1)), cv2.LINE_AA)

    cv2.putText(img, guidance, (s(38), s(176)), FONT, 0.8 * scale, color, max(1, s(2)), cv2.LINE_AA)
    return img


def render_road_map(left_line: Line, right_line: Line, size: int = 200) -> np.ndarray:
    """Bird's-eye mini map: the lane re-centred in the frame and the car at its offset."""
    canvas = np.full((WARP_SIZE, WARP_SIZE, 3), 30, np.uint8)
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    shift = WARP_SIZE / 2 - (left_line.startx + right_line.startx) / 2
    margin = left_line.window_margin / 4

    right_x = right_line.allx + shift
    left_x = right_x - lane_width  # draw both edges from the right line for a clean parallel lane

    def band(xs_l, xs_r):
        return np.int32([np.vstack([np.column_stack([xs_l, ploty]), np.flipud(np.column_stack([xs_r, ploty]))])])

    cv2.fillPoly(canvas, band(left_x + margin, right_x - margin), (70, 110, 60))
    cv2.fillPoly(canvas, band(left_x - margin, left_x + margin), WHITE)
    cv2.fillPoly(canvas, band(right_x - margin, right_x + margin), WHITE)

    icon = _car_icon()
    if icon is not None and icon.shape[2] == 4:
        car_w = int(np.clip(abs(lane_width) * 0.45, 60, 160))
        car_h = int(car_w * icon.shape[0] / icon.shape[1])
        car = cv2.resize(icon, (car_w, car_h), interpolation=cv2.INTER_AREA)
        car_center = WARP_SIZE / 2 + shift  # the camera (car) is at the image centre, moved by the same shift
        _paste_rgba(canvas, car, int(car_center - car_w / 2), WARP_SIZE - car_h - 20)

    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def draw_road_map(img: np.ndarray, left_line: Line, right_line: Line) -> np.ndarray:
    size = int(200 * img.shape[1] / 1280)
    pad = int(20 * img.shape[1] / 1280)
    road_map = render_road_map(left_line, right_line, size)
    x0, y0 = img.shape[1] - size - pad, pad
    img[y0:y0 + size, x0:x0 + size] = road_map
    cv2.rectangle(img, (x0, y0), (x0 + size - 1, y0 + size - 1), MUTED, 1, cv2.LINE_AA)
    return img
