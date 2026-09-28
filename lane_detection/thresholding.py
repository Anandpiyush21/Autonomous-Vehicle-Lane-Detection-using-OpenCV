"""Gradient (Sobel) and colour (HLS) thresholding to isolate lane-marking pixels.

All functions return uint8 binary masks with values 0 / 255.
"""

from __future__ import annotations

import cv2
import numpy as np


def _scale_to_uint8(values: np.ndarray) -> np.ndarray:
    peak = np.max(values)
    if peak == 0:
        return np.zeros(values.shape, np.uint8)
    return np.uint8(255 * values / peak)


def _in_range(values: np.ndarray, thresh: tuple[float, float]) -> np.ndarray:
    binary = np.zeros(values.shape, np.uint8)
    binary[(values >= thresh[0]) & (values <= thresh[1])] = 255
    return binary


def sobel_xy(img: np.ndarray, orient: str = "x", thresh=(20, 100)) -> np.ndarray:
    """Threshold the absolute Sobel derivative along x (near-vertical edges) or y."""
    dx, dy = (1, 0) if orient == "x" else (0, 1)
    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, dx, dy))
    return _in_range(_scale_to_uint8(abs_sobel), thresh)


def mag_thresh(img: np.ndarray, sobel_kernel: int = 3, thresh=(0, 255)) -> np.ndarray:
    """Threshold the gradient magnitude."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return _in_range(_scale_to_uint8(np.hypot(sobelx, sobely)), thresh)


def dir_thresh(img: np.ndarray, sobel_kernel: int = 3, thresh=(0.7, 1.3)) -> np.ndarray:
    """Threshold the gradient direction (radians, 0 = horizontal gradient)."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return _in_range(np.arctan2(np.absolute(sobely), np.absolute(sobelx)), thresh)


def ch_thresh(ch: np.ndarray, thresh=(80, 255)) -> np.ndarray:
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255
    return binary


def gradient_combine(roi: np.ndarray, th_x, th_y, th_mag, th_dir) -> np.ndarray:
    """Combine Sobel x/y, magnitude and direction masks computed on the red channel."""
    red = roi[:, :, 2]
    sobelx = sobel_xy(red, "x", th_x)
    sobely = sobel_xy(red, "y", th_y)
    mag_img = mag_thresh(red, 3, th_mag)
    dir_img = dir_thresh(red, 15, th_dir)

    combined = np.zeros_like(dir_img)
    combined[((sobelx > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobelx > 1) & (sobely > 1))] = 255
    return combined


def hls_combine(roi: np.ndarray, th_h, th_l, th_s) -> np.ndarray:
    """Colour mask in HLS space that handles lane lines both in and out of shadow."""
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    h_img = ch_thresh(hls[:, :, 0], th_h)
    l_img = ch_thresh(hls[:, :, 1], th_l)
    s_img = ch_thresh(hls[:, :, 2], th_s)

    combined = np.zeros_like(s_img)
    combined[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255
    return combined


def comb_result(grad: np.ndarray, hls: np.ndarray) -> np.ndarray:
    """Merge both masks, keeping them distinguishable (gradient=100, colour=255)."""
    result = np.zeros_like(hls)
    result[grad > 1] = 100
    result[hls > 1] = 255
    return result
