"""Camera calibration from chessboard images, with an on-disk cache."""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

CHESSBOARD_SIZE = (9, 6)  # inner corners per row, per column
CACHE_NAME = "calibration.npz"


def calibrate(image_dir: str, pattern: str = "calibration*.jpg") -> tuple[np.ndarray, np.ndarray]:
    """Compute the camera matrix and distortion coefficients from 9x6 chessboard photos."""
    images = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not images:
        raise FileNotFoundError(f"No calibration images matching {pattern!r} in {image_dir!r}")

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []  # 3D points in world space, 2D points in image plane
    image_size = None
    for fname in images:
        gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

    if not objpoints:
        raise RuntimeError("Chessboard corners were not found in any calibration image")

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return mtx, dist


def load_or_calibrate(image_dir: str, cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load cached calibration if present, otherwise compute it (and cache it)."""
    cache_path = os.path.join(image_dir, CACHE_NAME)
    if cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["mtx"], data["dist"]

    mtx, dist = calibrate(image_dir)
    if cache:
        try:
            np.savez(cache_path, mtx=mtx, dist=dist)
        except OSError:
            pass  # read-only checkout; just recompute next time
    return mtx, dist


def undistort(img: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return cv2.undistort(img, mtx, dist, None, mtx)
