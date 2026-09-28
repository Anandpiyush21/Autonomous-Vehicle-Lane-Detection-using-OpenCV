"""Regenerate the images and GIFs used in README.md.

    python scripts/make_readme_assets.py

Requires Pillow in addition to the normal requirements (for GIF encoding).
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lane_detection import LanePipeline, debug_mosaic  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images")
DATA = os.path.join(ROOT, "data")

GALLERY = {
    "straight": "test_images/straight_lines1.jpg",
    "right_curve": "test_images/test5.jpg",
    "left_curve": "test_images/test2.jpg",
    "shadows": "test_images/test4.jpg",
    "bridge": "test_images/highway.jpeg",
    "city_traffic": "test_images/vehicle.jpg",
    "zebra_crossing": "test_images/Zebra.jpeg",
    "unmarked_road": "test_images/sand.jpeg",
    "fog": "test_images/fog2.jpg",
    "pedestrian": "test_images/person.jpg",
}


def save_jpg(name: str, img: np.ndarray, width: int) -> None:
    h = int(img.shape[0] * width / img.shape[1])
    cv2.imwrite(os.path.join(OUT, name), cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def make_gif(video: str, name: str, width: int = 480, step: int = 3, start: int = 0, max_frames: int = 70) -> None:
    cap = cv2.VideoCapture(os.path.join(DATA, video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    pipeline = LanePipeline()
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipeline.process(frame).output  # every frame, so the temporal smoothing behaves as in real use
        if i >= start and (i - start) % step == 0:
            h = int(out.shape[0] * width / out.shape[1])
            rgb = cv2.cvtColor(cv2.resize(out, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        i += 1
    cap.release()
    path = os.path.join(OUT, name)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * step / fps), loop=0,
                   optimize=True)
    print(f"{name}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")


def main() -> None:
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    pipeline = LanePipeline()

    result = pipeline.process(cv2.imread(os.path.join(DATA, "test_images/test2.jpg")))
    save_jpg("pipeline_stages.jpg", debug_mosaic(result), 1440)

    for name, path in GALLERY.items():
        pipeline.reset()
        save_jpg(f"results/{name}.jpg", pipeline.process(cv2.imread(os.path.join(DATA, path))).output, 640)
    print("Saved stills")

    make_gif("videos/file2.mp4", "demo.gif")
    make_gif("videos/file1.mp4", "demo_highway.gif", start=60)


if __name__ == "__main__":
    main()
