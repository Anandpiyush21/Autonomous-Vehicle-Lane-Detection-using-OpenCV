"""Command-line interface.

    python -m lane_detection image data/test_images/test1.jpg
    python -m lane_detection video data/videos/file7.mp4 --output out.mp4 --no-show
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from .pipeline import LanePipeline, debug_mosaic

WINDOW = "Lane Detection"


def _show(img, wait: int) -> int:
    try:
        cv2.imshow(WINDOW, img)
        return cv2.waitKey(wait) & 0xFF
    except cv2.error:
        print("No display available (headless OpenCV?). Use --output and --no-show.", file=sys.stderr)
        raise SystemExit(1) from None


def run_image(args) -> int:
    img = cv2.imread(args.input)
    if img is None:
        print(f"Could not read image: {args.input}", file=sys.stderr)
        return 1

    result = LanePipeline(draw_hud=not args.no_hud).process(img)
    frame = debug_mosaic(result) if args.debug else result.output
    if result.status is None:
        print("No lane lines detected.")
    else:
        s = result.status
        radius = "straight" if s.curvature_m is None else f"{s.curvature_m:.0f} m"
        print(f"Road: {s.direction} | radius: {radius} | offset: {s.offset_pct:.1f}% {s.offset_side.lower()}")

    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"Saved {args.output}")
    if not args.no_show:
        _show(frame, 0)
        cv2.destroyAllWindows()
    return 0


def run_video(args) -> int:
    source = int(args.input) if args.input.isdigit() else args.input  # "0" -> webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video: {args.input}", file=sys.stderr)
        return 1

    pipeline = LanePipeline(draw_hud=not args.no_hud)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    writer = None
    frames, started = 0, time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame)
            out = debug_mosaic(result) if args.debug else result.output
            frames += 1

            if args.output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (out.shape[1], out.shape[0]))
                writer.write(out)
            if not args.no_show:
                key = _show(out, 1)
                if key == ord("q"):
                    break
                if key in (ord("p"), ord(" ")):  # pause until any key
                    cv2.waitKey(0)
            elif frames % 25 == 0:
                progress = f"{frames}/{total}" if total else str(frames)
                print(f"\rProcessed {progress} frames", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"\nProcessed {frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} FPS)")
    if args.output:
        print(f"Saved {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lane_detection", description="Detect road lanes in images and videos.")
    sub = parser.add_subparsers(dest="mode", required=True)

    for mode, help_text in (("image", "process a single image"), ("video", "process a video file or webcam index")):
        p = sub.add_parser(mode, help=help_text)
        input_help = "path to the input file" + (" (or a webcam index like 0)" if mode == "video" else "")
        p.add_argument("input", help=input_help)
        p.add_argument("-o", "--output", help="save the annotated result to this path")
        p.add_argument("--no-show", action="store_true", help="don't open a preview window")
        p.add_argument("--no-hud", action="store_true", help="draw only the lane, without the status panel and map")
        p.add_argument("--debug", action="store_true", help="show every pipeline stage in a grid")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    return run_image(args) if args.mode == "image" else run_video(args)
