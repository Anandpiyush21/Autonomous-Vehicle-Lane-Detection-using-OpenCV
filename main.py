"""Convenience entry point: ``python main.py video data/videos/file7.mp4``. See ``--help``."""

from lane_detection.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
