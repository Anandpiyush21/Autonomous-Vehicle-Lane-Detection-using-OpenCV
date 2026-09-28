"""Lane detection for autonomous vehicles with classical computer vision (OpenCV)."""

from .lanes import RoadStatus
from .pipeline import FrameResult, LanePipeline, Thresholds, debug_mosaic

__all__ = ["LanePipeline", "FrameResult", "RoadStatus", "Thresholds", "debug_mosaic"]
__version__ = "2.0.0"
