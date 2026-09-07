"""Pure geometry helpers with no external CV dependency (fully unit-testable)."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1


def sort_detections_clockwise(detections: Sequence) -> list:
    """Sort AprilTag detections (objects with a `.center` (x, y) attribute) clockwise from the centroid."""
    import numpy as np

    centers = np.array([d.center for d in detections])
    cx, cy = np.mean(centers, axis=0)
    angles = np.arctan2(centers[:, 1] - cy, centers[:, 0] - cx)
    sorted_indices = np.argsort(angles)
    return [detections[i] for i in sorted_indices]
