"""Common interface for pluggable vision detectors.

Any detection backend (YOLO today, something else tomorrow) implements the
``Detector`` interface and returns a list of ``Detection`` objects from
``detect(frame)``.  This keeps ``main.py`` agnostic to the underlying vision
method so backends can be swapped via the ``--vision`` flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """A single detected object in pixel space.

    Attributes:
        center: (cx, cy) representative point of the object. For segmentation
            this is the mask-contour mean; for boxes it is the bbox center.
            This is the RAW center; arena/perspective corrections (e.g.
            ``utils.geometry.adjust_center``) are applied by the caller, not
            the detector.
        bbox: (x1, y1, x2, y2) bounding box in pixel space.
        confidence: detection confidence in [0, 1].
        mask: optional downsampled segmentation polygon as a list of
            [x, y] pixel points, or ``None`` for box-only detectors.
        class_id: optional model class index.
    """

    center: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    mask: Optional[List[List[float]]] = None
    class_id: Optional[int] = None


class Detector(ABC):
    """Abstract base class for a single-frame object detector."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one BGR frame and return detections."""
        raise NotImplementedError
