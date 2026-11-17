"""Pluggable vision backends for the Nemesis dashboard.

Use :func:`get_detector` to construct a detector by name. New vision methods
can be added by implementing :class:`~vision.base.Detector` and registering
them in :data:`AVAILABLE_DETECTORS`.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .base import Detection, Detector


def _yolo(**kwargs) -> Detector:
    # Imported here so that importing this package does not pull in
    # ultralytics/torch unless a YOLO detector is actually requested.
    from .yolo import YoloDetector

    return YoloDetector(**kwargs)


class NoDetector(Detector):
    """Finds nothing. For running the dashboard without a model (tests, page work)."""

    def __init__(self, **_options):
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


# Registry of available vision methods. Keys are the values accepted by the
# ``--vision`` flag in main.py.
AVAILABLE_DETECTORS: Dict[str, Callable[..., Detector]] = {
    "yolo": _yolo,
    "none": NoDetector,
}

VISION_METHODS = tuple(AVAILABLE_DETECTORS.keys())


def get_detector(name: str, **kwargs) -> Detector:
    """Construct a detector by name.

    Args:
        name: vision method, e.g. ``"yolo"``.
        **kwargs: forwarded to the detector constructor.

    Raises:
        ValueError: if ``name`` is not a known vision method.
    """
    try:
        factory = AVAILABLE_DETECTORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown vision method '{name}'. Available: {', '.join(VISION_METHODS)}"
        ) from None
    return factory(**kwargs)


__all__ = [
    "Detection",
    "Detector",
    "NoDetector",
    "get_detector",
    "AVAILABLE_DETECTORS",
    "VISION_METHODS",
]
