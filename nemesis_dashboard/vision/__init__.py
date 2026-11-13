"""Pluggable vision backends for the Nemesis dashboard.

Use :func:`get_detector` to construct a detector by name. New vision methods
can be added by implementing :class:`~vision.base.Detector` and registering
them in :data:`AVAILABLE_DETECTORS`.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import Detection, Detector

# Registry of available vision methods. Keys are the values accepted by the
# ``--vision`` flag in main.py. YOLO is the only one implemented today.
AVAILABLE_DETECTORS: Dict[str, Callable[..., Detector]] = {}


def _register_yolo() -> None:
    # Imported lazily so that simply importing this package does not pull in
    # ultralytics/torch unless a YOLO detector is actually requested.
    from .yolo import YoloDetector

    AVAILABLE_DETECTORS["yolo"] = YoloDetector


_register_yolo()

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
    "get_detector",
    "AVAILABLE_DETECTORS",
    "VISION_METHODS",
]
