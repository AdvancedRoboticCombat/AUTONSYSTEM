"""YOLO detection backend.

Wraps an Ultralytics YOLO model behind the ``Detector`` interface. Supports
both segmentation models (returns mask polygons + contour-mean centers) and
detection-only models (returns bbox centers, no mask).
"""

from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from .base import Detection, Detector


class YoloDetector(Detector):
    """Per-frame YOLO detector.

    Args:
        model_path: path to the YOLO ``.pt`` checkpoint.
        mode: ``"seg"`` to use segmentation masks (contour-mean center +
            mask polygon) or ``"box"`` to use bounding boxes only.
        imgsz: inference image size passed to the model.
        seg_points: target number of points to downsample each mask polygon
            to before returning (keeps payloads small for the frontend).
        max_detections: optional cap on the number of detections returned
            (highest-confidence first). ``None`` returns all.
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "seg",
        imgsz: int = 640,
        seg_points: int = 60,
        max_detections: int | None = None,
    ):
        if mode not in ("seg", "box"):
            raise ValueError(f"Unknown YOLO mode '{mode}'; expected 'seg' or 'box'.")
        self.model = YOLO(model_path)
        names = self.model.names  # {class index: name}
        self.nemesis_class = next((i for i, name in names.items() if name == "nemesis"), None)
        self.mode = mode
        self.imgsz = imgsz
        self.seg_points = seg_points
        self.max_detections = max_detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, imgsz=self.imgsz, verbose=False)[0]

        if self.mode == "seg":
            detections = self._detect_seg(results)
        else:
            detections = self._detect_box(results)

        if self.max_detections is not None and len(detections) > self.max_detections:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            detections = detections[: self.max_detections]
        return detections

    def _detect_seg(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if results.masks is None or results.boxes is None or len(results.boxes) == 0:
            return detections

        for contour, box in zip(results.masks.xy, results.boxes):
            if len(contour) == 0:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = float(contour[:, 0].mean())
            cy = float(contour[:, 1].mean())
            conf = float(box.conf[0])
            step = max(1, len(contour) // self.seg_points)
            downsampled = contour[::step]
            seg = [[float(pt[0]), float(pt[1])] for pt in downsampled]
            class_id = int(box.cls[0]) if box.cls is not None else None
            detections.append(
                Detection(
                    center=(cx, cy),
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    mask=seg,
                    class_id=class_id,
                )
            )
        return detections

    def _detect_box(self, results) -> List[Detection]:
        detections: List[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            conf = float(box.conf[0])
            class_id = int(box.cls[0]) if box.cls is not None else None
            detections.append(
                Detection(
                    center=(cx, cy),
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    mask=None,
                    class_id=class_id,
                )
            )
        return detections
