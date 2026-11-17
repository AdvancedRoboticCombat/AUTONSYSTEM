"""What the threads share: the latest camera frame and the numbers on the page."""

from __future__ import annotations

import json
import threading
from typing import Optional, Tuple

import numpy as np

from .config import WATCHDOG_TIMEOUT

MOTOR_ALERT_KEYS = (
    "weapon_motor", "weapon_esc",
    "arm_motor", "arm_esc",
    "leftWheel_motor", "leftWheel_esc",
    "rightWheel_motor", "rightWheel_esc",
)


class FrameStore:
    """The newest frame. Readers sleep until a newer one arrives instead of polling."""

    def __init__(self):
        self._changed = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._seq = 0

    def put(self, frame: np.ndarray, ts: float):
        with self._changed:
            self._frame = frame
            self._ts = ts
            self._seq += 1
            self._changed.notify_all()

    def get(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._changed:
            return self._frame, self._ts, self._seq

    def wait_newer(self, seq: int, timeout: float) -> Tuple[Optional[np.ndarray], float, int]:
        """The frame after number seq, as soon as it exists. (None, 0.0, seq) if
        none came within timeout seconds."""
        with self._changed:
            if self._seq == seq:
                self._changed.wait(timeout)
            if self._seq == seq or self._frame is None:
                return None, 0.0, seq
            return self._frame, self._ts, self._seq


class DashboardState:
    """The numbers the page shows. Any thread may update them; the web server
    sends them on when something changed."""

    def __init__(self, override: bool):
        self._lock = threading.Lock()
        self._dirty = True
        # Last time each robot component was heard from on ROS (0 = never)
        self._last_heard = {key: 0.0 for key in MOTOR_ALERT_KEYS}
        self._data = {
            "systemHealth": {
                "vision": {"fps": 0, "captureFps": 0, "latencyMs": 0, "nemConfidence": 0, "oppConfidence": 0, "track": "ok"},
                "path": {"validity": True},
            },
            "imu": {"rotation": 0, "track": 0, "linSpeed": 0},
            "telemetry": {
                "driveL": {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
                "driveR": {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
                "arm":    {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
                "weapon": {"escVolt": 0, "escCurr": 0, "escTemp": 0, "motorRPM": 0},
            },
            "controls": {"vision": [0, 0, 0], "path": [0, 0]},
            "machineCondition": {"motors": [], "wheels": []},
            "calculations": {
                "WheelSlip": {"odometrySpeed": 0, "absVelDiff": 0},
                "weapon": {"kineticEnergy": 0, "kineticETransfer": 0},
            },
            # True = that component has gone silent
            "motorAlerts": {key: False for key in MOTOR_ALERT_KEYS},
            # True = manual (driver) control, False = autonomous. Mirrors /nemesis/override.
            "override": override,
        }

    def update(self, *path: str, **values):
        """update("systemHealth", "vision", fps=60.0) sets fields of one section;
        update(override=True) sets top-level ones."""
        with self._lock:
            section = self._data
            for key in path:
                section = section[key]
            for key, value in values.items():
                if section.get(key) != value:
                    section[key] = value
                    self._dirty = True

    def heard(self, component: str, now: float):
        """A ROS message from that component arrived at Unix time now."""
        with self._lock:
            self._last_heard[component] = now

    def refresh_alerts(self, now: float):
        """Flag the components that have been silent for WATCHDOG_TIMEOUT seconds."""
        with self._lock:
            alerts = self._data["motorAlerts"]
            for key, last in self._last_heard.items():
                silent = (now - last) > WATCHDOG_TIMEOUT
                if alerts[key] != silent:
                    alerts[key] = silent
                    self._dirty = True

    def capture_fps(self) -> float:
        with self._lock:
            return self._data["systemHealth"]["vision"]["captureFps"]

    def to_json(self) -> str:
        """The whole state, as the message the page expects."""
        with self._lock:
            return json.dumps({"type": "state", "data": self._data})

    def changed_json(self) -> Optional[str]:
        """to_json(), or None when nothing changed since the last call of this."""
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
            return json.dumps({"type": "state", "data": self._data})
