#!/usr/bin/env python3
"""Camera setup popup; Done starts the existing Nemesis dashboard.

    python camera_setup.py                  camera setup, then the dashboard
    python camera_setup.py --preview-only   camera test only: live preview, YOLO
                                            overlay, photos, video, calibration
"""

import ipaddress
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from fractions import Fraction
import webbrowser
import urllib.request

import cv2
import numpy as np
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")

from gi.repository import Gst, GstVideo
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import main as backend
from utils.camera_intrinsics import compute_intrinsics, DEFAULT_OUTPUT_NPZ


# key, label, default, initial slider min/max, increment, integer
CAMERA = [
    ("AcquisitionFrameRate", "Camera FPS", 70, 1, 240, 1, False),
    ("Gamma", "Gamma", 0.4, 0.1, 4, 0.05, False),
    ("BinningHorizontal", "Horizontal binning", 1, 1, 4, 1, True),
    ("BinningVertical", "Vertical binning", 1, 1, 4, 1, True),
    ("Width", "Width (pixels)", 1920, 64, 4096, 1, True),
    ("Height", "Height (pixels)", 1440, 64, 4096, 1, True),
    ("OffsetX", "X offset (pixels)", 256, 0, 4096, 1, True),
    ("OffsetY", "Y offset (pixels)", 304, 0, 4096, 1, True),
    (
        "DeviceLinkThroughputLimit",
        "Throughput limit (bytes/s)",
        312500000,
        1000000,
        1000000000,
        1000000,
        True,
    ),
    ("ExposureTime", "Exposure (microseconds)", 8000, 100, 100000, 100, False),
]

SOFTWARE = [
    ("VisionFPS", "YOLO FPS limit (0 = unlimited)", 0, 0, 240, 1, False),
    ("YOLOSize", "YOLO image size", 640, 32, 1280, 32, True),
    ("StreamFPS", "Browser stream FPS", 30, 1, 120, 1, False),
    ("JPEGQuality", "Browser JPEG quality", 40, 1, 100, 1, True),
]


# What the operator sees -> GStreamer videoflip method. Applied in the pipeline, so the
# preview, photos, recordings, calibration and the dashboard all see the same picture.
ORIENTATIONS = {
    "Normal": "none",
    "Rotate 180": "rotate-180",
    "Mirror left-right": "horizontal-flip",
    "Flip upside down": "vertical-flip",
    "Rotate 90 clockwise": "clockwise",
    "Rotate 90 counter-clockwise": "counterclockwise",
}


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def arv(address, *items):
    command = [
        "arv-tool-0.8",
        "--address",
        address,
        "--range-check=enable",
        "control",
        *items,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=10
    )
    output = result.stdout + "\n" + result.stderr

    if result.returncode or re.search(
        r"\berror\b|not found|no device|failed|could not|can't",
        output,
        re.I,
    ):
        raise RuntimeError(output.strip() or "Camera command failed.")

    return result.stdout


def read_features(address, keys):
    output = arv(address, *keys)
    values, ranges = {}, {}

    for line in output.splitlines():
        match = re.match(r"^\s*(\w+)\s*=\s*(\S+)", line)
        if not match or match[1] not in keys:
            continue

        key = match[1]
        values[key] = float(match[2])

        low = re.search(r"\bmin:\s*(\S+)", line)
        high = re.search(r"\bmax:\s*(\S+)", line)
        inc = re.search(r"\binc:\s*(\S+)", line)

        if low and high:
            ranges[key] = (
                float(low[1]),
                float(high[1]),
                float(inc[1]) if inc else None,
            )

    missing = set(keys) - values.keys()
    if missing:
        raise RuntimeError(
            "Missing camera readback: " + ", ".join(sorted(missing))
        )

    return values, ranges


def configure_camera(config, reset):
    address = config["CameraIP"]

    if reset:
        arv(address, "UserSetSelector=Default", "UserSetLoad=1")

    arv(
        address,
        "GammaEnable=1",
        "ExposureAuto=Off",
        "OffsetX=0",
        "OffsetY=0",
    )

    _, limits = read_features(address, ["Width", "Height"])

    if not all(key in limits for key in ("Width", "Height")):
        raise RuntimeError(
            "Cannot determine the camera's minimum image dimensions."
        )

    # Shrink the ROI before changing binning, then set the final ROI.
    arv(
        address,
        "Width=" + str(int(limits["Width"][0])),
        "Height=" + str(int(limits["Height"][0])),
    )
    arv(
        address,
        "BinningHorizontal=" + str(config["BinningHorizontal"]),
        "BinningVertical=" + str(config["BinningVertical"]),
    )

    arv(
        address,
        *[
            key + "=" + str(config[key])
            for key in (
                "Width",
                "Height",
                "OffsetX",
                "OffsetY",
                "DeviceLinkThroughputLimit",
                "ExposureTime",
                "Gamma",
                "AcquisitionFrameRate",
            )
        ],
    )

    values, limits = read_features(
        address, [item[0] for item in CAMERA]
    )

    for key, _, _, _, _, _, integer in CAMERA:
        config[key] = int(values[key]) if integer else values[key]

    return config, limits


def pipeline(config):
    rate = Fraction(
        str(config["AcquisitionFrameRate"])
    ).limit_denominator(10000)

    features = " ".join(
        [
            "GammaEnable=1",
            "Gamma=" + str(config["Gamma"]),
            "DeviceLinkThroughputLimit="
            + str(config["DeviceLinkThroughputLimit"]),
            "AcquisitionFrameRate="
            + str(config["AcquisitionFrameRate"]),
        ]
    )

    return (
        'aravissrc camera-name="' + config["CameraIP"] + '" '
        "h-binning=" + str(config["BinningHorizontal"]) + " "
        "v-binning=" + str(config["BinningVertical"]) + " "
        "offset-x=" + str(config["OffsetX"]) + " "
        "offset-y=" + str(config["OffsetY"]) + " "
        "exposure-auto=off exposure="
        + str(config["ExposureTime"]) + " "
        'features="' + features + '" ! '
        "video/x-bayer,format=rggb,width="
        + str(config["Width"])
        + ",height=" + str(config["Height"])
        + ",framerate="
        + str(rate.numerator) + "/" + str(rate.denominator) + " ! "
        "bayer2rgb ! videoconvert ! "
        "videoflip method=" + ORIENTATIONS[config.get("Orientation", "Normal")] + " ! "
        "video/x-raw,format=BGR ! "
        "glupload ! glcolorconvert ! gldownload ! "
        "video/x-raw,format=BGR ! "
        "appsink name=opencvsink max-buffers=1 drop=true sync=false"
    )


class Preview:
    def __init__(self, config, detector=None, enabled=True):
        self.config = config
        self.detector = detector
        self.enabled = enabled

        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Condition()

        self.frame = self.annotated = None
        self.sequence = 0
        self.received = 0.0
        self.error = ""

        self.metrics = dict(
            camera=0.0,
            yolo=0.0,
            inference=0.0,
            latency=0.0,
            skipped=0,
        )

        self.record_stop = None
        self.record_thread = None

        self.capture = threading.Thread(
            target=self.capture_loop, daemon=True
        )
        self.inference = threading.Thread(
            target=self.inference_loop, daemon=True
        )

        self.capture.start()
        self.inference.start()

        if not self.ready.wait(8) or self.frame is None:
            error = self.error or (
                "No camera frames received. Check power and Ethernet."
            )
            self.stop()
            raise RuntimeError(error)

    def snapshot(self):
        with self.lock:
            return self.frame, self.received, self.sequence

    def capture_loop(self):
        stream = None

        try:
            stream = Gst.parse_launch(pipeline(self.config))
            sink = stream.get_by_name("opencvsink")
            bus = stream.get_bus()

            if (
                stream.set_state(Gst.State.PLAYING)
                == Gst.StateChangeReturn.FAILURE
            ):
                raise RuntimeError("Cannot start camera stream.")

            count = 0
            start = last_frame = time.perf_counter()

            while not self.stop_event.is_set():
                sample = sink.emit(
                    "try-pull-sample", 200 * Gst.MSECOND
                )
                now = time.perf_counter()

                if sample is None:
                    message = bus.pop_filtered(
                        Gst.MessageType.ERROR | Gst.MessageType.EOS
                    )

                    if message is not None:
                        if message.type == Gst.MessageType.ERROR:
                            error, debug = message.parse_error()
                            raise RuntimeError(
                                str(error) + "\n" + str(debug or "")
                            )
                        raise RuntimeError("Camera stream ended.")

                    if now - last_frame > 5:
                        raise RuntimeError(
                            "No frames received. Check camera and "
                            "click Connect / Apply."
                        )
                    continue

                last_frame = now
                info = GstVideo.VideoInfo.new_from_caps(
                    sample.get_caps()
                )
                buffer = sample.get_buffer()
                ok, mapped = buffer.map(Gst.MapFlags.READ)

                if not ok:
                    raise RuntimeError("Cannot map camera frame.")

                try:
                    frame = np.ndarray(
                        (info.height, info.width, 3),
                        dtype=np.uint8,
                        buffer=mapped.data,
                        strides=(info.stride[0], 3, 1),
                    ).copy()
                finally:
                    buffer.unmap(mapped)

                with self.lock:
                    self.frame = frame
                    self.received = now
                    self.sequence += 1
                    self.lock.notify_all()

                self.ready.set()
                count += 1

                if now - start >= 1:
                    self.metrics["camera"] = count / (now - start)
                    count, start = 0, now

        except Exception as error:
            self.error = str(error)
            self.ready.set()
        finally:
            if stream is not None:
                stream.set_state(Gst.State.NULL)

    def inference_loop(self):
        last = -1
        warm = True
        count = skipped = 0
        infer_sum = latency_sum = 0.0
        start = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                if not self.enabled:
                    self.stop_event.wait(0.05)
                    last, warm = -1, True
                    count = skipped = 0
                    infer_sum = latency_sum = 0.0
                    start = time.perf_counter()
                    self.metrics.update(
                        yolo=0, inference=0, latency=0, skipped=0
                    )
                    continue

                with self.lock:
                    self.lock.wait_for(
                        lambda: self.stop_event.is_set()
                        or (
                            self.frame is not None
                            and self.sequence != last
                        ),
                        timeout=0.1,
                    )
                    frame = self.frame
                    received = self.received
                    seq = self.sequence

                if self.stop_event.is_set():
                    break
                if frame is None or seq == last:
                    continue

                if self.detector is None:
                    self.detector = backend.get_detector(
                        "yolo",
                        model_path=backend.VISION_MODEL_PATH,
                        mode=backend.VISION_MODE,
                        imgsz=self.config["YOLOSize"],
                    )
                    # Get a fresh frame after loading the model.
                    continue

                t0 = time.perf_counter()
                detections = self.detector.detect(frame)
                finished = time.perf_counter()

                display = frame.copy()

                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    cv2.rectangle(
                        display, (x1, y1), (x2, y2),
                        (0, 255, 0), 2,
                    )
                    cx, cy = map(int, detection.center)
                    cv2.circle(
                        display, (cx, cy), 4, (0, 255, 255), -1
                    )
                    cv2.putText(
                        display,
                        f"{detection.confidence:.2f}",
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                self.annotated = display

                if warm:
                    warm = False
                    start = time.perf_counter()
                else:
                    count += 1
                    skipped += max(0, seq - last - 1)
                    infer_sum += (finished - t0) * 1000
                    latency_sum += (finished - received) * 1000

                    if finished - start >= 1:
                        self.metrics.update(
                            yolo=count / (finished - start),
                            inference=infer_sum / count,
                            latency=latency_sum / count,
                            skipped=skipped,
                        )
                        count = skipped = 0
                        infer_sum = latency_sum = 0.0
                        start = finished

                last = seq
                limit = self.config["VisionFPS"]

                if limit > 0:
                    self.stop_event.wait(
                        max(
                            0,
                            1 / limit - (time.perf_counter() - t0),
                        )
                    )

        except Exception as error:
            self.error = "YOLO preview: " + str(error)
            self.enabled = False

    def start_recording(self, path):
        frame, _, _ = self.snapshot()

        if frame is None:
            raise RuntimeError("No frame available.")

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (frame.shape[1], frame.shape[0]),
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Cannot open MJPEG video writer.")

        self.record_stop = threading.Event()

        def record():
            deadline = time.perf_counter()

            try:
                while not self.record_stop.is_set():
                    frame, _, _ = self.snapshot()

                    if frame is not None:
                        writer.write(frame)

                    deadline += 1 / 30

                    if time.perf_counter() - deadline > 0.5:
                        raise RuntimeError(
                            "Encoder cannot keep up with 30 FPS; "
                            "reduce image size."
                        )

                    self.record_stop.wait(
                        max(0, deadline - time.perf_counter())
                    )

            except Exception as error:
                self.error = "Recording: " + str(error)
            finally:
                writer.release()

        self.record_thread = threading.Thread(
            target=record, daemon=True
        )
        self.record_thread.start()

    def stop_recording(self):
        if self.record_thread is not None:
            self.record_stop.set()
            self.record_thread.join(5)

            if self.record_thread.is_alive():
                raise RuntimeError(
                    "Video writer has not stopped yet."
                )

            self.record_thread = None

    def stop(self):
        self.stop_recording()
        self.stop_event.set()

        with self.lock:
            self.lock.notify_all()

        for worker in (self.capture, self.inference):
            worker.join(30)

            if worker.is_alive():
                raise RuntimeError(
                    "Camera/YOLO worker has not stopped; "
                    "please retry."
                )


class Setup:
    def __init__(self, root, preview_only=False):
        self.root = root
        self.preview_only = preview_only
        self.engine = None
        self.applied = self.result = None
        self.reset_address = None
        self.busy = self.closed = False
        self.events = queue.Queue()

        self.variables = {}
        self.sliders = {}
        self.inputs = []

        self.status = tk.StringVar(
            value="Connect the camera, then click Connect / Apply."
        )
        self.stats = tk.StringVar(value="Waiting for camera...")
        self.ip = tk.StringVar(value="169.254.211.45")
        self.orientation = tk.StringVar(value="Normal")
        self.yolo = tk.BooleanVar(value=True)
        self.calibration = tk.BooleanVar(value=False)

        self.cols = tk.IntVar(value=8)
        self.rows = tk.IntVar(value=5)
        self.square = tk.DoubleVar(value=21.61)

        self.calibration_dir = None
        self.photos = self.usable = 0
        self.last_error = ""

        root.title("Nemesis camera setup")
        root.geometry("1180x760")
        root.protocol("WM_DELETE_WINDOW", self.cancel)

        # Packed bottom-up before the body: the status line and the buttons keep
        # their space however large the video gets.
        ttk.Label(
            root,
            textvariable=self.status,
            wraplength=1100,
            padding=(10, 0, 10, 10),
        ).pack(side="bottom", fill="x")

        footer = ttk.Frame(root, padding=10)
        footer.pack(side="bottom", fill="x")

        body = ttk.Frame(root, padding=10)
        body.pack(side="top", fill="both", expand=True)

        tabs = ttk.Notebook(body)
        tabs.pack(side="left", fill="y", padx=(0, 12))

        camera_tab, software_tab, calibration_tab = [
            ttk.Frame(tabs, padding=8) for _ in range(3)
        ]

        for tab, title in zip(
            (camera_tab, software_tab, calibration_tab),
            ("Camera", "YOLO / stream", "Calibration"),
        ):
            tabs.add(tab, text=title)

        ttk.Label(camera_tab, text="Camera IP").grid(
            row=0, column=0, sticky="w"
        )
        ip_entry = ttk.Entry(
            camera_tab, textvariable=self.ip, width=16
        )
        ip_entry.grid(row=0, column=1, sticky="e")
        self.inputs.append(ip_entry)

        ttk.Label(camera_tab, text="Orientation").grid(
            row=1, column=0, sticky="w", pady=(4, 6)
        )
        orientation_box = ttk.Combobox(
            camera_tab,
            textvariable=self.orientation,
            values=list(ORIENTATIONS),
            state="readonly",
            width=24,
        )
        orientation_box.grid(row=1, column=1, sticky="e", pady=(4, 6))
        self.orientation_box = orientation_box

        self.build_fields(camera_tab, CAMERA, 2)
        self.build_fields(software_tab, SOFTWARE, 0)

        ttk.Label(
            software_tab,
            text=(
                "0 removes the YOLO FPS throttle.\n"
                "The first inference is warmup.\n"
                "Photos and videos contain raw frames."
            ),
            justify="left",
        ).grid(
            row=9, column=0, columnspan=2,
            sticky="w", pady=15,
        )

        for i, (label, variable) in enumerate(
            (
                ("Inner corners: columns", self.cols),
                ("Inner corners: rows", self.rows),
                ("Square size (mm)", self.square),
            )
        ):
            ttk.Label(calibration_tab, text=label).grid(
                row=i, column=0, sticky="w", pady=6
            )
            entry = ttk.Entry(
                calibration_tab,
                textvariable=variable,
                width=9,
            )
            entry.grid(row=i, column=1)
            self.inputs.append(entry)

        toggle = ttk.Checkbutton(
            calibration_tab,
            text="Calibration mode",
            variable=self.calibration,
            command=self.toggle_calibration,
        )
        toggle.grid(
            row=3, column=0, columnspan=2,
            sticky="w", pady=12,
        )
        self.inputs.append(toggle)

        self.calibration_label = ttk.Label(
            calibration_tab,
            text="No calibration session.",
            wraplength=290,
        )
        self.calibration_label.grid(
            row=4, column=0, columnspan=2, sticky="w"
        )

        self.compute_button = ttk.Button(
            calibration_tab,
            text="Compute calibration",
            command=self.calibrate,
        )
        self.compute_button.grid(
            row=5, column=0, columnspan=2,
            sticky="ew", pady=12,
        )

        ttk.Label(
            calibration_tab,
            text=(
                "Aim for 15+ varied checkerboard photos.\n"
                "At least 10 usable photos are required.\n"
                "Set camera geometry and focus first."
            ),
            justify="left",
            wraplength=290,
        ).grid(row=6, column=0, columnspan=2)

        view = ttk.Frame(body)
        view.pack(side="left", fill="both", expand=True)

        # Controls first, from the bottom up; the video gets what is left
        actions = ttk.Frame(view)
        actions.pack(side="bottom", fill="x", pady=8)

        ttk.Label(
            view, textvariable=self.stats, justify="left"
        ).pack(side="bottom", anchor="w", pady=5)

        yolo_toggle = ttk.Checkbutton(
            view,
            text="Run YOLO preview",
            variable=self.yolo,
            command=self.toggle_yolo,
        )
        yolo_toggle.pack(side="bottom", anchor="w", pady=5)
        self.inputs.append(yolo_toggle)

        # The picture is scaled to this frame. The frame does not grow to fit
        # the picture (pack_propagate off), or each frame would enlarge it.
        self.preview_area = tk.Frame(view, bg="black")
        self.preview_area.pack(side="top", fill="both", expand=True)
        self.preview_area.pack_propagate(False)
        self.preview = tk.Label(self.preview_area, bg="black")
        self.preview.pack(fill="both", expand=True)

        self.photo_button = ttk.Button(
            actions, text="Take photo", command=self.take_photo
        )
        self.photo_button.pack(side="left", padx=3)

        self.record_button = ttk.Button(
            actions,
            text="Record video (30 FPS)",
            command=self.record,
        )
        self.record_button.pack(side="left", padx=3)

        self.apply_button = ttk.Button(
            footer,
            text="Connect / Apply",
            command=self.apply,
        )
        self.apply_button.pack(side="left")

        self.done_button = ttk.Button(
            footer,
            text="Close" if preview_only else "Done - open dashboard",
            command=self.cancel if preview_only else self.finish,
        )
        self.done_button.pack(side="right")

        # The dashboard with no video: for checking the web page, the override
        # button and the ROS side when the camera is not plugged in.
        self.no_camera_button = ttk.Button(
            footer,
            text="Open dashboard without camera",
            command=self.finish_without_camera,
        )
        if not preview_only:
            self.no_camera_button.pack(side="right", padx=8)

        self.buttons = [
            self.apply_button,
            self.done_button,
            self.no_camera_button,
            self.photo_button,
            self.record_button,
            self.compute_button,
        ]

        root.after(100, self.poll)

    def build_fields(self, parent, fields, start):
        for i, (
            key, label, default, low, high, step, integer
        ) in enumerate(fields):
            row = start + i * 2
            variable = tk.DoubleVar(value=default)
            self.variables[key] = variable

            ttk.Label(parent, text=label).grid(
                row=row, column=0, sticky="w"
            )

            entry = ttk.Spinbox(
                parent,
                from_=low,
                to=high,
                increment=step,
                textvariable=variable,
                width=12,
            )
            entry.grid(row=row, column=1, sticky="e")

            slider = tk.Scale(
                parent,
                variable=variable,
                from_=low,
                to=high,
                resolution=1 if integer else step,
                orient="horizontal",
                showvalue=False,
                length=300,
                highlightthickness=0,
            )
            slider.grid(
                row=row + 1, column=0,
                columnspan=2, sticky="ew",
            )

            self.sliders[key] = (slider, entry, step)
            self.inputs.extend((entry, slider))

    def settings(self):
        config = {
            "CameraIP": str(
                ipaddress.IPv4Address(self.ip.get().strip())
            ),
            "Orientation": self.orientation.get(),
        }
        if config["Orientation"] not in ORIENTATIONS:
            raise ValueError("Choose an orientation from the list.")

        for key, _, _, _, _, _, integer in CAMERA + SOFTWARE:
            value = self.variables[key].get()

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Settings must be finite, nonnegative numbers."
                )

            config[key] = round(value) if integer else value

        for key in (
            "AcquisitionFrameRate",
            "Gamma",
            "ExposureTime",
            "Width",
            "Height",
            "BinningHorizontal",
            "BinningVertical",
            "DeviceLinkThroughputLimit",
            "StreamFPS",
        ):
            if config[key] <= 0:
                raise ValueError(
                    key + " must be greater than zero."
                )

        if not 1 <= config["JPEGQuality"] <= 100:
            raise ValueError("JPEG quality must be 1-100.")

        if config["YOLOSize"] <= 0 or config["YOLOSize"] % 32:
            raise ValueError(
                "YOLO image size must be a positive multiple of 32."
            )

        return config

    def job(self, function, success, text):
        if self.busy:
            return

        self.busy = True
        self.status.set(text)

        for widget in self.buttons + self.inputs:
            widget.configure(state="disabled")

        def work():
            try:
                result = function()
                self.events.put((success, result, None))
            except (Exception, SystemExit) as error:
                self.events.put((success, None, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def apply(self):
        try:
            config = self.settings()
        except Exception as error:
            messagebox.showerror("Settings", str(error))
            return

        old = self.applied
        self.applied = None
        preview_enabled = self.yolo.get()

        def work():
            detector = None

            if self.engine is not None:
                self.engine.stop()

                if old and old["YOLOSize"] == config["YOLOSize"]:
                    detector = self.engine.detector

            actual, limits = configure_camera(
                config,
                reset=config["CameraIP"] != self.reset_address,
            )
            self.reset_address = config["CameraIP"]

            engine = Preview(
                actual, detector, preview_enabled
            )
            return engine, actual, limits

        def success(result):
            self.engine, self.applied, limits = result
            self.engine.enabled = preview_enabled

            for key, value in self.applied.items():
                if key in self.variables:
                    if key in limits:
                        low, high, inc = limits[key]
                        slider, entry, step = self.sliders[key]

                        # Preserve fractional readback values.
                        resolution = (
                            inc if inc and inc > 0 else -1
                        )
                        slider.configure(
                            from_=low,
                            to=high,
                            resolution=resolution,
                        )
                        entry.configure(
                            from_=low,
                            to=high,
                            increment=inc or step,
                        )

                    self.variables[key].set(value)

            self.last_error = ""
            self.record_button.configure(
                text="Record video (30 FPS)"
            )
            self.status.set(
                "Camera ready. Settings were read back "
                "from the device."
            )

        self.job(
            work, success,
            "Applying settings and opening camera...",
        )

    def toggle_yolo(self):
        if self.engine:
            self.engine.enabled = self.yolo.get()

    def toggle_calibration(self):
        if self.calibration.get():
            self.calibration_dir = (
                ROOT / "camera_calibration_images" / stamp()
            )
            self.photos = self.usable = 0
            self.yolo.set(False)
            self.toggle_yolo()

            self.calibration_label.configure(
                text="Session started: 0 photos."
            )
            self.status.set(
                "Calibration mode: subsequent photos "
                "go into a new session."
            )
        else:
            self.status.set(
                "Calibration mode off; photos go to "
                "saved_camera_recordings/photos."
            )

    def take_photo(self):
        if self.engine is None:
            return

        frame, received, _ = self.engine.snapshot()

        if (
            frame is None
            or time.perf_counter() - received > 2
        ):
            messagebox.showerror(
                "Photo", "No recent frame available."
            )
            return

        frame = frame.copy()
        calibration = self.calibration.get()

        try:
            pattern = (self.cols.get(), self.rows.get())
            if min(pattern) < 2:
                raise ValueError(
                    "Checkerboard dimensions must be at least 2."
                )
        except Exception as error:
            messagebox.showerror("Calibration", str(error))
            return

        directory = (
            self.calibration_dir
            if calibration
            else ROOT / "saved_camera_recordings" / "photos"
        )

        def work():
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("photo_" + stamp() + ".png")

            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Could not save photo.")

            board = False

            if calibration:
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )
                board, _ = cv2.findChessboardCorners(
                    gray,
                    pattern,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_FAST_CHECK,
                )

            return path, board

        def success(result):
            path, board = result

            if calibration:
                self.photos += 1
                self.usable += int(board)
                self.calibration_label.configure(
                    text=(
                        f"{self.photos} photos; "
                        f"{self.usable} with detected board."
                    )
                )

            self.status.set(
                "Saved " + str(path.relative_to(ROOT))
            )

        self.job(work, success, "Saving photo...")

    def calibrate(self):
        directory = self.calibration_dir

        if directory is None or not directory.is_dir():
            messagebox.showerror(
                "Calibration",
                "Start calibration mode and take photos first.",
            )
            return

        try:
            cols = self.cols.get()
            rows = self.rows.get()
            square = self.square.get()

            if (
                min(cols, rows) < 2
                or not math.isfinite(square)
                or square <= 0
            ):
                raise ValueError(
                    "Enter valid board dimensions and square size."
                )

        except Exception as error:
            messagebox.showerror("Calibration", str(error))
            return

        def work():
            output = Path(DEFAULT_OUTPUT_NPZ)
            temporary = directory / "result.npz"

            result = compute_intrinsics(
                image_dir=directory,
                pattern_cols=cols,
                pattern_rows=rows,
                square_size=square,
                output_npz=str(temporary),
                min_images=10,
            )

            os.replace(temporary, output)
            return result

        def success(result):
            self.status.set(
                f"Calibration saved: "
                f"{result['images_used']} usable photos; "
                f"RMS reprojection error "
                f"{result['rms']:.3f} pixels."
            )

        self.job(
            work, success, "Computing calibration..."
        )

    def record(self):
        if self.engine is None:
            return

        if self.engine.record_thread is not None:
            self.job(
                self.engine.stop_recording,
                lambda _: (
                    self.record_button.configure(
                        text="Record video (30 FPS)"
                    ),
                    self.status.set("Recording saved."),
                ),
                "Finishing recording...",
            )
            return

        directory = (
            ROOT / "saved_camera_recordings" / "videos"
        )
        path = directory / ("video_" + stamp() + ".avi")

        try:
            _, received, _ = self.engine.snapshot()

            if time.perf_counter() - received > 2:
                raise RuntimeError("No recent frame available.")

            directory.mkdir(parents=True, exist_ok=True)
            self.engine.start_recording(path)

            self.record_button.configure(
                text="Stop recording"
            )
            self.status.set(
                "Recording raw video at 30 FPS: "
                + str(path.relative_to(ROOT))
            )

        except Exception as error:
            messagebox.showerror("Recording", str(error))

    def finish(self):
        try:
            if self.engine is None or self.applied is None:
                raise RuntimeError(
                    "Connect / Apply settings first."
                )

            if self.settings() != self.applied:
                raise RuntimeError(
                    "Settings changed: click Connect / Apply "
                    "before Done."
                )

            _, received, _ = self.engine.snapshot()

            if time.perf_counter() - received > 2:
                raise RuntimeError(
                    "No recent camera frame; reconnect before Done."
                )

            self.check_dashboard_can_start()

        except Exception as error:
            messagebox.showerror(
                "Cannot start dashboard", str(error)
            )
            return

        def success(_):
            self.result = self.applied.copy()
            self.closed = True
            self.root.destroy()

        self.job(
            self.engine.stop,
            success,
            "Releasing camera and starting dashboard...",
        )

    def check_dashboard_can_start(self):
        """Raises if the dashboard's ports are taken or its frontend is not installed."""
        for port in (3001, 8080):
            with socket.socket() as check:
                check.bind(("127.0.0.1", port))

        if not (ROOT / "node_modules").is_dir():
            raise RuntimeError(
                "Frontend dependencies missing. "
                "Run npm install first."
            )

    def finish_without_camera(self):
        """Start the dashboard with no video stream and no vision."""
        try:
            self.check_dashboard_can_start()
            # Stream and JPEG settings still apply; camera values are unused
            config = self.settings()
        except Exception as error:
            messagebox.showerror(
                "Cannot start dashboard", str(error)
            )
            return

        config["NoCamera"] = True

        def success(_):
            self.result = config
            self.closed = True
            self.root.destroy()

        self.job(
            lambda: self.engine.stop() if self.engine else None,
            success,
            "Starting dashboard without the camera...",
        )

    def cancel(self):
        if self.busy:
            self.status.set(
                "Please wait for the current operation to finish."
            )
            return

        def success(_):
            self.closed = True
            self.root.destroy()

        self.job(
            lambda: self.engine.stop() if self.engine else None,
            success,
            "Closing camera...",
        )

    def poll(self):
        try:
            while True:
                callback, result, error = (
                    self.events.get_nowait()
                )
                self.busy = False

                for widget in self.buttons + self.inputs:
                    widget.configure(state="normal")

                if error is not None:
                    self.status.set(error)
                    messagebox.showerror(
                        "Camera setup", error
                    )
                else:
                    callback(result)

                if self.closed:
                    return

        except queue.Empty:
            pass

        if self.engine:
            engine = self.engine
            raw, received, _ = engine.snapshot()

            image = (
                engine.annotated
                if self.yolo.get()
                and engine.annotated is not None
                else raw
            )

            if image is not None:
                width = max(
                    160, self.preview_area.winfo_width()
                )
                height = max(
                    120, self.preview_area.winfo_height()
                )
                scale = min(
                    width / image.shape[1],
                    height / image.shape[0],
                )

                small = cv2.resize(
                    image,
                    (
                        max(1, int(image.shape[1] * scale)),
                        max(1, int(image.shape[0] * scale)),
                    ),
                )
                bitmap = ImageTk.PhotoImage(
                    Image.fromarray(
                        cv2.cvtColor(
                            small, cv2.COLOR_BGR2RGB
                        )
                    )
                )

                self.preview.configure(image=bitmap)
                self.preview.image = bitmap

            metrics = engine.metrics.copy()
            age = (
                (time.perf_counter() - received) * 1000
                if received else 0
            )
            requested = engine.config[
                "AcquisitionFrameRate"
            ]

            self.stats.set(
                f"Camera: {metrics['camera']:.1f} received FPS"
                f" | configured: {requested:.2f} FPS\n"
                f"YOLO: {metrics['yolo']:.1f} FPS"
                f" | inference: {metrics['inference']:.1f} ms\n"
                f"Received frame -> result: "
                f"{metrics['latency']:.1f} ms"
                f" | latest frame age: {age:.0f} ms\n"
                f"Frames skipped by YOLO in last measurement: "
                f"{metrics['skipped']}\n"
                "Camera exposure/network delay is not included "
                "in these latency timers."
            )

            if (
                engine.error
                and engine.error != self.last_error
            ):
                self.last_error = engine.error
                self.status.set(engine.error)

        self.root.after(33, self.poll)


def stop_frontend(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        process.wait()


def run_dashboard(config, detector):
    # None: the backend runs without a video stream (no frames, no detections)
    backend.VIDEO_SOURCE = None if config.get("NoCamera") else pipeline(config)
    backend.CAPTURE_FPS = config["AcquisitionFrameRate"]
    backend.VISION_FPS = config["VisionFPS"]
    backend.STREAM_FPS = config["StreamFPS"]
    backend.JPEG_QUALITY = config["JPEGQuality"]

    factory = backend.get_detector

    def get_detector(name, **kwargs):
        if detector is not None and name == "yolo":
            return detector

        kwargs["imgsz"] = config["YOLOSize"]
        return factory(name, **kwargs)

    backend.get_detector = get_detector
    # Backend options given to this script (--record DIR) go on to main.py
    sys.argv = ["main.py", "--vision", "yolo"] + [a for a in sys.argv[1:] if a != "--preview-only"]

    frontend = None
    stopping = threading.Event()

    def monitor():
        opened = False
        deadline = time.monotonic() + 60

        while not stopping.wait(0.5):
            if frontend.poll() is not None:
                print(
                    "Frontend exited; stopping dashboard.",
                    flush=True,
                )
                os.kill(os.getpid(), signal.SIGINT)
                return

            if opened or time.monotonic() > deadline:
                continue

            try:
                for port in (3001, 8080):
                    with socket.create_connection(
                        ("127.0.0.1", port), timeout=0.3
                    ):
                        pass

                with urllib.request.urlopen(
                    "http://127.0.0.1:8080", timeout=0.8
                ):
                    pass

            except OSError:
                continue

            webbrowser.open("http://localhost:8080")
            opened = True

    try:
        frontend = subprocess.Popen(
            [
                "npm",
                "run",
                "dev",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--strictPort",
            ],
            cwd=ROOT,
            start_new_session=True,
        )

        threading.Thread(
            target=monitor, daemon=True
        ).start()

        print(
            "Starting dashboard. "
            "Press Ctrl+C here to stop it.",
            flush=True,
        )

        backend.main()

    finally:
        stopping.set()

        if frontend is not None:
            stop_frontend(frontend)


def main():
    preview_only = "--preview-only" in sys.argv[1:]

    for executable in ("arv-tool-0.8",) if preview_only else ("arv-tool-0.8", "npm"):
        if shutil.which(executable) is None:
            raise SystemExit(
                "Required executable not found: " + executable
            )

    if not re.search(
        r"GStreamer:\s+YES", cv2.getBuildInformation()
    ):
        raise SystemExit(
            "OpenCV needs GStreamer support. "
            "Use the system OpenCV build."
        )

    Gst.init(None)

    if not Path(backend.VISION_MODEL_PATH).is_file():
        raise SystemExit(
            "YOLO model not found: "
            + backend.VISION_MODEL_PATH
        )

    root = tk.Tk()
    app = Setup(root, preview_only)

    try:
        root.mainloop()
    finally:
        if app.engine:
            app.engine.stop()

    if app.result is not None:
        run_dashboard(
            app.result, app.engine.detector if app.engine else None
        )


if __name__ == "__main__":
    main()