"""The loops, one thread each:

    capture_loop      GigE camera (GStreamer) -> newest frame
    fight_video_loop  recorded fight -> newest frame, instead of the camera
    vision_loop       newest frame -> detector -> homography -> tracker -> ROS, page
    stream_loop       newest frame -> JPEG -> page, only when the page asks

Every loop takes the Dashboard (app.py) and ends when app.stop is set.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from utils.geometry import adjust_center

from .config import STATUS_PRINT_S
from .tracker import DualTracker

if TYPE_CHECKING:
    from .app import Dashboard

log = logging.getLogger("nemesis.pipeline")

WAIT_S = 0.25  # longest a loop sleeps before it looks at app.stop again


def capture_loop(app: "Dashboard"):
    cfg = app.config
    if cfg.video_source is None:
        log.info("No camera selected: the dashboard is running without video or detections.")
        return
    cap = cv2.VideoCapture(cfg.video_source, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        log.error("❌ Could not open video source: %s", cfg.video_source)
        return

    initialized = False

    try:
        frame_count = 0
        fps_timer = time.perf_counter()
        while not app.stop.is_set() and app.fight_run == 0:  # the fight video takes over from the camera
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t_grab = time.time()  # Unix time, for --record
            if not ok:
                time.sleep(0.005)
                continue
            if not initialized:
                log.info("✅ Camera resolution detected: %dx%d", frame.shape[1], frame.shape[0])
                initialized = True

            # Camera frame rate, for the dashboard and the vision loop's status line
            frame_count += 1
            if frame_count >= 30:
                elapsed = time.perf_counter() - fps_timer
                app.state.update("systemHealth", "vision", captureFps=round(frame_count / elapsed, 1))
                frame_count = 0
                fps_timer = time.perf_counter()

            app.frames.put(frame, t0)
            if app.recorder:
                app.recorder.add(frame, t_grab, cfg.capture_fps)
            app.ros.publish_image(frame)
    finally:
        cap.release()


def fight_video_loop(app: "Dashboard", run: int):
    """Feeds the fight video to the pipeline at its own frame rate, then holds
    the last frame. Gives way as soon as the video is started again."""
    path = app.config.fight_video_path
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error("❌ Could not open fight video: %s", path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    app.state.update("systemHealth", "vision", captureFps=round(fps, 1))
    log.info("▶ Fight video: %s (%.0f FPS)", path, fps)

    start = time.perf_counter()
    frame_no = 0
    while not app.stop.is_set() and run == app.fight_run:
        ok, frame = cap.read()
        if not ok:
            log.info("Fight video finished; holding the last frame.")
            break
        wait = start + frame_no / fps - time.perf_counter()  # real time, no faster
        if wait > 0:
            time.sleep(wait)
        app.frames.put(frame, time.perf_counter())
        if app.recorder:
            app.recorder.add(frame, time.time(), fps)
        frame_no += 1
    cap.release()


def stream_loop(app: "Dashboard"):
    """JPEG frames for the page. One is encoded only when a page has asked for
    the next frame (server.frame_wanted), so nothing queues up behind a slow
    browser and nothing is encoded while nobody is watching."""
    cfg = app.config
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
    period = 1.0 / float(cfg.stream_fps) if cfg.stream_fps > 0 else 0.0
    last_sent_seq = -1
    last_sent_t = 0.0

    while not app.stop.is_set():
        if not app.server.frame_wanted.wait(WAIT_S):
            continue
        frame, _, seq = app.frames.wait_newer(last_sent_seq, WAIT_S)
        if frame is None:
            continue

        # No faster than stream_fps; after the pause, send what is newest by then
        pause = last_sent_t + period - time.perf_counter()
        if pause > 0:
            if app.stop.wait(pause):
                break
            frame, _, seq = app.frames.get()
        last_sent_seq = seq
        last_sent_t = time.perf_counter()

        height, width = frame.shape[:2]
        if cfg.stream_width and width > cfg.stream_width:
            size = (int(cfg.stream_width), max(1, round(height * cfg.stream_width / width)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            app.server.frame_wanted.clear()
            app.server.publish_frame(buf.tobytes())


def vision_loop(app: "Dashboard"):
    cfg = app.config
    period = 1.0 / float(cfg.vision_fps) if cfg.vision_fps > 0 else 0.0
    last_frame_seq = -1
    tracker = DualTracker()

    vision_frame_count = 0
    vision_fps_timer = time.perf_counter()
    skipped_frames = 0  # frames produced by the capture loop but never seen by the detector
    latency_acc = 0.0   # accumulated capture→output latency over current batch
    last_status_print = 0.0
    last_error_print = 0.0

    while not app.stop.is_set():
        # Sleeps until the next frame lands, then starts on it at once
        frame, frame_ts, seq = app.frames.wait_newer(last_frame_seq, WAIT_S)
        if frame is None:
            continue
        t0 = time.perf_counter()

        # Track skipped frames (seq jumped by more than 1)
        if last_frame_seq != -1 and seq > last_frame_seq + 1:
            skipped_frames += seq - last_frame_seq - 1

        last_frame_seq = seq

        # ── Run pluggable detector, collect raw detections ─────────────
        current_detector = app.detector  # read once: the fight-video button can swap it
        homography = app.homography      # likewise
        if current_detector is None:     # the fight model failed to load
            continue
        try:
            detections = current_detector.detect(frame)
        except Exception:
            # One bad frame must not end vision for the rest of the match
            now = time.perf_counter()
            if now - last_error_print >= STATUS_PRINT_S:
                last_error_print = now
                log.exception("❌ Detector failed on a frame")
            continue

        raw_detections = []  # each: {pixel, conf, seg, cls}
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            adjusted_cx, adjusted_cy = adjust_center(
                det.center[0], det.center[1], y1, y2, frame_height=frame.shape[0]
            )
            raw_detections.append(
                {'pixel': (adjusted_cx, adjusted_cy), 'conf': det.confidence, 'seg': det.mask,
                 'cls': det.class_id}
            )

        # A model with a "nemesis" class tells the robots apart itself: Nemesis is
        # its most confident nemesis detection, the opponent the most confident other.
        nemesis_class = current_detector.nemesis_class
        nem_idx = opp_idx = None
        if nemesis_class is not None:
            def most_confident(indices):
                return max(indices, key=lambda i: raw_detections[i]['conf'], default=None)
            nem_idx = most_confident(i for i, d in enumerate(raw_detections) if d['cls'] == nemesis_class)
            opp_idx = most_confident(i for i, d in enumerate(raw_detections) if d['cls'] != nemesis_class)

        nem = None
        opp = None
        nem_h = None
        opp_h = None
        nem_conf = None
        opp_conf = None
        nem_seg = None
        opp_seg = None
        nem_seen = False  # matched to a detection this frame (False = tracker is coasting)
        opp_seen = False

        if homography.matrix is not None:
            # ── Homography FIRST: pixel → arena for every detection ──
            arena_positions = []
            for det in raw_detections:
                pt = np.array([[[det['pixel'][0], det['pixel'][1]]]], dtype=np.float32)
                res = cv2.perspectiveTransform(pt, homography.matrix)
                arena_positions.append((float(res[0][0][0]), float(res[0][0][1])))

            # ── Tracker assigns labels in arena coords ──────────────
            if nemesis_class is not None:
                nem_info, opp_info = tracker.step_known(arena_positions, nem_idx, opp_idx, t0)
            else:
                nem_info, opp_info = tracker.step(arena_positions, t0)

            # Extract nemesis result
            if nem_info is not None:
                arena_pos, det_idx = nem_info
                nem_h = arena_pos
                if det_idx is not None:
                    det = raw_detections[det_idx]
                    nem = det['pixel']
                    nem_conf = det['conf']
                    nem_seg = det['seg']
                    nem_seen = True
                elif homography.inv_matrix is not None:
                    # Coasting — project interpolated arena pos back to pixel
                    pt = np.array([[[arena_pos[0], arena_pos[1]]]], dtype=np.float32)
                    px = cv2.perspectiveTransform(pt, homography.inv_matrix)
                    nem = (float(px[0][0][0]), float(px[0][0][1]))

            # Extract opponent result
            if opp_info is not None:
                arena_pos, det_idx = opp_info
                opp_h = arena_pos
                if det_idx is not None:
                    det = raw_detections[det_idx]
                    opp = det['pixel']
                    opp_conf = det['conf']
                    opp_seg = det['seg']
                    opp_seen = True
                elif homography.inv_matrix is not None:
                    pt = np.array([[[arena_pos[0], arena_pos[1]]]], dtype=np.float32)
                    px = cv2.perspectiveTransform(pt, homography.inv_matrix)
                    opp = (float(px[0][0][0]), float(px[0][0][1]))
        elif nemesis_class is not None:
            # ── No homography, two-class model: identity from the class ──
            if nem_idx is not None:
                d = raw_detections[nem_idx]
                nem, nem_conf, nem_seg = d['pixel'], d['conf'], d['seg']
            if opp_idx is not None:
                d = raw_detections[opp_idx]
                opp, opp_conf, opp_seg = d['pixel'], d['conf'], d['seg']
        else:
            # ── No homography: fallback sort-by-x (old behaviour) ───
            raw_detections.sort(key=lambda d: d['pixel'][0])
            if len(raw_detections) >= 1:
                d = raw_detections[0]
                nem, nem_conf, nem_seg = d['pixel'], d['conf'], d['seg']
            if len(raw_detections) >= 2:
                d = raw_detections[1]
                opp, opp_conf, opp_seg = d['pixel'], d['conf'], d['seg']

        # Swap applied atomically before any publish/emit
        if app.swap:
            nem, opp = opp, nem
            nem_conf, opp_conf = opp_conf, nem_conf
            nem_h, opp_h = opp_h, nem_h
            nem_seg, opp_seg = opp_seg, nem_seg
            nem_seen, opp_seen = opp_seen, nem_seen

        # The robot first: centroids to ROS before anything for the page
        if nem_h is not None:
            app.ros.publish_centroid("nemesis", nem_h[0], nem_h[1], coasting=not nem_seen)
        if opp_h is not None:
            app.ros.publish_centroid("opponent", opp_h[0], opp_h[1], coasting=not opp_seen)

        # Confidences for the page (after the swap, so labels match)
        app.state.update(
            "systemHealth", "vision",
            nemConfidence=round(nem_conf * 100, 1) if nem_conf is not None else 0,
            oppConfidence=round(opp_conf * 100, 1) if opp_conf is not None else 0,
        )

        # Positions and outlines for the page, in pixels of the camera frame
        app.frame_size = (frame.shape[1], frame.shape[0])
        app.server.publish_vision({
            "detW": frame.shape[1],
            "detH": frame.shape[0],
            "nemesis": None if nem is None else {"cx": nem[0], "cy": nem[1]},
            "opponent": None if opp is None else {"cx": opp[0], "cy": opp[1]},
            "nemesisHomog": None if nem_h is None else {"x": nem_h[0], "y": nem_h[1]},
            "opponentHomog": None if opp_h is None else {"x": opp_h[0], "y": opp_h[1]},
            "nemesisSegment": nem_seg,
            "opponentSegment": opp_seg,
            "invHomog": homography.inv_matrix.flatten().tolist() if homography.inv_matrix is not None else None,
            "ts": t0,
        })

        latency_acc += (time.perf_counter() - frame_ts) * 1000
        vision_frame_count += 1
        if vision_frame_count >= 20:
            now = time.perf_counter()
            vision_fps = vision_frame_count / (now - vision_fps_timer)
            avg_latency = latency_acc / vision_frame_count
            # latencyMs: frame grabbed -> positions published
            app.state.update("systemHealth", "vision", fps=round(vision_fps, 1), latencyMs=round(avg_latency, 1))
            # One status line every few seconds instead of one per batch of frames
            if now - last_status_print >= STATUS_PRINT_S:
                last_status_print = now
                log.info("Camera %.1f FPS | Vision %.1f FPS | grab-to-result %.1f ms | skipped %d frames",
                         app.state.capture_fps(), vision_fps, avg_latency, skipped_frames)
            vision_frame_count = 0
            vision_fps_timer = now
            skipped_frames = 0
            latency_acc = 0.0

        elapsed = time.perf_counter() - t0
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
