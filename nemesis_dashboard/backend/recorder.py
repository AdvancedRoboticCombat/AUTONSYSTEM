"""Recording (--record DIR): the grabbed frames to an mp4, their times to a csv."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading

import cv2
import numpy as np

from .config import STATUS_PRINT_S

log = logging.getLogger("nemesis.recorder")

# Recording (--record DIR): ffmpeg options for DIR/video.mp4. On the CPU, so
# the detector keeps the GPU to itself. Measured on this PC: 1920x1440 at
# 70 FPS uses about one CPU core, and the encoder can take 320 FPS. No B-frames
# (-bf 0), so frame N of the file is exactly the N-th frame written.
# GPU alternative (half the CPU, a third of the file size, but it shares the GPU):
#   ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "23", "-bf", "0"]
RECORD_ENCODER = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-bf", "0"]
RECORD_QUEUE_FRAMES = 64  # grabbed frames that may wait for the encoder (8 MB each); beyond that they are dropped


class FrameRecorder:
    """Saves the grabbed frames to DIR/video.mp4 and, for each frame in that
    file, the Unix time it was grabbed to DIR/frames.csv (columns frame, t).

    The frames wait in a bounded queue for a writer thread that pipes them to
    ffmpeg. When the encoder cannot keep up the queue fills and new frames are
    dropped (no row for them) rather than slowing the capture down."""

    def __init__(self, folder: str, scale: float = 1.0):
        self.video_path = os.path.join(folder, "video.mp4")
        self.csv_path = os.path.join(folder, "frames.csv")
        if os.path.exists(self.video_path):
            raise SystemExit(f"{self.video_path} already exists: give --record another folder.")
        os.makedirs(folder, exist_ok=True)
        self.scale = scale
        self.dropped = 0
        self._last_drop_print = 0.0
        self._queue = queue.Queue(maxsize=RECORD_QUEUE_FRAMES)
        self._thread = threading.Thread(target=self._writer_thread_fn, daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray, t: float, fps: float):
        """Queue one grabbed frame; never blocks. t: time.time() at the grab.
        fps: nominal rate of the source, the playback speed of the video."""
        try:
            self._queue.put_nowait((frame, t, fps))
        except queue.Full:
            self.dropped += 1
            if t - self._last_drop_print >= STATUS_PRINT_S:
                self._last_drop_print = t
                log.warning(f"⚠️ Recording cannot keep up: {self.dropped} frames dropped so far")

    def close(self):
        """Encode what is still queued and finish both files."""
        self._queue.put(None)
        self._thread.join()

    def _writer_thread_fn(self):
        ffmpeg = None
        written = 0
        try:
            with open(self.csv_path, "w", buffering=1) as rows:  # line-buffered: complete after a crash too
                rows.write("frame,t\n")
                while (item := self._queue.get()) is not None:
                    frame, t, fps = item
                    if ffmpeg is None:
                        # The first frame sets the size (H.264 needs even numbers)
                        h, w = frame.shape[:2]
                        size = (int(w * self.scale) // 2 * 2, int(h * self.scale) // 2 * 2)
                        # Own session: a Ctrl-C in the terminal must not reach ffmpeg
                        # before the queued frames are written.
                        ffmpeg = subprocess.Popen(
                            ["ffmpeg", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
                             *RECORD_ENCODER, "-pix_fmt", "yuv420p", self.video_path],
                            stdin=subprocess.PIPE, start_new_session=True,
                        )
                        log.info(f"⏺ Recording {size[0]}x{size[1]} to {self.video_path}")
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    ffmpeg.stdin.write(frame)
                    ffmpeg.stdin.flush()
                    rows.write(f"{written},{t:.6f}\n")
                    written += 1
        except Exception as error:  # ffmpeg missing or stopped (its own message is above), disk full...
            log.error(f"❌ Recording stopped after {written} frames: {error!r}")
            while self._queue.get() is not None:  # keep the queue empty until close()
                pass
        if ffmpeg:
            # Closing its input makes ffmpeg finish the mp4; without this it would not play.
            try:
                ffmpeg.stdin.close()
            except OSError:
                pass
            if ffmpeg.wait() != 0:
                log.error(f"❌ ffmpeg failed: {self.video_path} is not usable")
                return
        log.info(f"⏹ Recording: {written} frames in {self.video_path}, {self.dropped} dropped")
