"""Top-down replay of a recorded fight.

Runs the segmentation detector on every frame, takes the best `nemesis` and the best `robot`
(opponent) detection, projects each robot's floor point through the arena homography and renders
a side-by-side video: camera frame with the two outlines on the left, the 48 x 48 ft arena seen
from above on the right (4 ft grid, 2 s trails). Per-frame positions go to a CSV next to the video.

Arena frame (same as the homography JSON): (0, 0) is the far-left floor corner as seen in the
video, x runs right along the far wall, y runs toward the camera, both in feet.

    python ZACVISIONQUEST/fight_topdown.py            # defaults reproduce runs/fight_video/
"""
import argparse
import csv
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER

ROOT = Path(__file__).resolve().parent
ARENA_FT = 48.0
RED, GREEN = (0, 0, 255), (0, 220, 0)                      # BGR: Nemesis, opponent
ROBOTS = {"nemesis": ("Nemesis", RED), "robot": ("Opponent", GREEN)}
W, H = 1280, 720                                           # output size
CAM_W, CAM_H = 760, 570                                    # camera panel (4:3)
PX_PER_FT = 10                                             # top-down scale
MAP_X0, MAP_Y0 = CAM_W + 32, 130                           # top-left corner of the arena drawing


def load_homography(path):
    """Pixel -> arena feet, computed exactly as the dashboard does."""
    d = json.loads(Path(path).read_text())
    Hm, _ = cv2.findHomography(np.array(d["pixel_points"], np.float32),
                               np.array(d["arena_points_ft"], np.float32), cv2.RANSAC, 5.0)
    return Hm


def floor_point(poly):
    """Where the robot meets the floor: horizontally centred, 85 % of the way down its mask."""
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    return np.array([(x0 + x1) / 2, y0 + 0.85 * (y1 - y0)])


def to_arena(Hm, pt):
    return cv2.perspectiveTransform(np.array([[pt]], np.float64), Hm)[0, 0]


def best_detections(result, names):
    """{class name: (confidence, mask polygon)} for the highest-confidence detection of each class."""
    best = {}
    if result.masks is None:
        return best
    for cls, conf, poly in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.masks.xy):
        name = names[int(cls)]
        if name in ROBOTS and len(poly) >= 3 and conf > best.get(name, (0,))[0]:
            best[name] = (conf, poly)
    return best


def map_px(xy):
    """Arena feet -> pixel in the output canvas (clamped to the arena drawing)."""
    x, y = np.clip(xy, 0, ARENA_FT) * PX_PER_FT
    return int(round(MAP_X0 + x)), int(round(MAP_Y0 + y))


def draw_map_background():
    canvas = np.full((H, W, 3), 24, np.uint8)
    size = int(ARENA_FT * PX_PER_FT)
    cv2.rectangle(canvas, (MAP_X0, MAP_Y0), (MAP_X0 + size, MAP_Y0 + size), (48, 48, 48), -1)
    for ft in range(0, int(ARENA_FT) + 1, 4):
        p = ft * PX_PER_FT
        cv2.line(canvas, (MAP_X0 + p, MAP_Y0), (MAP_X0 + p, MAP_Y0 + size), (80, 80, 80), 1)
        cv2.line(canvas, (MAP_X0, MAP_Y0 + p), (MAP_X0 + size, MAP_Y0 + p), (80, 80, 80), 1)
        if ft % 8 == 0:
            cv2.putText(canvas, str(ft), (MAP_X0 + p - 8, MAP_Y0 + size + 16), 0, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(ft), (MAP_X0 - 24, MAP_Y0 + p + 4), 0, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (MAP_X0, MAP_Y0), (MAP_X0 + size, MAP_Y0 + size), (200, 200, 200), 2)
    cv2.putText(canvas, "far wall  (x ft ->)", (MAP_X0 + 170, MAP_Y0 - 8), 0, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "camera side  (4 ft grid)", (MAP_X0 + 150, MAP_Y0 + size + 36), 0, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    for i, (label, colour) in enumerate(ROBOTS.values()):
        cv2.circle(canvas, (MAP_X0 + 10 + 150 * i, 82), 6, colour, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (MAP_X0 + 24 + 150 * i, 87), 0, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "hollow = not detected, last position held", (MAP_X0, 108), 0, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def draw_trail(canvas, trail, colour):
    """Fading line through the last ~2 s of detected positions (older = dimmer)."""
    pts = list(trail)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        fade = i / len(pts)
        c = tuple(int(48 + (ch - 48) * fade) for ch in colour)
        cv2.line(canvas, map_px(pts[i - 1]), map_px(pts[i]), c, 2, cv2.LINE_AA)


def open_writer(path, fps):
    """H.264 through ffmpeg when available, otherwise OpenCV's mp4v."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return (lambda f: proc.stdin.write(f.tobytes())), (lambda: (proc.stdin.close(), proc.wait()))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    return vw.write, vw.release


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="/home/arc-lab-pc/battle_gamma08_fight.mp4")
    ap.add_argument("--weights", default=str(ROOT / "runs/sweep/yolo26m-seg_1280_fightbg/weights/chosen.pt"))
    ap.add_argument("--homography", default=str(ROOT.parent / "nemesis_dashboard/fight_video_homography.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "runs/fight_video"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--trail-s", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=0, help="stop early (0 = whole video)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Hm = load_homography(args.homography)
    model = YOLO(args.weights)
    LOGGER.setLevel("ERROR")                                           # predict() warns about half= on every batch
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    scale = CAM_W / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_y0 = (H - CAM_H) // 2

    background = draw_map_background()
    write, close = open_writer(out_dir / "fight_topdown.mp4", fps)
    csv_file = open(out_dir / "positions.csv", "w", newline="")
    rows = csv.writer(csv_file)
    rows.writerow(["frame", "time_s"] + [f"{n}_{k}" for n in ROBOTS for k in ("detected", "conf", "px_x", "px_y", "x_ft", "y_ft")])

    last = {n: None for n in ROBOTS}                                   # last known arena position
    trails = {n: deque(maxlen=int(round(args.trail_s * fps))) for n in ROBOTS}
    seen = {n: 0 for n in ROBOTS}
    frame_no = 0
    while frame_no < total:
        frames = []
        while len(frames) < args.batch and frame_no + len(frames) < total:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            break
        results = model.predict(frames, imgsz=args.imgsz, conf=args.conf, half=True, verbose=False)
        for frame, result in zip(frames, results):
            dets = best_detections(result, model.names)
            canvas = background.copy()
            cam = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
            row = [frame_no, f"{frame_no / fps:.3f}"]
            for name, (label, colour) in ROBOTS.items():
                if name in dets:
                    conf, poly = dets[name]
                    px = floor_point(poly)
                    last[name] = to_arena(Hm, px)
                    seen[name] += 1
                    trails[name].append(last[name])
                    cv2.polylines(cam, [np.round(poly * scale).astype(np.int32)], True, colour, 2, cv2.LINE_AA)
                    cv2.circle(cam, tuple(np.round(px * scale).astype(int)), 3, colour, -1, cv2.LINE_AA)
                    row += [1, f"{conf:.3f}", f"{px[0]:.1f}", f"{px[1]:.1f}"]
                else:
                    trails[name].append(None)
                    row += [0, "", "", ""]
                row += ["", ""] if last[name] is None else [f"{last[name][0]:.2f}", f"{last[name][1]:.2f}"]
                draw_trail(canvas, trails[name], colour)
                if last[name] is not None:
                    cv2.circle(canvas, map_px(last[name]), 7, colour, -1 if name in dets else 2, cv2.LINE_AA)
            rows.writerow(row)
            canvas[cam_y0:cam_y0 + CAM_H, :CAM_W] = cam
            cv2.putText(canvas, f"frame {frame_no:5d} / {total}    t = {frame_no / fps:6.2f} s", (MAP_X0, 44), 0, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            write(canvas)
            frame_no += 1
        if frame_no % 400 < len(frames):
            print(f"{frame_no}/{total} frames", flush=True)
    close()
    csv_file.close()
    cap.release()
    print(f"wrote {out_dir / 'fight_topdown.mp4'} and positions.csv ({frame_no} frames)")
    for name, (label, _) in ROBOTS.items():
        print(f"  {label}: detected in {seen[name]}/{frame_no} frames ({100 * seen[name] / max(frame_no, 1):.1f} %)")


if __name__ == "__main__":
    main()
