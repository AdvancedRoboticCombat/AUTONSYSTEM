"""
Turn the Unity renders into YOLO segmentation labels.

    python ZACVISIONQUEST/unity_to_yolo.py                        # the clips in Zac_vis_final_segmented_dataset/Datasets
    python ZACVISIONQUEST/unity_to_yolo.py --renders DIR [DIR..]  # these folders instead (searched for run_*/sim*.mp4)
    python ZACVISIONQUEST/unity_to_yolo.py --preview 60           # also save 60 labelled frames to look at

Writes dataset/sim/{images,labels}. Renders are for training only; they never
go into valid or test.

The Unity export has video and a yaw angle per frame, but no boxes or masks.
The robots are rendered over a still photo of the arena, so the labels are
rebuilt from the picture:

  1. The empty arena is the per-pixel median over all the clips. Whatever
     differs from it in a frame is robot. That gives the exact outline of
     the robots, but not which robot is which, and two touching robots come
     out as one shape.
  2. Apart: Nemesis is the outline with red and gold paint (class 1), the
     other is class 0. These frames are also shown to SAM 2 so it knows what
     each robot looks like.
  3. Touching: SAM 2 says which pixels belong to which robot and the joint
     outline is divided along that boundary.
  4. A touching frame is skipped when the division is not credible: SAM gave
     the same pixels to both robots, or a robot came out the wrong size.
     That mostly happens when one robot is on top of the other.
"""
import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RENDERS = HERE / "Zac_vis_final_segmented_dataset" / "Datasets"
OUT = HERE / "dataset" / "sim"
SAM_WEIGHTS = HERE / "weights" / "sam2.1_b.pt"
ROBOT, NEMESIS = 0, 1

DIFF = 16            # grey levels a pixel must differ from the empty arena to count as robot
MIN_AREA = 250       # px; smaller blobs are compression noise
NEMESIS_RED = 0.03   # share of a robot that is red / gold paint for it to be Nemesis
LONG_CLIP = 2000     # frames; longer clips are sampled more thinly


def find_clips(folders):
    clips = []
    for folder in folders:
        clips += sorted(Path(folder).glob("**/run_*/sim*.mp4"))
    return clips


def clip_name(video):
    return re.sub(r"[^A-Za-z0-9\-]+", "_", f"{video.parts[-3]}_{video.parts[-2]}_{video.stem}")


def empty_arena(videos):
    frames = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in np.linspace(0, n - 1, 8).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                frames.append(img)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def red_share(frame, mask):
    """Share of a mask that is Nemesis's paint: red trim or gold sunburst."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 8) | (h > 170)) & (s > 120) & (v > 70)
    gold = (h > 15) & (h < 35) & (s > 70) & (v > 150)
    return float(((red | gold) & (mask > 0)).sum()) / max(int(mask.sum()), 1)


def robot_blobs(frame, arena_blur, close=13):
    """Filled masks of everything that is not arena, largest first."""
    d = np.abs(cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16) - arena_blur).max(axis=2)
    m = (d > DIFF).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < MIN_AREA:
            continue
        mask = np.zeros(m.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 1, cv2.FILLED)
        blobs.append(mask)
    return blobs


def largest_part(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    return (lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.uint8)


def polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 6 and cv2.contourArea(c) >= MIN_AREA else None


def identify(frame, blobs):
    """Index of the Nemesis blob and of the other one, by paint. None when the paint does not settle it."""
    shares = [red_share(frame, b) for b in blobs]
    nem = int(np.argmax(shares))
    if shares[nem] < NEMESIS_RED or shares[1 - nem] > NEMESIS_RED:
        return None
    return nem, 1 - nem


def box_of(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def divide(blob, sam, sizes):
    """Divide one outline holding both robots along the tracker's boundary. None when the result is not credible."""
    both = int((sam[0] & sam[1]).sum())
    if both > 0.25 * min(int(sam[0].sum()), int(sam[1].sum())):      # the tracker gave the same pixels to both robots
        return None
    mine = [blob & sam[0] & (1 - sam[1]), blob & sam[1] & (1 - sam[0])]
    if min(int(m.sum()) for m in mine) < MIN_AREA:
        return None
    # pixels neither or both trackers claim go to the nearer robot
    d = [cv2.distanceTransform(1 - m, cv2.DIST_L2, 3) for m in mine]
    undecided = (blob > 0) & (mine[0] == 0) & (mine[1] == 0)
    mine[0] = mine[0] | (undecided & (d[0] <= d[1])).astype(np.uint8)
    mine[1] = mine[1] | (undecided & (d[0] > d[1])).astype(np.uint8)
    mine = [largest_part(m) for m in mine]
    for k in range(2):
        if not 0.5 < mine[k].sum() / sizes[k] < 1.8:                 # robots do not change size from one frame to the next
            return None
        if not 0.6 < mine[k].sum() / max(int(sam[k].sum()), 1) < 1.6:
            return None
    return mine


class Labeller:
    """Labels the frames of one clip. Index 0 is Nemesis, index 1 the other robot."""

    def __init__(self, arena_blur):
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor
        self.arena_blur = arena_blur
        self.sam = SAM2DynamicInteractivePredictor(
            overrides=dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model=str(SAM_WEIGHTS), save=False,
                           verbose=False), max_obj_num=2)
        self.sizes = {}          # each robot's area the last time they were apart
        self.shown = 0           # times the tracker has been shown the robots apart

    def apart(self, frame, blobs):
        who = identify(frame, blobs)
        if who is None:
            return None
        ordered = [blobs[who[0]], blobs[who[1]]]
        if self.sizes and not all(0.4 < ordered[k].sum() / self.sizes[k] < 2.5 for k in range(2)):
            return None
        self.sizes = {k: int(ordered[k].sum()) for k in range(2)}
        if self.shown % 3 == 0:                                      # keep the tracker's memory of each robot fresh
            self.sam(source=frame, bboxes=[box_of(m) for m in ordered], obj_ids=[0, 1], update_memory=True)
        self.shown += 1
        return [(NEMESIS, ordered[0]), (ROBOT, ordered[1])]

    def label(self, frame, robots_in_clip):
        """([(class, mask), ...], touching) for one frame, or None when it cannot be labelled cleanly."""
        blobs = robot_blobs(frame, self.arena_blur)
        if len(blobs) > robots_in_clip:                              # a fork or wheel came out as its own blob: close harder
            blobs = robot_blobs(frame, self.arena_blur, close=27)
        if robots_in_clip == 1:
            if len(blobs) != 1:
                return None
            return [(NEMESIS if red_share(frame, blobs[0]) > NEMESIS_RED else ROBOT, blobs[0])], False
        if len(blobs) == 2:
            out = self.apart(frame, blobs)
            return (out, False) if out else None
        if len(blobs) != 1 or not self.sizes:
            return None
        # one outline. Nearly touching robots are only joined by the closing step: try without it
        loose = robot_blobs(frame, self.arena_blur, close=3)
        if len(loose) == 2:
            out = self.apart(frame, loose)
            if out:
                return out, False
        res = self.sam(source=frame)[0]
        if res.masks is None or len(res.masks.data) != 2:
            return None
        masks = res.masks.data.cpu().numpy().astype(np.uint8)
        sam = [None, None]
        for j, obj in enumerate(res.boxes.cls.tolist()):
            sam[int(obj)] = masks[j]
        if sam[0] is None or sam[1] is None:
            return None
        parts = divide(blobs[0], sam, self.sizes)
        if parts is None:
            return None
        return [(NEMESIS, parts[0]), (ROBOT, parts[1])], True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", nargs="+", default=[str(RENDERS)], help="folders holding Unity output")
    ap.add_argument("--every", type=int, default=4, help="short clips: keep one frame in this many (60 fps; default 4)")
    ap.add_argument("--every-long", type=int, default=24, help=f"clips over {LONG_CLIP} frames: one in this many (default 24)")
    ap.add_argument("--preview", type=int, default=0, help="save this many labelled frames to dataset/sim/preview")
    args = ap.parse_args()

    videos = find_clips(args.renders)
    print(f"{len(videos)} clips")
    arena = empty_arena(videos)
    arena_blur = cv2.GaussianBlur(arena, (3, 3), 0).astype(np.int16)
    shutil.rmtree(OUT, ignore_errors=True)
    for sub in ("images", "labels") + (("preview",) if args.preview else ()):
        (OUT / sub).mkdir(parents=True)
    SAM_WEIGHTS.parent.mkdir(exist_ok=True)

    total = dict(kept=0, touching=0, skipped=0, nemesis=0, robot=0)
    previews = []
    print(f"{'clip':46s} {'frames':>6s} {'robots':>6s} {'kept':>5s} {'touching':>8s} {'skipped':>7s}")
    for v in videos:
        clip = clip_name(v)
        cap = cv2.VideoCapture(str(v))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        every = args.every_long if n_frames > LONG_CLIP else args.every
        # how many robots the clip has: the most blobs seen in a spread of frames
        counts = []
        for f in np.linspace(0, n_frames - 1, 12).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                counts.append(len(robot_blobs(img, arena_blur)))
        robots_in_clip = min(2, max(counts))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        labeller = Labeller(arena_blur)
        kept, touching, skipped = 0, 0, 0
        for f in range(n_frames):
            if not cap.grab():
                break
            if f % every:
                continue
            ok, img = cap.retrieve()
            out = labeller.label(img, robots_in_clip) if ok else None
            polys = [(cls, polygon(m)) for cls, m in out[0]] if out else []
            if not out or any(p is None for _, p in polys):
                skipped += 1
                continue
            kept += 1
            touching += out[1]
            h, w = img.shape[:2]
            name = f"unity_{clip}_f{f:05d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            with open(OUT / "labels" / f"{name}.txt", "w") as fh:
                for cls, p in polys:
                    fh.write(f"{cls} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p) + "\n")
                    total["nemesis" if cls == NEMESIS else "robot"] += 1
            previews.append((name, out[1]))
        cap.release()
        del labeller
        print(f"{clip:46s} {n_frames:6d} {robots_in_clip:6d} {kept:5d} {touching:8d} {skipped:7d}", flush=True)
        total["kept"] += kept; total["touching"] += touching; total["skipped"] += skipped
    print(f"\n{total['kept']} images written ({total['touching']} with the robots touching), {total['skipped']} skipped")
    print(f"objects: nemesis {total['nemesis']}, robot {total['robot']}")
    print(f"wrote {OUT}")

    if args.preview:
        rng = np.random.default_rng(0)
        touch = [p for p in previews if p[1]]
        apart = [p for p in previews if not p[1]]
        pick = [touch[i] for i in rng.permutation(len(touch))[:args.preview // 2]]
        pick += [apart[i] for i in rng.permutation(len(apart))[:args.preview - len(pick)]]
        for name, was_touching in pick:
            img = cv2.imread(str(OUT / "images" / f"{name}.jpg"))
            h, w = img.shape[:2]
            pts_all = []
            for line in (OUT / "labels" / f"{name}.txt").read_text().splitlines():
                vals = line.split()
                pts = (np.array(vals[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255) if vals[0] == "1" else (0, 255, 0), 1)
                pts_all.append(pts)
            pts_all = np.concatenate(pts_all)
            x0, y0 = np.maximum(pts_all.min(0) - 40, 0)
            x1, y1 = pts_all.max(0) + 40
            cv2.imwrite(str(OUT / "preview" / f"{'touching' if was_touching else 'apart'}_{name}.jpg"), img[y0:y1, x0:x1])
        print(f"{len(pick)} previews in {OUT / 'preview'} (red outline = nemesis, green = robot)")


if __name__ == "__main__":
    main()
