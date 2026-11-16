"""
Turn the Unity renders into YOLO segmentation labels.

    python ZACVISIONQUEST/unity_to_yolo.py                        # every clip under Datasets/ and unity_renders/
    python ZACVISIONQUEST/unity_to_yolo.py --renders DIR [DIR..]  # these folders instead (searched for run_*/sim*.mp4)
    python ZACVISIONQUEST/unity_to_yolo.py --preview 60           # also save 60 labelled close-ups to look at

Writes
    dataset/sim/             the Unity robots pasted onto our own empty arena. Training uses this.
    dataset/sim_asrendered/  the same frames exactly as Unity rendered them. Not for training (see below).
    dataset/sim/report.csv   one row per clip: frames, how many were labelled, nemesis and robot counts

Renders are for training only; they never go into valid or test.

Why the robots are pasted onto another background

    Unity renders the robots over a still photo of the arena, and that photo is a frame
    of the held-out Nemesis fight. Training on the renders as they are would show the
    model the exact background of every valid and test image. So the robots are cut out
    and pasted onto an empty arena made from NEM_08, a session that is in train anyway.
    The six robots parked in that session stay in the picture, with their labels.

How the labels are made

    The Unity export has video and a yaw angle per frame, but no boxes or masks, so the
    labels are rebuilt from the picture:

  1. The empty Unity arena is the per-pixel median over all the clips. Whatever differs
     from it in a frame is robot. That gives the exact outline of the robots, but not
     which robot is which, and two touching robots come out as one shape.
  2. Apart: Nemesis is the outline with red and gold paint (class 1), the other is
     class 0. These frames are also shown to SAM 2 so it knows what each robot looks like.
  3. Touching: SAM 2 says which pixels belong to which robot and the joint outline is
     divided along that boundary. The paint must agree about which half is Nemesis.
  4. A frame is skipped when the result is not credible: the paint does not single out
     one robot, an outline is the wrong size for a robot standing at that spot, or SAM
     gave the same pixels to both robots. That mostly happens when one robot is on top
     of the other.

The paint rule assumes the opponent is not red or gold. The Unity project also has red
robots (Malice, Jackpot, Tombstone); clips with those would mostly be skipped, and the
report shows it. Exporting instance masks from Unity itself would be exact and would
replace all of this.
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RENDERS = [HERE / "Zac_vis_final_segmented_dataset" / "Datasets", HERE / "unity_renders"]
DATASET = HERE / "dataset"
OUT = DATASET / "sim"
OUT_RAW = DATASET / "sim_asrendered"
SAM_WEIGHTS = HERE / "weights" / "sam2.1_b.pt"
ROBOT, NEMESIS = 0, 1

OWN_ARENA = "NEM_08"        # the training session whose empty arena the robots are pasted onto
DIFF = 16                   # grey levels a pixel must differ from the empty arena to count as robot
MIN_AREA = 250              # px; smaller blobs are compression noise
NEMESIS_RED = 0.03          # share of a robot that is red / gold paint for it to be Nemesis
LONG_CLIP = 2000            # frames; longer clips are sampled more thinly
HORIZON = 300               # image row where a robot on the floor would shrink to nothing (Unity camera)
SIZE_ANY = (0.08, 0.30)     # size (see size()) of any robot; the ones seen so far are 0.15 to 0.17
SIZE_SAME = (0.70, 1.40)    # one robot's size against its own median so far


def find_clips(folders):
    clips = []
    for folder in folders:
        clips += sorted(Path(folder).glob("**/run_*/sim*.mp4"))
    return clips


def clip_name(video):
    return re.sub(r"[^A-Za-z0-9\-]+", "_", f"{video.parts[-3]}_{video.parts[-2]}_{video.stem}")


def empty_arena(videos):
    """The Unity arena with no robots in it: the per-pixel median over all the clips."""
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


def own_arena(session=OWN_ARENA, n=48):
    """Our own empty arena, from a training session: (picture, masks of the robots parked in it).

    The median over the session's frames. Nemesis drives around in them, so in each frame
    the pixels it covers are set to black or white in turn, which leaves the median alone.
    """
    files = sorted((DATASET / "train" / "images").glob(f"{session}__frame_*.jpg"))
    if not files:
        raise SystemExit(f"no {session} frames in {DATASET / 'train'}: run build_dataset.py first")
    files = [files[i] for i in np.linspace(0, len(files) - 1, n).astype(int)]
    frames, parked = [], []
    covered = None
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        nemesis, robots = np.zeros((h, w), np.uint8), []
        for line in (DATASET / "train" / "labels" / f"{f.stem}.txt").read_text().splitlines():
            vals = line.split()
            pts = (np.array(vals[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
            if int(vals[0]) == NEMESIS:
                cv2.fillPoly(nemesis, [pts], 1)
            else:
                robots.append(cv2.fillPoly(np.zeros((h, w), np.uint8), [pts], 1))
        nemesis = cv2.dilate(nemesis, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
        img[nemesis] = 0 if i % 2 else 255
        covered = nemesis.astype(np.uint8) if covered is None else covered + nemesis
        frames.append(img)
        parked.append(robots)
    if (covered > n // 3).any():
        raise SystemExit(f"Nemesis sits still for too much of {session} to make an empty arena from it")
    arena = np.median(np.stack(frames), axis=0).astype(np.uint8)
    # the parked robots must really be parked: the same pixels at both ends of the session
    first, last = np.any(parked[0], axis=0), np.any(parked[-1], axis=0)
    same = (first & last).sum() / max((first | last).sum(), 1)
    if same < 0.8:
        raise SystemExit(f"the robots parked in {session} move (overlap {same:.2f}); pick another session for OWN_ARENA")
    return arena, parked[n // 2]


def red_share(frame, mask):
    """Share of a mask that is Nemesis's paint: red trim or gold sunburst."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 8) | (h > 170)) & (s > 120) & (v > 70)
    gold = (h > 15) & (h < 35) & (s > 70) & (v > 150)
    return float(((red | gold) & (mask > 0)).sum()) / max(int(mask.sum()), 1)


def brightness(frame, mask):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[mask > 0].mean())


def differs(frame, arena_blur):
    """1 where the frame is not the empty arena."""
    d = np.abs(cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16) - arena_blur).max(axis=2)
    return (d > DIFF).astype(np.uint8)


def filled(mask, min_area=0):
    """The outer contours of a mask filled in, largest first, as separate masks."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) >= min_area:
            out.append(cv2.drawContours(np.zeros(mask.shape, np.uint8), [c], -1, 1, cv2.FILLED))
    return out


def robot_blobs(frame, arena_blur, close=13):
    """Filled masks of everything that is not arena, largest first. These become the outlines."""
    m = cv2.morphologyEx(differs(frame, arena_blur), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    return filled(m, MIN_AREA)


def robot_pixels(frame, arena_blur):
    """The pixels to cut out and paste: like robot_blobs, but without bridging the gaps between forks."""
    m = cv2.morphologyEx(differs(frame, arena_blur), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(np.zeros(m.shape, np.uint8), contours, -1, 1, cv2.FILLED)


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


def row_of(mask):
    rows = mask.sum(axis=1, dtype=np.float64)
    return float((np.arange(len(rows)) * rows).sum() / max(rows.sum(), 1.0))


def size(mask):
    """A robot's size with the perspective taken out: about the same number wherever it stands.

    A robot far from the camera is small in the picture. Its width in pixels grows in
    proportion to how far below the horizon row it is, so dividing by that gives a
    number that only depends on the robot (and a little on which way it is turned).
    """
    return float(np.sqrt(mask.sum())) / max(row_of(mask) - HORIZON, 40.0)


def at_edge(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def divide(blob, sam, usual):
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
        if not 0.71 < size(mine[k]) / usual[k] < 1.34:                # half to 1.8 times the robot's usual area
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
        self.sizes = [[], []]    # each robot's size in the frames where they were apart and fully in view
        self.shown = 0           # times the tracker has been shown the robots apart
        self.why = {}            # reasons frames were skipped, counted

    def skip(self, reason):
        self.why[reason] = self.why.get(reason, 0) + 1
        return None

    def usual(self, k):
        return float(np.median(self.sizes[k][-40:]))

    def credible(self, mask, k=None):
        """True when a mask is the size of a robot, and of robot k when k is given."""
        s, cut = size(mask), at_edge(mask)
        if s > SIZE_ANY[1] or (s < SIZE_ANY[0] and not cut):
            return False
        if k is not None and len(self.sizes[k]) >= 5:
            r = s / self.usual(k)
            if r > SIZE_SAME[1] or r < (0.5 if cut else SIZE_SAME[0]):
                return False
        return True

    def apart(self, frame, blobs):
        who = identify(frame, blobs)
        if who is None:
            return self.skip("paint does not single out one robot")
        ordered = [blobs[who[0]], blobs[who[1]]]
        if not all(self.credible(ordered[k], k) for k in range(2)):
            return self.skip("outline is the wrong size for a robot")
        for k in range(2):
            if not at_edge(ordered[k]):
                self.sizes[k].append(size(ordered[k]))
        if self.shown % 3 == 0:                                      # keep the tracker's memory of each robot fresh
            self.sam(source=frame, bboxes=[box_of(m) for m in ordered], obj_ids=[0, 1], update_memory=True)
        self.shown += 1
        return [(NEMESIS, ordered[0]), (ROBOT, ordered[1])]

    def label(self, frame, robots_in_clip, lone_is_nemesis):
        """([(class, mask), ...], touching) for one frame, or None when it cannot be labelled cleanly."""
        blobs = robot_blobs(frame, self.arena_blur)
        if len(blobs) > robots_in_clip:                              # a fork or wheel came out as its own blob: close harder
            blobs = robot_blobs(frame, self.arena_blur, close=27)
        if robots_in_clip == 1:
            if len(blobs) != 1 or not self.credible(blobs[0]):
                return self.skip("not one robot-sized outline")
            return [(NEMESIS if lone_is_nemesis else ROBOT, blobs[0])], False
        if len(blobs) == 2:
            out = self.apart(frame, blobs)
            return (out, False) if out else None
        if len(blobs) != 1:
            return self.skip("not one or two outlines")
        if min(len(s) for s in self.sizes) < 3:
            return self.skip("touching before the robots were seen apart")
        # one outline. Nearly touching robots are only joined by the closing step: try without it
        loose = robot_blobs(frame, self.arena_blur, close=3)
        if len(loose) == 2:
            out = self.apart(frame, loose)
            if out:
                return out, False
        res = self.sam(source=frame)[0]
        if res.masks is None or len(res.masks.data) != 2:
            return self.skip("touching: tracker lost a robot")
        masks = res.masks.data.cpu().numpy().astype(np.uint8)
        sam = [None, None]
        for j, obj in enumerate(res.boxes.cls.tolist()):
            sam[int(obj)] = masks[j]
        if sam[0] is None or sam[1] is None:
            return self.skip("touching: tracker lost a robot")
        parts = divide(blobs[0], sam, [self.usual(0), self.usual(1)])
        if parts is None:
            return self.skip("touching: division not credible")
        if red_share(frame, parts[0]) < NEMESIS_RED or red_share(frame, parts[1]) > NEMESIS_RED:
            return self.skip("touching: paint disagrees with the tracker")
        return [(NEMESIS, parts[0]), (ROBOT, parts[1])], True


def look_at_clip(cap, n_frames, arena_blur):
    """(robots in the clip, paint share of the robot when it is alone) from a spread of frames."""
    counts, shares = [], []
    for f in np.linspace(0, n_frames - 1, min(48, n_frames)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        blobs = robot_blobs(img, arena_blur)
        counts.append(len(blobs))
        if len(blobs) == 1:
            shares.append(red_share(img, blobs[0]))
    return min(2, max(counts, default=0)), float(np.median(shares)) if shares else 0.0


def paste(frame, objects, pixels, arena, parked):
    """The Unity robots of one frame on our own arena: (picture, [(class, mask), ...]).

    None when a pasted robot hides most of a parked one, which could not be labelled honestly.
    """
    frame = cv2.flip(frame, 1)                                       # Unity's view is the mirror image of our dataset's
    objects = [(cls, cv2.flip(m, 1)) for cls, m in objects]
    union = np.clip(np.sum([m for _, m in objects], axis=0), 0, 1).astype(np.uint8)
    cut = cv2.flip(pixels, 1) & cv2.dilate(union, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 0.8)[..., None]
    out = (frame * alpha + arena * (1 - alpha)).round().astype(np.uint8)
    for m in parked:
        seen = m
        if (m & cut).any():
            seen = largest_part(m & (1 - cut))
            if seen.sum() < 0.5 * m.sum():
                return None
        objects.append((ROBOT, seen))
    return out, objects


def outlines(objects):
    """[(class, polygon), ...] for a list of (class, mask), or None when a mask has no usable outline."""
    polys = [(cls, polygon(m)) for cls, m in objects]
    return None if any(p is None for _, p in polys) else polys


def write(folder, name, img, polys):
    h, w = img.shape[:2]
    cv2.imwrite(str(folder / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    with open(folder / "labels" / f"{name}.txt", "w") as fh:
        for cls, p in polys:
            fh.write(f"{cls} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p) + "\n")


def save_previews(picks):
    """Close-ups of the Unity robots in the pasted images, outlines drawn (red = nemesis, green = robot)."""
    for name, was_touching, n_unity in picks:
        img = cv2.imread(str(OUT / "images" / f"{name}.jpg"))
        h, w = img.shape[:2]
        pts_all = []
        for i, line in enumerate((OUT / "labels" / f"{name}.txt").read_text().splitlines()):
            vals = line.split()
            pts = (np.array(vals[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255) if vals[0] == "1" else (0, 255, 0), 1)
            if i < n_unity:
                pts_all.append(pts)
        pts_all = np.concatenate(pts_all)
        x0, y0 = np.maximum(pts_all.min(0) - 40, 0)
        x1, y1 = pts_all.max(0) + 40
        cv2.imwrite(str(OUT / "preview" / f"{'touching' if was_touching else 'apart'}_{name}.jpg"), img[y0:y1, x0:x1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", nargs="+", default=[str(r) for r in RENDERS], help="folders holding Unity output")
    ap.add_argument("--every", type=int, default=4, help="short clips: keep one frame in this many (60 fps; default 4)")
    ap.add_argument("--every-long", type=int, default=12, help=f"clips over {LONG_CLIP} frames: one in this many (default 12)")
    ap.add_argument("--preview", type=int, default=0, help="save this many labelled close-ups to dataset/sim/preview")
    args = ap.parse_args()

    videos = find_clips(args.renders)
    print(f"{len(videos)} clips")
    arena_blur = cv2.GaussianBlur(empty_arena(videos), (3, 3), 0).astype(np.int16)
    ours, parked = own_arena()
    print(f"pasting onto the empty arena of {OWN_ARENA}, which has {len(parked)} parked robots")
    for folder in (OUT, OUT_RAW):
        shutil.rmtree(folder, ignore_errors=True)
        for sub in ("images", "labels"):
            (folder / sub).mkdir(parents=True)
    if args.preview:
        (OUT / "preview").mkdir()
    SAM_WEIGHTS.parent.mkdir(exist_ok=True)

    report, previews, fit, why_all = [], [], [], {}
    print(f"{'clip':46s} {'frames':>6s} {'robots':>6s} {'looked':>6s} {'kept':>5s} {'touching':>8s} {'skipped':>7s} {'nemesis':>7s} {'robot':>6s}")
    for v in videos:
        clip = clip_name(v)
        cap = cv2.VideoCapture(str(v))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        every = args.every_long if n_frames > LONG_CLIP else args.every
        robots_in_clip, lone_share = look_at_clip(cap, n_frames, arena_blur)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        labeller = Labeller(arena_blur)
        row = dict(clip=clip, frames=n_frames, robots=robots_in_clip, looked=0, kept=0, touching=0, skipped=0, nemesis=0,
                   robot=0, parked_hidden=0, nemesis_not_darker=0,
                   lone_robot="" if robots_in_clip != 1 else f"{'nemesis' if lone_share > NEMESIS_RED else 'not nemesis'} "
                                                             f"(paint share {lone_share:.3f})")
        for f in range(n_frames):
            if not cap.grab():
                break
            if f % every:
                continue
            ok, img = cap.retrieve()
            row["looked"] += 1
            out = labeller.label(img, robots_in_clip, lone_share > NEMESIS_RED) if ok else None
            if not out:
                row["skipped"] += 1
                continue
            objects, touching = out
            name = f"unity_{clip}_f{f:05d}"
            pasted = paste(img, objects, robot_pixels(img, arena_blur), ours, parked)
            if pasted is None:
                row["skipped"] += 1
                row["parked_hidden"] += 1
                continue
            polys, polys_pasted = outlines(objects), outlines(pasted[1])
            if polys is None or polys_pasted is None:
                row["skipped"] += 1
                continue
            write(OUT_RAW, name, img, polys)
            write(OUT, name, pasted[0], polys_pasted)
            row["kept"] += 1
            row["touching"] += touching
            for cls, m in objects:
                row["nemesis" if cls == NEMESIS else "robot"] += 1
                if not touching and not at_edge(m):
                    fit.append((row_of(m), np.sqrt(m.sum())))
            if len(objects) == 2 and brightness(img, objects[0][1]) >= brightness(img, objects[1][1]):
                row["nemesis_not_darker"] += 1                       # second opinion on who is who: Nemesis is the black one
            previews.append((name, touching, len(objects)))
        cap.release()
        for reason, n in labeller.why.items():
            why_all[reason] = why_all.get(reason, 0) + n
        del labeller
        report.append(row)
        print(f"{clip:46s} {n_frames:6d} {robots_in_clip:6d} {row['looked']:6d} {row['kept']:5d} {row['touching']:8d} "
              f"{row['skipped']:7d} {row['nemesis']:7d} {row['robot']:6d}", flush=True)

    with open(OUT / "report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    total = {k: sum(r[k] for r in report) for k in ("looked", "kept", "touching", "skipped", "nemesis", "robot",
                                                    "parked_hidden", "nemesis_not_darker")}
    print(f"\n{total['kept']} images written ({total['touching']} with the robots touching), {total['skipped']} skipped of "
          f"{total['looked']} looked at")
    print(f"Unity robots labelled: nemesis {total['nemesis']}, robot {total['robot']}; "
          f"plus {len(parked)} parked robots in every image")
    print("skipped because: " + "; ".join(f"{n} {reason}" for reason, n in sorted(why_all.items(), key=lambda x: -x[1]))
          + f"; {total['parked_hidden']} a parked robot was hidden")

    # the checks: is Nemesis counted properly?
    two = [r for r in report if r["robots"] == 2]
    print("\nchecks")
    bad = [r["clip"] for r in two if not r["nemesis"] == r["robot"] == r["kept"]]
    print(f"  two-robot clips: {len(two)}; every image has exactly one nemesis and one other robot: "
          f"{'yes' if not bad else 'NO: ' + ', '.join(bad)}")
    print(f"  second opinion (Nemesis is the darker robot) disagrees with the paint in "
          f"{sum(r['nemesis_not_darker'] for r in two)} of {sum(r['kept'] for r in two)} images")
    for r in report:
        if r["robots"] == 1:
            print(f"  one-robot clip {r['clip']}: the robot is {r['lone_robot']}")
        if r["looked"] and r["kept"] < 0.25 * r["looked"]:
            print(f"  LOOK AT {r['clip']}: only {r['kept']} of {r['looked']} frames could be labelled")
    if len(fit) > 50:
        k, b = np.polyfit(*np.array(fit).T, 1)
        print(f"  robots shrink to nothing at row {-b / k:.0f} (this script assumes {HORIZON})"
              + ("" if abs(-b / k - HORIZON) < 60 else "  <- the camera has changed: update HORIZON"))
    print(f"wrote {OUT} and {OUT_RAW}")

    if args.preview:
        rng = np.random.default_rng(0)
        touch = [p for p in previews if p[1]]
        apart = [p for p in previews if not p[1]]
        pick = [touch[i] for i in rng.permutation(len(touch))[:args.preview // 2]]
        pick += [apart[i] for i in rng.permutation(len(apart))[:args.preview - len(pick)]]
        save_previews(pick)
        print(f"{len(pick)} previews in {OUT / 'preview'} (red outline = nemesis, green = robot)")


if __name__ == "__main__":
    main()
