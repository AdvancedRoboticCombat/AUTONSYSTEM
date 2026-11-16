"""
Fine-tune the detector on a new look of Nemesis (and a new camera angle) from one video of it driving alone.

    python ZACVISIONQUEST/USETHISFORFINETUNE.py ~/new_nemesis.mp4
    python ZACVISIONQUEST/USETHISFORFINETUNE.py ~/new_nemesis.mp4 --orient rotate-180 --start 60 --end 1260
    python ZACVISIONQUEST/USETHISFORFINETUNE.py ~/new_nemesis.mp4 --build-only    # make the images, check preview.jpg, no training
    python ZACVISIONQUEST/USETHISFORFINETUNE.py ~/new_nemesis.mp4 --skip-build    # train on images already made

What to record
  About 20 minutes of the new Nemesis driving around the arena ON ITS OWN, with the camera exactly
  where it will be on fight day. Record it through the dashboard (scripts/dashboard.sh --record DIR)
  so the picture is turned the same way the dashboard turns it; a raw camera recording is upside
  down, so pass --orient to match the dashboard's Orientation setting. Drive everywhere: walls,
  corners, centre, fast, slow, spinning. Robots parked in the arena are fine (they become `robot`
  labels). People walking in are fine: those frames are skipped.

What it does
  1. the empty arena: the median of frames spread over the video (Nemesis moves, so it disappears)
  2. finds Nemesis in every sampled frame: the one detection of the current model that moves against
     the empty arena, outlined again by SAM 2; when the model misses it, the one moving blob of the
     right size on the floor. Robots the model finds in the empty arena are labelled `robot`.
  3. makes fights from it: real frames with an opponent pasted in (apart, touching, on top,
     underneath), the new Nemesis cut out and pasted next to an opponent on the empty arena,
     opponents alone, empty arenas. Opponents are cut from the TV footage and the Unity renders.
  4. mixes in old training images so the model does not forget the old Nemesis and arena
  5. fine-tunes the current best model, then scores old and new model on the new held-out part
     and on the old held-out fight

The last 15% of the video is held out (valid): no frame or cutout from it is trained on.
Writes dataset/finetune_<video>/ (images, labels, preview.jpg, frames.csv) and runs/finetune/<video>/weights/best.pt
"""
import argparse
import csv
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paste_fights import cut, cutouts, polygon  # noqa: E402

DATA = HERE / "dataset"
RUNS = HERE / "runs" / "finetune"
BEST = HERE / "runs" / "sweep" / "yolo26m-seg_1280_all_fixes" / "weights" / "chosen.pt"
SAM_WEIGHTS = HERE / "weights" / "sam2.1_b.pt"
ROBOT, NEMESIS = 0, 1
ORIENT = {"normal": lambda f: f,                                       # the dashboard's Orientation setting
          "rotate-180": lambda f: cv2.rotate(f, cv2.ROTATE_180),
          "mirror": lambda f: cv2.flip(f, 1),
          "upside-down": lambda f: cv2.flip(f, 0)}
VALID_SHARE = 0.15
# Fine-tuning: low learning rate, optimizer pinned (see sweep.py), augmentation pushed toward a camera that moved
TRAIN = dict(optimizer="MuSGD", lr0=0.002, momentum=0.9, warmup_epochs=1, warmup_bias_lr=0.0, close_mosaic=3, patience=0,
             workers=12, plots=False, seed=0, verbose=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.6, degrees=10.0,
             translate=0.2, scale=0.7, perspective=0.0005, flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0)


# ---------------------------------------------------------------- 1. frames and the empty arena

def grab_frames(video, out, every, start, end, orient):
    """Save one frame every `every` seconds between start and end. Returns [(frame number, seconds, file)]."""
    cap = cv2.VideoCapture(str(video))
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        sys.exit(f"cannot read {video}")
    first, last = int(start * fps), min(int(end * fps) if end else total, total)
    step = max(int(round(every * fps)), 1)
    wanted, frames, n = set(range(first, last, step)), [], 0
    turn = ORIENT[orient]
    while n < last:
        ok = cap.grab()
        if not ok:
            break
        if n in wanted:
            ok, img = cap.retrieve()
            if ok:
                f = out / f"{n:06d}.jpg"
                cv2.imwrite(str(f), turn(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames.append((n, n / fps, f))
        n += 1
    print(f"{len(frames)} frames from {video.name} ({first / fps:.0f} s to {n / fps:.0f} s, one every {every} s)", flush=True)
    return frames


def empty_arena(frames, n=150):
    pick = [frames[i][2] for i in np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    stack = np.stack([cv2.imread(str(f)) for f in pick])
    return np.median(stack, axis=0).astype(np.uint8)


# ---------------------------------------------------------------- 2. find Nemesis

def raster(poly, shape):
    return cv2.fillPoly(np.zeros(shape, np.uint8), [np.asarray(poly, np.int32)], 1)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def detections(net, img, conf):
    r = net.predict(img, imgsz=1280, conf=conf, verbose=False)[0]
    if r.masks is None:
        return []
    return [raster(p, img.shape[:2]) for p in r.masks.xy if len(p) >= 3]


def merge(masks):
    """One mask per object: the model sometimes finds the same robot twice, once per class."""
    out = []
    for m in masks:
        for i, o in enumerate(out):
            if (m & o).sum() > 0.3 * min(m.sum(), o.sum()):
                out[i] = m | o
                break
        else:
            out.append(m)
    return out


def colours(img, mask):
    """Hue-saturation histogram of the object, to tell whether it is the same robot every time."""
    h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def outline(sam, img, mask):
    """SAM 2's outline of the object in `mask`'s box, or None when it does not agree with `mask`."""
    ys, xs = np.nonzero(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    r = sam(img, bboxes=[box], verbose=False)[0]
    if r.masks is None:
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
    if m.shape != mask.shape:
        m = cv2.resize(m, mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return m if iou(m, mask) >= 0.6 else None


def motion(img, bg, thr):
    """Pixels that differ from the empty arena, after matching overall brightness (the arena lights change)."""
    gain = np.median(bg) / max(np.median(img), 1.0)
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    d = cv2.absdiff(cv2.GaussianBlur(img, (7, 7), 0), cv2.GaussianBlur(bg, (7, 7), 0)).max(axis=2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def side(mask):
    return float(np.sqrt(mask.sum()))


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def find_nemesis(frames, bg, weights, thr):
    """{frame file: Nemesis polygon}, parked robot polygons, and a log row per frame."""
    from ultralytics import SAM, YOLO
    net, sam = YOLO(str(weights)), SAM(str(SAM_WEIGHTS))
    shape = bg.shape[:2]
    parked = [m for m in detections(net, bg, 0.3)]
    print(f"{len(parked)} parked robots found in the empty arena (labelled `robot` everywhere)", flush=True)
    found, unsure, log = {}, [], {}
    for k, (n, t, f) in enumerate(frames):
        img = cv2.imread(str(f))
        move = motion(img, bg, thr)
        moving = [m for m in merge(detections(net, img, 0.15))
                  if (m & move).sum() > 0.3 * m.sum() and not any(iou(m, p) > 0.5 for p in parked)]
        if len(moving) == 1:
            m = moving[0]
            near = cv2.dilate(m, np.ones((31, 31), np.uint8))
            nem = outline(sam, img, m | (move & near))
            if nem is not None:
                found[f] = nem
                log[f] = "detector"
                continue
            log[f] = "SAM disagreed"
        elif len(moving) > 1:
            log[f] = f"{len(moving)} moving robots"
        else:
            unsure.append(f)                                           # second chance below, once the floor is known
            log[f] = "nothing moving"
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(frames)} frames: Nemesis found in {len(found)}", flush=True)
    if len(found) < 20:
        sys.exit(f"Nemesis found in only {len(found)} frames: check --orient and that it drives alone in the video")

    # what the found frames teach: where the floor is and how big Nemesis is at each row
    pts = np.array([centroid(m) for m in found.values()], np.float32)
    sides = np.array([side(m) for m in found.values()])
    floor = cv2.convexHull(pts)
    fit = np.polyfit(pts[:, 1], sides, 1)
    for _ in range(2):
        keep = np.abs(sides / np.polyval(fit, pts[:, 1]) - 1) < 0.35
        fit = np.polyfit(pts[keep, 1], sides[keep], 1)
    size = lambda y: max(float(np.polyval(fit, y)), 0.5 * sides.min())  # noqa: E731

    for f in unsure:                                                   # the model missed it: one moving blob of Nemesis's size on the floor
        img = cv2.imread(str(f))
        count, lab, stats, cents = cv2.connectedComponentsWithStats(motion(img, bg, thr))
        cands = [i for i in range(1, count)
                 if cv2.pointPolygonTest(floor, tuple(map(float, cents[i])), True) > -size(cents[i][1])
                 and abs(np.sqrt(stats[i, cv2.CC_STAT_AREA]) / size(cents[i][1]) - 1) < 0.35]
        if len(cands) == 1:
            nem = outline(sam, img, (lab == cands[0]).astype(np.uint8))
            if nem is not None:
                found[f] = nem
                log[f] = "moving blob"

    # last checks: the outline is the size Nemesis has there, nothing else robot-sized moves on the floor,
    # and it has Nemesis's colours (a different robot driving in must not be labelled Nemesis)
    hists = {f: colours(cv2.imread(str(f)), m) for f, m in found.items()}
    typical = np.median(np.stack(list(hists.values())), axis=0).astype(np.float32)
    out = {}
    for f, m in found.items():
        y = centroid(m)[1]
        p = polygon(m)
        if p is None or abs(side(m) / size(y) - 1) > 0.35:
            log[f] = "wrong size"
            continue
        if cv2.compareHist(hists[f], typical, cv2.HISTCMP_CORREL) < 0.5:
            log[f] = "colours unlike Nemesis"
            continue
        count, lab, stats, cents = cv2.connectedComponentsWithStats(motion(cv2.imread(str(f)), bg, thr) & (1 - cv2.dilate(m, np.ones((41, 41), np.uint8))))
        if any(cv2.pointPolygonTest(floor, tuple(map(float, cents[i])), True) > 0
               and 0.5 < np.sqrt(stats[i, cv2.CC_STAT_AREA]) / size(cents[i][1]) < 2.5      # robot-sized, not a patch of light
               for i in range(1, count)):
            log[f] = "something else moving on the floor"
            continue
        out[f] = p.astype(np.float32)
    parked_polys = [p.astype(np.float32) for p in (polygon(m) for m in parked) if p is not None]
    return out, parked_polys, log, (floor, fit, shape)


# ---------------------------------------------------------------- 3. make fights

def place(canvas, crop, mask, cx, cy, sd, rng, block=None):
    """Paste a cutout (square root of its area = sd) centred on (cx, cy), never over `block`.
    Returns (visible mask, pixels it would cover without `block`)."""
    H, W = canvas.shape[:2]
    s = sd / max(np.sqrt((mask > 0).sum()), 1)
    w, h = max(int(crop.shape[1] * s), 2), max(int(crop.shape[0] * s), 2)
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        crop, mask = cv2.flip(crop, 1), cv2.flip(mask, 1)
    if rng.random() < 0.5:                                             # any heading: the camera sees robots from above
        k = int(rng.integers(1, 4))
        crop, mask = np.rot90(crop, k).copy(), np.rot90(mask, k).copy()
        h, w = crop.shape[:2]
    blur = rng.uniform(0, 1.5)
    if blur > 0.3:
        crop = cv2.GaussianBlur(crop, (0, 0), blur)
    crop = np.clip(crop.astype(np.float32) * rng.uniform(0.7, 1.15), 0, 255)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    full = np.zeros((H, W), np.uint8)
    if ax1 <= ax0 or ay1 <= ay0:
        return full, 0
    sub = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
    alpha = cv2.GaussianBlur(mask[sub].astype(np.float32) / 255, (0, 0), 0.8)
    whole = int((alpha > 0.5).sum())
    if block is not None:
        alpha = alpha * (1 - block[ay0:ay1, ax0:ax1])
    region = canvas[ay0:ay1, ax0:ax1]
    canvas[ay0:ay1, ax0:ax1] = (crop[sub] * alpha[..., None] + region * (1 - alpha[..., None])).astype(np.uint8)
    full[ay0:ay1, ax0:ax1] = alpha > 0.5
    return full, whole


def setup(*state):
    global S
    S = dict(zip(("bg", "nem", "bots", "parked", "floor", "fit", "out"), state))


def size_at(y):
    return max(float(np.polyval(S["fit"], y)), 15.0)


def spot(rng):
    """A point on the floor Nemesis drove on."""
    floor = S["floor"]
    x0, y0, w, h = cv2.boundingRect(floor)
    for _ in range(200):
        x, y = rng.uniform(x0, x0 + w), rng.uniform(y0, y0 + h)
        if cv2.pointPolygonTest(floor, (float(x), float(y)), False) >= 0:
            return x, y
    return x0 + w / 2, y0 + h / 2


def make(job):
    """One image. kind: real_frame (Nemesis alone), real_plus_enemy, two_on_empty, enemy_alone, empty."""
    i, kind, seed = job
    rng = np.random.default_rng(seed)
    nem_file, nem_poly = S["nem"][rng.integers(len(S["nem"]))]
    shape = S["bg"].shape[:2]
    objects = []                                                       # [cls, visible mask, whole area, parked]
    for p in S["parked"]:
        m = raster(p, shape)
        objects.append([ROBOT, m, m.sum(), True])

    if kind in ("real_frame", "real_plus_enemy"):
        canvas = cv2.imread(str(nem_file))
        nem = raster(nem_poly, shape)
        nem_whole = nem.sum()
        nx, ny = centroid(nem)
    else:
        canvas = S["bg"].copy()
        nem = None
        if kind == "two_on_empty":
            nx, ny = spot(rng)
            nem, nem_whole = place(canvas, *cut((nem_file, nem_poly)), nx, ny, size_at(ny) * rng.uniform(0.95, 1.05), rng)
    canvas = np.clip(canvas.astype(np.float32) * rng.uniform(0.8, 1.15), 0, 255).astype(np.uint8)
    if nem is not None:
        for o in objects:
            o[1] = o[1] & (1 - nem)
        objects.append([NEMESIS, nem, nem_whole, False])

    if kind in ("real_plus_enemy", "two_on_empty", "enemy_alone"):
        if nem is not None and rng.random() < 0.5:                     # touching Nemesis
            reach = 0.55 * size_at(ny) * rng.uniform(1.6, 2.2)
            a = rng.uniform(0, 2 * np.pi)
            ex, ey = nx + reach * np.cos(a), ny + reach * np.sin(a)
        else:
            ex, ey = spot(rng)
        under = nem is not None and rng.random() < 0.4                  # the opponent goes under Nemesis
        enemy, enemy_whole = place(canvas, *cut(S["bots"][rng.integers(len(S["bots"]))]), ex, ey,
                                   size_at(ey) * rng.uniform(0.7, 1.4), rng, block=nem if under else None)
        for o in objects:
            o[1] = o[1] & (1 - enemy)
        objects.append([ROBOT, enemy, enemy_whole, False])

    rows = []
    for c, m, whole, is_parked in objects:
        p = polygon(m)
        seen = cv2.contourArea(p) / max(whole, 1) if p is not None else 0
        if is_parked:
            if seen > 0.3:                                             # a parked robot mostly covered loses its label
                rows.append((c, p))
        elif seen < 0.5:
            return None                                                # Nemesis or the opponent mostly hidden or off the picture
        else:
            rows.append((c, p))
    H, W = shape
    name = f"{kind}_{i:05d}"
    cv2.imwrite(str(S["out"] / "images" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(S["out"] / "labels" / f"{name}.txt", "w") as fh:
        for c, p in rows:
            fh.write(f"{c} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in p) + "\n")
    return name


def opponents(limit_sim=2000):
    """Robot cutouts: the TV footage in train, and robots from the Unity renders."""
    _, bots, _, _ = cutouts()
    sim = []
    for f in sorted((DATA / "sim" / "labels").glob("*.txt")):
        for line in f.read_text().splitlines():
            v = line.split()
            if v and int(v[0]) == ROBOT:
                p = np.array(v[1:], np.float32).reshape(-1, 2) * [1920, 1440]
                x0, y0, x1, y1 = *p.min(0), *p.max(0)
                if x0 > 4 and y0 > 4 and x1 < 1908 and y1 < 1428 and min(x1 - x0, y1 - y0) >= 40:
                    sim.append((DATA / "sim" / "images" / f"{f.stem}.jpg", p))
    random.Random(0).shuffle(sim)
    return bots + sim[:limit_sim]


def build(frames, bg, nem, parked, scene, out, n_fakes, workers):
    floor, fit, _ = scene
    cut_at = frames[int(len(frames) * (1 - VALID_SHARE))][0]
    parts = {"train": [(f, p) for f, p in nem.items() if int(f.stem) < cut_at],
             "valid": [(f, p) for f, p in nem.items() if int(f.stem) >= cut_at]}
    bots = opponents()
    print(f"Nemesis in {len(parts['train'])} train frames and {len(parts['valid'])} held-out frames; {len(bots)} opponent cutouts", flush=True)
    rng = np.random.default_rng(0)
    for part, items in parts.items():
        if not items:
            sys.exit("no Nemesis frames in the held-out part: use a longer video")
        d = out / part
        shutil.rmtree(d, ignore_errors=True)
        for sub in ("images", "labels"):
            (d / sub).mkdir(parents=True)
        share = 1 if part == "train" else VALID_SHARE
        setup(bg, items, bots, parked, floor, fit, d)
        for i, item in enumerate(items):                               # every real frame once, as it is
            S["nem"] = [item]
            make((i, "real_frame", i))
        kinds = list(rng.choice(["real_plus_enemy", "two_on_empty", "enemy_alone"], int(n_fakes * share), p=[0.6, 0.3, 0.1]))
        kinds += ["empty"] * int(150 * share)
        jobs = [(len(items) + i, k, int(rng.integers(1 << 31))) for i, k in enumerate(kinds)]
        with Pool(workers, initializer=setup, initargs=(bg, items, bots, parked, floor, fit, d)) as pool:
            made = [r for r in pool.imap_unordered(make, jobs, chunksize=8) if r]
        print(f"  {part}: {len(list((d / 'images').glob('*.jpg')))} images ({len(jobs) - len(made)} fakes dropped: a robot mostly hidden)", flush=True)


def preview(out, n=24):
    files = sorted((out / "train" / "images").glob("*.jpg"))
    rng = random.Random(1)
    tiles = []
    for f in rng.sample(files, min(n, len(files))):
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        for line in (out / "train" / "labels" / f"{f.stem}.txt").read_text().splitlines():
            v = line.split()
            pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255) if v[0] == "1" else (0, 255, 0), max(W // 400, 2))
        tile = cv2.resize(img, (480, int(480 * H / W)))
        cv2.putText(tile, f.stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / "preview.jpg"), np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]))
    print(f"look at {out / 'preview.jpg'}: red = nemesis, green = robot", flush=True)


# ---------------------------------------------------------------- 4 and 5. train and score

def identity_check(out, weights, n=120):
    """What the current model calls the robot this script labelled Nemesis. Mostly `robot` means: is it really Nemesis?"""
    from ultralytics import YOLO
    net = YOLO(str(weights))
    files = sorted((out / "train" / "images").glob("real_frame_*.jpg"))[:n]
    votes = {NEMESIS: 0, ROBOT: 0, None: 0}
    for f in files:
        img = cv2.imread(str(f))
        truth = [l.split() for l in (out / "train" / "labels" / f"{f.stem}.txt").read_text().splitlines() if l.startswith("1 ")]
        t = raster(np.array(truth[0][1:], np.float32).reshape(-1, 2) * img.shape[1::-1], img.shape[:2])
        r = net.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
        best, cls = 0.3, None
        if r.masks is not None:
            for poly, c in zip(r.masks.xy, r.boxes.cls.tolist()):
                if len(poly) >= 3 and iou(raster(poly, img.shape[:2]), t) > best:
                    best, cls = iou(raster(poly, img.shape[:2]), t), int(c)
        votes[cls] += 1
    total = max(sum(votes.values()), 1)
    print(f"the current model calls the robot labelled Nemesis here: nemesis {votes[NEMESIS] / total:.0%}, "
          f"robot {votes[ROBOT] / total:.0%}, nothing {votes[None] / total:.0%}", flush=True)
    if votes[ROBOT] > votes[NEMESIS]:
        print("WARNING: the current model thinks this is an OPPONENT, not Nemesis. Fine if Nemesis really looks new;\n"
              "         if any other robot drove in this video, fine-tuning will teach the model to call opponents Nemesis.", flush=True)


def write_yaml(out, n_replay):
    old = []
    for d in ("train", "paste", "sim_asrendered"):                     # what the current best model was trained on
        old += sorted((DATA / d / "images").glob("*.jpg"))
    random.Random(0).shuffle(old)
    (out / "replay.txt").write_text("\n".join(str(f) for f in old[:n_replay]) + "\n")
    y = out / "data.yaml"
    y.write_text(f"# new Nemesis video + {n_replay} old training images; valid = the held-out end of the video\n"
                 f"path: {out}\ntrain:\n  - train/images\n  - replay.txt\nval: valid/images\nnames:\n  0: robot\n  1: nemesis\n")
    return y


def train(data, weights, name, epochs, batch):
    import torch
    from ultralytics import YOLO
    while True:
        try:
            YOLO(str(weights)).train(data=str(data), imgsz=1280, epochs=epochs, batch=batch, project=str(RUNS), name=name,
                                     exist_ok=True, **TRAIN)
            return RUNS / name / "weights" / "best.pt"
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch //= 2
            print(f"out of GPU memory: retrying with batch {batch}", flush=True)
            if batch < 2:
                raise


def report(data, old, new):
    from ultralytics import YOLO
    import sweep
    scores = {}
    print("\n                         new video (held out)            old held-out fight")
    print("weights                  nemesis mAP  robot mAP          fight  nemesis  opponent  mixed up")
    for label, w in (("before", old), ("after", new)):
        m = YOLO(str(w)).val(data=str(data), imgsz=1280, batch=8, plots=False, verbose=False, split="val")
        ap = m.seg.ap                                                  # mask mAP50-95 per class present
        per = dict(zip(m.ap_class_index.tolist(), ap.tolist()))
        f = sweep.fight(w, 1280)
        scores[label] = f
        print(f"{label:24s} {per.get(NEMESIS, float('nan')):11.3f}  {per.get(ROBOT, float('nan')):9.3f}          "
              f"{f['fight']:.3f}  {f['nemesis']:7.3f}  {f['opponent']:8.3f}  {f['mixed_up']:8.3f}")
    print(f"\nnew weights: {new}")
    if scores["after"]["mixed_up"] > scores["before"]["mixed_up"] + 0.05:
        print("WARNING: DO NOT USE THESE WEIGHTS. The model now mixes up Nemesis and the opponent on the old fight:\n"
              "         the robot labelled Nemesis in the video is probably not Nemesis (look at preview.jpg).")
    elif scores["after"]["fight"] < scores["before"]["fight"] - 0.05:
        print("WARNING: the old fight got worse: the model forgot the old arena. Train again with a larger --replay.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--orient", choices=ORIENT, default="normal", help="turn the picture like the dashboard's Orientation setting")
    ap.add_argument("--start", type=float, default=0, help="seconds to skip at the start")
    ap.add_argument("--end", type=float, default=0, help="stop at this second (0 = the end)")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--fakes", type=int, default=3000, help="made-up fight images")
    ap.add_argument("--replay", type=int, default=4000, help="old training images mixed in")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", type=Path, default=BEST, help="model to start from")
    ap.add_argument("--motion", type=float, default=35, help="pixel change that counts as movement (0-255)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--build-only", action="store_true", help="make the images and stop")
    ap.add_argument("--skip-build", action="store_true", help="train on the images already made")
    args = ap.parse_args()

    name = args.video.stem
    out = DATA / f"finetune_{name}"
    if not args.skip_build:
        shutil.rmtree(out, ignore_errors=True)
        (out / "frames").mkdir(parents=True)
        frames = grab_frames(args.video.expanduser(), out / "frames", args.every, args.start, args.end, args.orient)
        bg = empty_arena(frames)
        cv2.imwrite(str(out / "empty_arena.jpg"), bg)
        nem, parked, log, scene = find_nemesis(frames, bg, args.weights, args.motion)
        with open(out / "frames.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "seconds", "result"])
            for n, t, f in frames:
                w.writerow([n, f"{t:.2f}", "nemesis (" + log[f] + ")" if f in nem else "skipped: " + log[f]])
        print(f"Nemesis outlined in {len(nem)} of {len(frames)} frames (why the rest were skipped: frames.csv)", flush=True)
        build(frames, bg, nem, parked, scene, out, args.fakes, args.workers)
        preview(out)
        if args.build_only:
            return
    identity_check(out, args.weights)
    data = write_yaml(out, args.replay)
    new = train(data, args.weights, name, args.epochs, args.batch)
    report(data, args.weights, new)


if __name__ == "__main__":
    main()
