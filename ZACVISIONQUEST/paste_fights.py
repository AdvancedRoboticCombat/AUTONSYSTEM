"""
Make training images of Nemesis next to another robot by cutting and pasting real pixels.

    python ZACVISIONQUEST/paste_fights.py              # 3000 fights + 300 empty arenas
    python ZACVISIONQUEST/paste_fights.py --n 500 --preview 24

Writes dataset/paste/{images,labels} and dataset/data_paste.yaml. Training only.

Every real frame of Nemesis in train shows it driving alone. This puts a real Nemesis
cutout and a real robot cutout (from the TV footage in train) on the empty fight arena:

  - each robot is resized to the size a robot has at the spot it lands on. On our
    camera that size grows in step with how far below the horizon row the robot is;
    the rule is measured from the Nemesis labels of NEM_08.
  - in 4 of 10 images the two robots touch, and the one on top hides part of the other
  - both robots get a slight random blur
  - 1 in 10 images has Nemesis alone, 1 in 10 the other robot alone
  - the empty arenas have no robot at all, so the model also sees "nothing here"

Only cutouts from train are used, so valid and test stay unseen.
"""
import argparse
import csv
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
OUT = DATA / "paste"
ARENA = DATA / "fight_arena.png"
UNITY_ARENA = Path.home() / "Downloads/Unity-3D-BattlebotSim-main/Assets/BattleBotBackground (1920x1440).png"
ROBOT, NEMESIS = 0, 1
W, H = 1920, 1440
# the arena floor in our dataset's view (the mirror image of the camera's), as a polygon
FLOOR = np.array([(520, 620), (1520, 620), (1860, 1010), (1900, 1400), (40, 1400), (60, 1100)], np.int32)


def cutouts():
    """([nemesis cutouts], [robot cutouts], horizon row, nemesis size per row). A cutout is (image file, polygon in pixels)."""
    rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] == "train"]
    nemesis, robots, fit = [], [], []
    for r in rows:
        own = r["group"] in ("NEM_08", "NEM_04")
        if not own and not r["group"].startswith(("tv_", "bilibili")):
            continue
        img = DATA / "train" / "images" / r["file"]
        lines = (DATA / "train" / "labels" / (Path(r["file"]).stem + ".txt")).read_text().splitlines()
        size = cv2.imread(str(img), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape
        w, h = size[1] * 8, size[0] * 8
        for line in lines:
            v = line.split()
            p = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.float32)
            x0, y0, x1, y1 = *p.min(0), *p.max(0)
            edge = x0 < 4 or y0 < 4 or x1 > w - 12 or y1 > h - 12
            if own and int(v[0]) == NEMESIS and not edge:
                nemesis.append((img, p))
                if r["group"] == "NEM_08":
                    fit.append((p[:, 1].mean(), np.sqrt(cv2.contourArea(p))))
            elif not own and not edge and min(x1 - x0, y1 - y0) >= 70:
                robots.append((img, p))
    k, b = np.polyfit(*np.array(fit).T, 1)
    return nemesis, robots, -b / k, k


def arena():
    if not ARENA.exists():
        cv2.imwrite(str(ARENA), cv2.flip(cv2.imread(str(UNITY_ARENA)), 1))   # mirrored into our dataset's view
    return cv2.imread(str(ARENA))


def cut(item):
    img_file, poly = item
    img = cv2.imread(str(img_file))
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    mask = cv2.fillPoly(np.zeros(img.shape[:2], np.uint8), [poly.astype(np.int32)], 255)
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def place(canvas, crop, mask, cx, cy, side, rng):
    """Paste a cutout, resized so the square root of its area is `side`, centred on (cx, cy). Returns its full-frame mask."""
    s = side / max(np.sqrt((mask > 0).sum()), 1)
    w, h = max(int(crop.shape[1] * s), 2), max(int(crop.shape[0] * s), 2)
    crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        crop = cv2.flip(crop, 1)
        mask = cv2.flip(mask, 1)
    blur = rng.uniform(0, 1.2)
    if blur > 0.3:
        crop = cv2.GaussianBlur(crop, (0, 0), blur)
    crop = np.clip(crop.astype(np.float32) * rng.uniform(0.7, 1.1), 0, 255)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if ax1 <= ax0 or ay1 <= ay0:
        return np.zeros((H, W), np.uint8)
    sub = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
    alpha = cv2.GaussianBlur(mask[sub].astype(np.float32) / 255, (0, 0), 0.8)[..., None]
    region = canvas[ay0:ay1, ax0:ax1]
    canvas[ay0:ay1, ax0:ax1] = (crop[sub] * alpha + region * (1 - alpha)).astype(np.uint8)
    full = np.zeros((H, W), np.uint8)
    full[ay0:ay1, ax0:ax1] = mask[sub] > 127
    return full


def spot(rng):
    while True:
        x, y = rng.uniform(40, W - 40), rng.uniform(620, 1400)
        if cv2.pointPolygonTest(FLOOR, (float(x), float(y)), False) >= 0:
            return x, y


def polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    return c if len(c) >= 6 and cv2.contourArea(c) >= 250 else None


def make(job):
    i, kind, seed = job
    rng = np.random.default_rng(seed)
    canvas = np.clip(BASE.astype(np.float32) * rng.uniform(0.75, 1.2), 0, 255).astype(np.uint8)
    objects = []
    if kind != "empty":
        nx, ny = spot(rng)
        ex, ey = spot(rng)
        if kind == "touching":
            reach = K * (ny - HORIZON) * rng.uniform(0.7, 1.1)
            angle = rng.uniform(0, 2 * np.pi)
            ex, ey = nx + reach * np.cos(angle) * 1.3, max(ny + reach * np.sin(angle) * 0.6, 630)
        todo = []
        if kind != "enemy_alone":
            todo.append((NEMESIS, NEM[rng.integers(len(NEM))], nx, ny, K * (ny - HORIZON) * rng.uniform(0.92, 1.08)))
        if kind != "nemesis_alone":
            todo.append((ROBOT, BOTS[rng.integers(len(BOTS))], ex, ey, K * (ey - HORIZON) * rng.uniform(0.8, 1.3)))
        if rng.random() < 0.5:
            todo.reverse()
        for cls, item, x, y, side in todo:
            mask = place(canvas, *cut(item), x, y, side, rng)
            objects = [(c, m & (1 - mask)) for c, m in objects]             # the newer robot hides what is under it
            objects.append((cls, mask))
    polys = [(c, polygon(m), m) for c, m in objects]
    if any(p is None or cv2.contourArea(p) < 0.5 * m.sum() for _, p, m in polys):
        return 0                                                         # a robot mostly hidden or off the picture: drop the image
    name = f"paste_{kind}_{i:05d}"
    cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(OUT / "labels" / f"{name}.txt", "w") as fh:
        for c, p, _ in polys:
            fh.write(f"{c} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in p) + "\n")
    return 1


def setup(base, nem, bots, horizon, k):
    global BASE, NEM, BOTS, HORIZON, K
    BASE, NEM, BOTS, HORIZON, K = base, nem, bots, horizon, k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000, help="fight images")
    ap.add_argument("--empty", type=int, default=300, help="empty arenas")
    ap.add_argument("--preview", type=int, default=0, help="save a contact sheet of this many images to dataset/paste/preview.jpg")
    args = ap.parse_args()
    nem, bots, horizon, k = cutouts()
    print(f"{len(nem)} Nemesis cutouts, {len(bots)} robot cutouts from TV footage; a robot's size is {k:.3f} x (row - {horizon:.0f})")
    shutil.rmtree(OUT, ignore_errors=True)
    for sub in ("images", "labels"):
        (OUT / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    kinds = rng.choice(["apart", "touching", "nemesis_alone", "enemy_alone"], args.n, p=[0.4, 0.4, 0.1, 0.1])
    jobs = [(i, kind, i) for i, kind in enumerate(kinds)] + [(args.n + i, "empty", args.n + i) for i in range(args.empty)]
    with Pool(4, initializer=setup, initargs=(arena(), nem, bots, horizon, k)) as pool:
        written = sum(pool.imap_unordered(make, jobs, chunksize=16))
    names = "  0: robot\n  1: nemesis\n"
    (DATA / "data_paste.yaml").write_text(
        f"# real footage + Unity renders (fight background) + pasted fights and empty arenas\npath: {DATA}\n"
        f"train:\n  - train/images\n  - sim_asrendered/images\n  - paste/images\nval: valid/images\ntest: test/images\nnames:\n{names}")
    print(f"{written} images written to {OUT} ({len(jobs) - written} dropped)")
    if args.preview:
        files = sorted((OUT / "images").glob("*.jpg"))
        tiles = []
        for f in [files[j] for j in rng.permutation(len(files))[:args.preview]]:
            img = cv2.imread(str(f))
            for line in (OUT / "labels" / f"{f.stem}.txt").read_text().splitlines():
                v = line.split()
                pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255) if v[0] == "1" else (0, 255, 0), 2)
            tiles.append(cv2.resize(img, (480, 360)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / "preview.jpg"), np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
