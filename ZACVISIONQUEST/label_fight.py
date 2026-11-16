"""
Label every frame of the held-out Nemesis fight, for scoring only.

    python ZACVISIONQUEST/label_fight.py

Reads   ~/battle_gamma08_fight.mp4 (5,104 frames) and the 327 hand-labelled frames in dataset/valid and dataset/test
Writes  dataset/fight_full/{images,labels}, frames.csv, own_camera_fight_{valid,test,rest}.txt, preview.jpg

Hand-labelled frames keep their hand labels. Every other frame is labelled by SAM 2,
which follows Nemesis and the opponent through the video and is corrected at every
hand-labelled frame. Those machine labels have not been checked by a person:

  valid   frames up to 1090: SAM corrected every few frames. Same half as dataset/valid.
  test    frames 1100 to 2400: the same. Same half as dataset/test.
  rest    frames after 2400: no hand labels at all, SAM runs on from the last one.

A frame is left out when SAM loses a robot or the two masks collapse into one.
Never train on any of this: it is the held-out fight.

CHECKED 2026-11-16: the labels are NOT usable. On preview.jpg the Nemesis mask sits on
empty floor in most machine-labelled frames (and stays frozen on one floor patch from
about frame 2,700 on), and sometimes the two classes are swapped. The own_camera_fight_*
lists were deleted so nothing scores against them. Do not trust a score built on this.
"""
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
OUT = DATA / "fight_full"
VIDEO = Path.home() / "battle_gamma08_fight.mp4"
SAM_WEIGHTS = HERE / "weights" / "sam2.1_b.pt"
NEMESIS, ROBOT = 1, 0
MEMORY = 7                                                       # hand-labelled frames SAM remembers


def hand_labels():
    """{frame number: {class: polygon in pixels}} from the hand-labelled frames."""
    out = {}
    for split in ("valid", "test"):
        for f in (DATA / split / "labels").glob("NEM_V_WitchDoctor__frame_*.txt"):
            n = int(re.search(r"frame_(\d+)$", f.stem).group(1))
            out[n] = {int(l.split()[0]): np.array(l.split()[1:], np.float32).reshape(-1, 2) * [1920, 1440]
                      for l in f.read_text().splitlines() if l.strip()}
    return out


def polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
    return c.astype(np.float32) if len(c) >= 6 and cv2.contourArea(c) >= 250 else None


def main():
    from ultralytics.models.sam import SAM2DynamicInteractivePredictor
    torch.cuda.set_per_process_memory_fraction(0.3)                  # share the GPU with a training run
    sam = SAM2DynamicInteractivePredictor(
        overrides=dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model=str(SAM_WEIGHTS), save=False, verbose=False),
        max_obj_num=2)
    hand = hand_labels()
    shutil.rmtree(OUT, ignore_errors=True)
    for sub in ("images", "labels"):
        (OUT / sub).mkdir(parents=True)
    cap = cv2.VideoCapture(str(VIDEO))
    rows, lists, started = [], {"valid": [], "test": [], "rest": []}, False
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                                   # the dataset is the mirror image of the recording
        polys, source = None, "tracked"
        if n in hand and len(hand[n]) == 2:
            polys, source, started = hand[n], "hand", True
            boxes = [[*hand[n][c].min(0), *hand[n][c].max(0)] for c in (NEMESIS, ROBOT)]
            sam(source=frame, bboxes=boxes, obj_ids=[0, 1], update_memory=True)
            del sam.memory_bank[:-MEMORY]                            # SAM 2 keeps 7; an unbounded bank ran out of GPU memory at frame ~1,600
        elif started:
            res = sam(source=frame)[0]
            if res.masks is not None and len(res.masks.data) == 2:
                masks = res.masks.data.cpu().numpy().astype(np.uint8)
                by_obj = {int(o): masks[j] for j, o in enumerate(res.boxes.cls.tolist())}
                if set(by_obj) == {0, 1} and (by_obj[0] & by_obj[1]).sum() < 0.25 * min(by_obj[0].sum(), by_obj[1].sum()):
                    p = {NEMESIS: polygon(by_obj[0]), ROBOT: polygon(by_obj[1] & (1 - by_obj[0]))}
                    if p[NEMESIS] is not None and p[ROBOT] is not None:
                        polys = p
        if polys is not None:
            name = f"fight__frame_{n:04d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(OUT / "labels" / f"{name}.txt", "w") as fh:
                for cls in (NEMESIS, ROBOT):
                    fh.write(f"{cls} " + " ".join(f"{x / 1920:.6f} {y / 1440:.6f}" for x, y in polys[cls]) + "\n")
            part = "valid" if n <= 1090 else "test" if 1100 <= n < 2400 else "rest" if n >= 2400 else None
            if part:
                lists[part].append(f"./fight_full/images/{name}.jpg")
            rows.append(dict(frame=n, part=part or "gap", source=source))
        elif started:
            rows.append(dict(frame=n, part="skipped", source="tracker lost a robot"))
        n += 1
        if n % 500 == 0:
            print(f"frame {n}: {sum(r['part'] != 'skipped' for r in rows)} labelled, {sum(r['part'] == 'skipped' for r in rows)} skipped", flush=True)
    for part, files in lists.items():
        (DATA / f"own_camera_fight_{part}.txt").write_text("\n".join(files) + "\n")
    with open(OUT / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "part", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{n} frames in the video; labelled: " + ", ".join(f"{k} {len(v)}" for k, v in lists.items())
          + f"; hand-labelled {sum(r['source'] == 'hand' for r in rows)}; skipped {sum(r['part'] == 'skipped' for r in rows)}")

    # a sheet of machine-labelled frames spread over the fight, to look at
    tracked = [r["frame"] for r in rows if r["source"] == "tracked"]
    tiles = []
    for f in [tracked[i] for i in np.linspace(0, len(tracked) - 1, 32).astype(int)]:
        img = cv2.imread(str(OUT / "images" / f"fight__frame_{f:04d}.jpg"))
        pts_all = []
        for line in (OUT / "labels" / f"fight__frame_{f:04d}.txt").read_text().splitlines():
            v = line.split()
            pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [1920, 1440]).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255) if v[0] == "1" else (0, 255, 0), 2)
            pts_all.append(pts)
        p = np.concatenate(pts_all)
        c = (p.min(0) + p.max(0)) // 2
        half = max(int((p.max(0) - p.min(0)).max() * 0.6) + 30, 120)
        x0, y0 = max(c[0] - half, 0), max(c[1] - half, 0)
        crop = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], (240, 240))
        cv2.putText(crop, str(f), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(crop)
    cv2.imwrite(str(OUT / "preview.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 32, 8)]))


if __name__ == "__main__":
    main()
