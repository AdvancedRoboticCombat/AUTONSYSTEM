"""
Train and compare YOLO segmentation models on the clean dataset.

    python ZACVISIONQUEST/sweep.py                  # all stages, resumes where it stopped
    python ZACVISIONQUEST/sweep.py --stage data     # one stage
    python ZACVISIONQUEST/sweep.py --report         # just print the table
    python ZACVISIONQUEST/sweep.py --score W.pt     # score any weights on the held-out fight
    python ZACVISIONQUEST/sweep.py --test           # the one look at test, for the winner

Stages, each building on the winner of the one before:

  data    what to train on, with one network (YOLO26 s at 960 px): real footage only,
          with the Unity robots, with copy-paste on top
  model   which network: YOLO26 n / m / l, and YOLO11 m (what the dashboard runs today)
  size    the winner at 640 and 1280 px
  final   the winner trained for longer

One more run, marked `leak`, is never a winner: it trains on the Unity renders as
rendered, whose background is the held-out fight, to measure what that leak is worth.

How a run is scored. Everything is measured on valid, which shares no footage with train.

  fight     share of the held-out fight's frames in which Nemesis and the opponent are
            both found, each with the right class, and nothing else is reported. This is
            what the dashboard needs from the model, so runs are ranked by it.
  nemesis   share of those frames in which Nemesis is found and called nemesis
  opponent  share in which the opponent is found and called robot
  extra     detections per frame that are not a robot (or are a second copy of one)
  mixed up  share of frames in which a robot is found but given the other robot's class
  own cam   mask mAP50-95 on the held-out fight. Breaks ties in `fight`.
  all       mask mAP50-95 over all of valid. Mostly TV footage: it shows whether the
            model still sees robots it has never met.
  ms        GPU time for one 1920x1440 camera frame

A robot counts as found when a detection's mask overlaps its labelled mask by half or
more (IoU 0.5), at the confidence the dashboard uses (0.25).

The fight has 164 valid frames, so `fight` moves in steps of 0.006 and differences
under 0.02 (three frames) are noise. Runs that close count as tied.

Each run saves every epoch and keeps the one that does best on the fight, because the
library's own choice of best epoch is made on all of valid, which is mostly TV footage.

Test is touched once, at the end, for the winner only.
Results: ZACVISIONQUEST/runs/sweep/results.csv
"""
import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
RUNS = HERE / "runs" / "sweep"
WEIGHTS = HERE / "weights"
RESULTS = RUNS / "results.csv"
ROBOT, NEMESIS = 0, 1
CONF = 0.25                 # the dashboard's detection threshold (the library default)
FOUND_IOU = 0.5
FIELDS = ["stage", "name", "model", "imgsz", "data", "extra", "tag", "epochs", "epoch", "fight", "nemesis", "opponent", "extra_dets",
          "mixed_up", "seen_both", "off_px", "own_camera", "own_camera50", "all_valid", "all_nemesis", "all_robot", "ms",
          "params_M", "train_min", "weights"]

BATCH = {640: 32, 960: 16, 1280: 8}
# Fixed for every run. Left on "auto", the library switches optimizer at 10,000 steps,
# which this dataset straddles: adding the Unity images alone would change the optimizer.
TRAIN = dict(optimizer="MuSGD", lr0=0.01, momentum=0.9, warmup_bias_lr=0.0, close_mosaic=5, patience=0, workers=12,
             plots=False, seed=0, verbose=False, save_period=1)


# ---------------------------------------------------------------- scoring

def polygons(label_file):
    """[(class, normalised polygon), ...] of a YOLO segmentation label file."""
    out = []
    for line in Path(label_file).read_text().splitlines():
        vals = line.split()
        if len(vals) >= 7:
            out.append((int(vals[0]), np.array(vals[1:], np.float32).reshape(-1, 2)))
    return out


def raster(poly, shape=(720, 960)):
    return cv2.fillPoly(np.zeros(shape, np.uint8), [(poly * [shape[1], shape[0]]).astype(np.int32)], 1)


def centre(mask):
    m = cv2.moments(mask, binaryImage=True)
    return np.array([m["m10"], m["m01"]]) / max(m["m00"], 1.0)


def fight(weights, imgsz, split="valid", conf=CONF, keep_frames=False):
    """How a model does on the held-out fight, frame by frame, the way the dashboard uses it.

    Every frame has one Nemesis and one opponent. A model with a single class is scored
    on finding them only; it cannot say which is which.
    """
    from ultralytics import YOLO
    files = [DATA / line[2:] for line in (DATA / f"own_camera_{split}.txt").read_text().split()]
    net = YOLO(str(weights))
    one_class = len(net.names) == 1
    tally, off, frames = Counter(), [], []
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        for f, r in zip(batch, net.predict([str(b) for b in batch], imgsz=imgsz, conf=conf, half=True, batch=16, verbose=False)):
            truth = {cls: raster(p) for cls, p in polygons(f.parent.parent / "labels" / f"{f.stem}.txt")}
            dets = []
            if r.masks is not None:
                for poly, cls, c in zip(r.masks.xyn, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    if len(poly) >= 3:
                        dets.append((c, int(cls), raster(np.asarray(poly, np.float32))))
            match, extra = {}, 0
            for c, cls, mask in sorted(dets, key=lambda d: -d[0]):
                hit = [k for k, t in truth.items() if k not in match
                       and (mask & t).sum() >= FOUND_IOU * (mask | t).sum() and t.any()]
                if hit:
                    match[hit[0]] = (cls, mask)
                else:
                    extra += 1
            right = {k: k in match and (one_class or match[k][0] == k) for k in (NEMESIS, ROBOT)}
            mixed = any(k in match and not right[k] for k in (NEMESIS, ROBOT))
            tally["frames"] += 1
            tally["nemesis"] += right[NEMESIS]
            tally["opponent"] += right[ROBOT]
            tally["seen_both"] += len(match) == 2
            tally["mixed_up"] += mixed
            tally["extra"] += extra
            tally["fight"] += right[NEMESIS] and right[ROBOT] and extra == 0
            off += [float(np.linalg.norm(centre(match[k][1]) - centre(truth[k]))) * 2 for k in match]    # in camera pixels
            if keep_frames:
                frames.append(dict(file=f.name, nemesis=right[NEMESIS], opponent=right[ROBOT], extra=extra, mixed_up=mixed))
    n = tally["frames"]
    out = dict(frames=n, fight=tally["fight"] / n, nemesis=tally["nemesis"] / n, opponent=tally["opponent"] / n,
               seen_both=tally["seen_both"] / n, mixed_up=tally["mixed_up"] / n, extra_dets=tally["extra"] / n,
               off_px=float(np.median(off)) if off else float("nan"), one_class=one_class)
    if keep_frames:
        out["per_frame"] = frames
    return out


def rank(row):
    """Sort key: fight in steps of 0.02 (closer than that is a tie), then own-camera mAP."""
    return round(float(row["fight"]) * 50), float(row["own_camera"])


def class_ap(metrics, cls):
    """mask mAP50-95 of one class, or nan when the split has none of it."""
    idx = list(metrics.seg.ap_class_index)
    return float(metrics.seg.class_result(idx.index(cls))[3]) if cls in idx else float("nan")


def own_camera_map(net, imgsz, split="val"):
    """(mask mAP50-95, mask mAP50) on the held-out fight."""
    m = net.val(data=str(DATA / "data_own_camera.yaml"), split=split, imgsz=imgsz, batch=BATCH[imgsz], plots=False,
                verbose=False, workers=8, project=str(RUNS / "_val"), name="own", exist_ok=True)
    return float(m.seg.map), float(m.seg.map50)


def latency_ms(weights, imgsz):
    """GPU milliseconds per camera frame at batch 1."""
    import torch
    from ultralytics import YOLO
    model = YOLO(str(weights))
    frame = np.random.default_rng(0).integers(0, 255, (1440, 1920, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(frame, imgsz=imgsz, verbose=False, half=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        model.predict(frame, imgsz=imgsz, verbose=False, half=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 50 * 1000


def score(weights, imgsz, split="val"):
    """Every number in the table for one set of weights."""
    from ultralytics import YOLO
    net = YOLO(str(weights))
    f = fight(weights, imgsz, "valid" if split == "val" else "test")
    own, own50 = own_camera_map(net, imgsz, split)
    everything = net.val(data=str(DATA / "data.yaml"), split=split, imgsz=imgsz, batch=BATCH[imgsz], plots=False, verbose=False,
                         workers=8, project=str(RUNS / "_val"), name="all", exist_ok=True)
    return dict(fight=round(f["fight"], 4), nemesis=round(f["nemesis"], 4), opponent=round(f["opponent"], 4),
                extra_dets=round(f["extra_dets"], 4), mixed_up=round(f["mixed_up"], 4), seen_both=round(f["seen_both"], 4),
                off_px=round(f["off_px"], 1), own_camera=round(own, 4), own_camera50=round(own50, 4),
                all_valid=round(float(everything.seg.map), 4), all_nemesis=round(class_ap(everything, NEMESIS), 4),
                all_robot=round(class_ap(everything, ROBOT), 4))


# ---------------------------------------------------------------- training

def epochs_done(run_dir):
    f = run_dir / "results.csv"
    return len(list(csv.DictReader(open(f)))) if f.exists() else 0


def choose_epoch(run_dir, imgsz):
    """Score every saved epoch on the held-out fight, keep the best as weights/chosen.pt, delete the rest."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import strip_optimizer
    rows = []
    for ck in sorted((run_dir / "weights").glob("epoch*.pt"), key=lambda p: int(p.stem[5:])):
        f = fight(ck, imgsz)
        own, own50 = own_camera_map(YOLO(str(ck)), imgsz)
        rows.append(dict(epoch=int(ck.stem[5:]) + 1, fight=round(f["fight"], 4), nemesis=round(f["nemesis"], 4),
                         opponent=round(f["opponent"], 4), extra_dets=round(f["extra_dets"], 4),
                         mixed_up=round(f["mixed_up"], 4), own_camera=round(own, 4), own_camera50=round(own50, 4), file=ck))
        print(f"    epoch {rows[-1]['epoch']:3d}: fight {f['fight']:.3f}  nemesis {f['nemesis']:.3f}  opponent {f['opponent']:.3f}  "
              f"extra {f['extra_dets']:.2f}  own cam {own:.3f}", flush=True)
    best = max(rows, key=rank)
    strip_optimizer(str(best["file"]), str(run_dir / "weights" / "chosen.pt"))
    with open(run_dir / "epochs.csv", "w", newline="") as fh:                # written last: it marks the run as finished
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "file"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        r["file"].unlink()


def train(name, model, imgsz, data, epochs, **extra):
    """Train one run, or pick it up where it stopped. Returns (weights, chosen epoch, minutes of training)."""
    import torch
    from ultralytics import YOLO
    run_dir = RUNS / name
    if not (run_dir / "epochs.csv").exists():
        if epochs_done(run_dir) < epochs:
            last = run_dir / "weights" / "last.pt"
            if last.exists() and epochs_done(run_dir) > 0:
                print(f"  resuming {name} after epoch {epochs_done(run_dir)}", flush=True)
                YOLO(str(last)).train(resume=True)
            else:
                batch = int(os.environ.get("SWEEP_BATCH", BATCH[imgsz]))     # smaller when the GPU is shared
                while True:
                    try:
                        YOLO(str(WEIGHTS / f"{model}.pt")).train(
                            data=str(DATA / data), imgsz=imgsz, epochs=epochs, batch=batch, project=str(RUNS), name=name,
                            exist_ok=True, **{**TRAIN, **extra})
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        batch //= 2
                        print(f"  out of GPU memory: retrying {name} with batch {batch}", flush=True)
                        if batch < 2:
                            raise
        choose_epoch(run_dir, imgsz)
    best = max(csv.DictReader(open(run_dir / "epochs.csv")), key=rank)
    minutes = float(list(csv.DictReader(open(run_dir / "results.csv")))[-1]["time"]) / 60
    return run_dir / "weights" / "chosen.pt", int(best["epoch"]), minutes


def load_results():
    return list(csv.DictReader(open(RESULTS))) if RESULTS.exists() else []


def run(stage, tag, model, imgsz, data="data.yaml", epochs=20, **extra):
    """One row of the table. `tag` names what is special about the run; with model and size it makes the run's name.

    A run that fails is reported and skipped, so one bad run does not stop a night's sweep.
    """
    name = f"{model}_{imgsz}_{tag}"
    done = {r["name"]: r for r in load_results()}
    if name in done:
        print(f"[{stage}] {name}: already done, fight {float(done[name]['fight']):.3f}")
        return done[name]
    print(f"\n[{stage}] {name}: {model} at {imgsz}px on {data} {extra or ''}", flush=True)
    try:
        weights, epoch, minutes = train(name, model, imgsz, data, epochs, **extra)
    except Exception:
        import traceback
        print(f"[{stage}] {name}: FAILED\n{traceback.format_exc()}", flush=True)
        return None
    from ultralytics import YOLO
    row = dict(stage=stage, name=name, model=model, imgsz=imgsz, data=data, extra=json.dumps(extra), tag=tag,
               epochs=epochs_done(RUNS / name), epoch=epoch, **score(weights, imgsz),
               ms=round(latency_ms(weights, imgsz), 1),
               params_M=round(sum(p.numel() for p in YOLO(str(weights)).model.parameters()) / 1e6, 1),
               train_min=round(minutes, 1), weights=str(weights))
    RUNS.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{stage}] {name}: fight {row['fight']:.3f}  nemesis {row['nemesis']:.3f}  opponent {row['opponent']:.3f}  "
          f"own cam {row['own_camera']:.3f}  all {row['all_valid']:.3f}  {row['ms']} ms, epoch {epoch}, {minutes:.0f} min", flush=True)
    return row


def winner(stages):
    return max((r for r in load_results() if r["stage"] in stages), key=rank)


def settings(row):
    """What a later stage inherits from a winner: (model, image size, tag, data, extra training settings)."""
    return row["model"], int(row["imgsz"]), row["tag"], row["data"], json.loads(row["extra"])


def report():
    rows = load_results()
    if not rows:
        print("no results yet")
        return
    print(f"\n{'stage':6s} {'run':34s} {'px':>5s} {'fight':>6s} {'nemesis':>8s} {'opponent':>9s} {'extra':>6s} {'mixed up':>9s} "
          f"{'own cam':>8s} {'all':>6s} {'ms':>5s} {'params':>7s} {'epoch':>6s} {'min':>5s}")
    for r in sorted(rows, key=rank, reverse=True):
        print(f"{r['stage']:6s} {r['name']:34s} {r['imgsz']:>5s} {float(r['fight']):6.3f} {float(r['nemesis']):8.3f} "
              f"{float(r['opponent']):9.3f} {float(r['extra_dets']):6.2f} {float(r['mixed_up']):9.3f} {float(r['own_camera']):8.3f} "
              f"{float(r['all_valid']):6.3f} {float(r['ms']):5.1f} {float(r['params_M']):6.1f}M {r['epoch']:>3s}/{r['epochs']:<2s} "
              f"{float(r['train_min']):5.0f}")


def test_winner():
    """The one look at test: the best run, on all of test and on the second half of the held-out fight."""
    best = winner({"compare", "fix"})
    s = score(best["weights"], int(best["imgsz"]), split="test")
    lines = [
        f"winner: {best['name']} ({best['model']} at {best['imgsz']} px), weights {best['weights']}",
        f"valid  fight {float(best['fight']):.3f}  nemesis {float(best['nemesis']):.3f}  opponent {float(best['opponent']):.3f}  "
        f"extra {float(best['extra_dets']):.2f}  own cam {float(best['own_camera']):.3f}  all {float(best['all_valid']):.3f}",
        f"test   fight {s['fight']:.3f}  nemesis {s['nemesis']:.3f}  opponent {s['opponent']:.3f}  "
        f"extra {s['extra_dets']:.2f}  own cam {s['own_camera']:.3f}  all {s['all_valid']:.3f}",
        f"test   mixed up {s['mixed_up']:.3f}  both seen (any class) {s['seen_both']:.3f}  centre off by {s['off_px']} px (median)  "
        f"own cam mAP50 {s['own_camera50']:.3f}  nemesis mAP50-95 {s['all_nemesis']:.3f}  robot mAP50-95 {s['all_robot']:.3f}",
    ]
    (RUNS / "test_result.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["data", "model", "size", "final"], action="append")
    ap.add_argument("--epochs", type=int, default=20, help="epochs per run in the data, model and size stages")
    ap.add_argument("--final-epochs", type=int, default=60)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--test", action="store_true", help="score the winner on test (once)")
    ap.add_argument("--score", nargs="+", metavar="WEIGHTS", help="score these weights on the held-out fight and exit")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280], help="image sizes for --score")
    ap.add_argument("--split", default="valid", choices=["valid", "test", "fight_valid", "fight_test", "fight_rest"],
                    help="part of the fight for --score; the fight_ ones are every frame (label_fight.py)")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.test:
        return test_winner()
    if args.score:
        print(f"held-out fight, {args.split} half, confidence {CONF}")
        print(f"{'weights':30s} {'px':>5s} {'fight':>6s} {'nemesis':>8s} {'opponent':>9s} {'both seen':>10s} {'extra':>6s} {'mixed up':>9s} {'off px':>7s}")
        for w in args.score:
            for px in args.imgsz:
                f = fight(w, px, args.split)
                blind = "   (one class: cannot tell which robot is which)" if f["one_class"] else ""
                print(f"{Path(w).stem[:30]:30s} {px:5d} {f['fight']:6.3f} {f['nemesis']:8.3f} {f['opponent']:9.3f} {f['seen_both']:10.3f} "
                      f"{f['extra_dets']:6.2f} {f['mixed_up']:9.3f} {f['off_px']:7.1f}{blind}", flush=True)
        return

    # YOLO26 m and l at two image sizes, on real footage + the Unity renders with the fight background
    for m, px in (("yolo26m-seg", 960), ("yolo26l-seg", 960), ("yolo26m-seg", 1280)):
        run("compare", "fightbg", m, px, data="data_sim_asrendered.yaml", epochs=args.epochs)
    # YOLO26 m at 1280 again with every fix for the epoch-to-epoch swings in one run: the pasted fights
    # and empty arenas (paste_fights.py), a learning rate five times lower, and the picture rotated,
    # warped, flipped upside down and varied more in brightness.
    run("fix", "all_fixes", "yolo26m-seg", 1280, data="data_paste.yaml", epochs=args.epochs, lr0=0.002, degrees=10.0,
        perspective=0.0003, flipud=0.5, hsv_v=0.6)
    report()


if __name__ == "__main__":
    main()
