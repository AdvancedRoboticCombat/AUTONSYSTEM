"""
Sweep YOLO segmentation models on the clean dataset.

    python ZACVISIONQUEST/sweep.py                 # all stages, resumes where it stopped
    python ZACVISIONQUEST/sweep.py --stage model   # one stage
    python ZACVISIONQUEST/sweep.py --epochs 5      # quick dry run
    python ZACVISIONQUEST/sweep.py --report        # just print the table

Stages, each building on the winner of the one before:

  model   which network: YOLO26 n/s/m and YOLO11 m (what the dashboard runs today), all at 960 px
  size    the winner at 640 and 1280 px (robots are small in our camera)
  data    the winner with no Unity renders, with heavier rotation, with copy-paste

Every run is scored on valid, which shares no footage with train:

  score         mask mAP50-95 over both classes, all of valid. Runs are ranked by this.
  nemesis       mask mAP50-95 of the nemesis class (only our camera has it)
  own camera    mask mAP50-95 on the held-out Nemesis fight alone
  ms            GPU time per image at batch 1, at that run's image size

Test is touched once, at the end, for the winner only.
Results: ZACVISIONQUEST/runs/sweep/results.csv
"""
import argparse
import csv
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
RUNS = HERE / "runs" / "sweep"
WEIGHTS = HERE / "weights"
RESULTS = RUNS / "results.csv"
FIELDS = ["stage", "name", "model", "imgsz", "data", "extra", "epochs", "best_epoch", "score", "mask50", "box",
          "nemesis", "nemesis50", "robot", "own_camera", "own_camera50", "ms", "params_M", "train_min", "weights"]

MODELS = ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg", "yolo11m-seg"]     # also downloaded: yolo26l-seg, yolo11s-seg
BATCH = {640: 32, 960: 16, 1280: 8}


def load_results():
    if not RESULTS.exists():
        return []
    return list(csv.DictReader(open(RESULTS)))


def save_result(row):
    RUNS.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def class_ap(metrics, part, cls):
    """(mAP50, mAP50-95) of one class, or (nan, nan) when valid has none of it."""
    m = getattr(metrics, part)
    idx = list(m.ap_class_index)
    if cls not in idx:
        return float("nan"), float("nan")
    _, _, ap50, ap = m.class_result(idx.index(cls))
    return float(ap50), float(ap)


def latency_ms(weights, imgsz):
    """GPU milliseconds per camera frame at batch 1."""
    import numpy as np
    import torch
    from ultralytics import YOLO
    model = YOLO(weights)
    frame = np.random.default_rng(0).integers(0, 255, (1440, 1920, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(frame, imgsz=imgsz, verbose=False, half=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        model.predict(frame, imgsz=imgsz, verbose=False, half=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 50 * 1000


def run(stage, name, model, imgsz, data="data.yaml", epochs=20, **extra):
    done = {r["name"]: r for r in load_results()}
    if name in done:
        print(f"[{stage}] {name}: already done, score {float(done[name]['score']):.3f}")
        return done[name]
    from ultralytics import YOLO
    print(f"\n[{stage}] {name}: {model} at {imgsz}px on {data} {extra or ''}", flush=True)
    t0 = time.time()
    net = YOLO(str(WEIGHTS / f"{model}.pt"))
    net.train(data=str(DATA / data), imgsz=imgsz, epochs=epochs, batch=BATCH[imgsz], project=str(RUNS), name=name,
              exist_ok=True, workers=12, patience=10, close_mosaic=5, plots=False, seed=0, verbose=False, **extra)
    minutes = (time.time() - t0) / 60
    best = RUNS / name / "weights" / "best.pt"
    net = YOLO(str(best))
    kw = dict(imgsz=imgsz, batch=BATCH[imgsz], plots=False, verbose=False, workers=8)
    everything = net.val(data=str(DATA / "data.yaml"), split="val", project=str(RUNS / name), name="val_all", exist_ok=True, **kw)
    own = net.val(data=str(DATA / "data_own_camera.yaml"), split="val", project=str(RUNS / name), name="val_own", exist_ok=True, **kw)
    nem50, nem = class_ap(everything, "seg", 1)
    _, robot = class_ap(everything, "seg", 0)
    results_csv = list(csv.DictReader(open(RUNS / name / "results.csv")))
    key = next(k for k in results_csv[0] if "mAP50-95(M)" in k)
    best_epoch = 1 + max(range(len(results_csv)), key=lambda i: float(results_csv[i][key]))
    row = dict(stage=stage, name=name, model=model, imgsz=imgsz, data=data, extra=str(extra or ""), epochs=len(results_csv),
               best_epoch=best_epoch, score=round(float(everything.seg.map), 4), mask50=round(float(everything.seg.map50), 4),
               box=round(float(everything.box.map), 4), nemesis=round(nem, 4), nemesis50=round(nem50, 4), robot=round(robot, 4),
               own_camera=round(float(own.seg.map), 4), own_camera50=round(float(own.seg.map50), 4),
               ms=round(latency_ms(str(best), imgsz), 1), params_M=round(sum(p.numel() for p in net.model.parameters()) / 1e6, 1),
               train_min=round(minutes, 1), weights=str(best))
    save_result(row)
    print(f"[{stage}] {name}: score {row['score']:.3f} nemesis {row['nemesis']:.3f} own camera {row['own_camera']:.3f} "
          f"{row['ms']} ms, trained in {minutes:.0f} min", flush=True)
    return row


def winner(stages):
    rows = [r for r in load_results() if r["stage"] in stages]
    return max(rows, key=lambda r: float(r["score"]))


def report():
    rows = load_results()
    if not rows:
        print("no results yet")
        return
    print(f"\n{'stage':6s} {'run':30s} {'px':>5s} {'score':>6s} {'nemesis':>8s} {'robot':>6s} {'own cam':>8s} {'own@50':>7s} "
          f"{'ms':>5s} {'params':>7s} {'min':>5s}")
    for r in sorted(rows, key=lambda r: -float(r["score"])):
        print(f"{r['stage']:6s} {r['name']:30s} {r['imgsz']:>5s} {float(r['score']):6.3f} {float(r['nemesis']):8.3f} "
              f"{float(r['robot']):6.3f} {float(r['own_camera']):8.3f} {float(r['own_camera50']):7.3f} {float(r['ms']):5.1f} "
              f"{float(r['params_M']):6.1f}M {float(r['train_min']):5.0f}")


def test_winner():
    """The one look at test: the best run, on all of test and on the held-out fight alone."""
    from ultralytics import YOLO
    best = winner({"model", "size", "data"})
    out = RUNS / "test_result.txt"
    net = YOLO(best["weights"])
    kw = dict(imgsz=int(best["imgsz"]), batch=BATCH[int(best["imgsz"])], plots=True, verbose=False, workers=8)
    everything = net.val(data=str(DATA / "data.yaml"), split="test", project=str(RUNS / best["name"]), name="test_all", exist_ok=True, **kw)
    own = net.val(data=str(DATA / "data_own_camera.yaml"), split="test", project=str(RUNS / best["name"]), name="test_own", exist_ok=True, **kw)
    nem50, nem = class_ap(everything, "seg", 1)
    rob50, rob = class_ap(everything, "seg", 0)
    lines = [
        f"winner: {best['name']} ({best['model']} at {best['imgsz']} px), weights {best['weights']}",
        f"valid  score {float(best['score']):.3f}  nemesis {float(best['nemesis']):.3f}  own camera {float(best['own_camera']):.3f}",
        f"test   score {everything.seg.map:.3f}  mask mAP50 {everything.seg.map50:.3f}  box mAP50-95 {everything.box.map:.3f}",
        f"test   nemesis mask mAP50-95 {nem:.3f} (mAP50 {nem50:.3f})   robot {rob:.3f} (mAP50 {rob50:.3f})",
        f"test   own camera (held-out fight) mask mAP50-95 {own.seg.map:.3f}  mAP50 {own.seg.map50:.3f}  "
        f"precision {own.seg.mp:.3f}  recall {own.seg.mr:.3f}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def any_robot(weights, imgsz, split="val"):
    """Score a model on 'is there a robot here', ignoring which class it says.

    This is the only fair comparison with the old dashboard models, which have
    one class. Returns mask mAP50-95, mAP50, precision and recall for all of
    the split and for the held-out fight alone.
    """
    from ultralytics import YOLO
    net = YOLO(str(weights))
    out = {}
    for label, data in (("all", "data.yaml"), ("own camera", "data_own_camera.yaml")):
        m = net.val(data=str(DATA / data), split=split, imgsz=imgsz, batch=8, single_cls=True, plots=False, verbose=False,
                    workers=8, project=str(RUNS / "any_robot"), name=Path(weights).stem, exist_ok=True)
        out[label] = (float(m.seg.map), float(m.seg.map50), float(m.seg.mp), float(m.seg.mr))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--any-robot", nargs="+", metavar="WEIGHTS", help="score these weights class-blind and exit")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280], help="image sizes for --any-robot")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="split for --any-robot")
    ap.add_argument("--stage", choices=["model", "size", "data", "test"], action="append")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.any_robot:
        print(f"class-blind scores on {args.split} (mask mAP50-95 / mAP50 / precision / recall)")
        for w in args.any_robot:
            for px in args.imgsz:
                s = any_robot(w, px, args.split)
                print(f"  {Path(w).stem:28s} {px:5d}px   all {s['all'][0]:.3f} / {s['all'][1]:.3f} / {s['all'][2]:.3f} / {s['all'][3]:.3f}"
                      f"   own camera {s['own camera'][0]:.3f} / {s['own camera'][1]:.3f} / {s['own camera'][2]:.3f} / {s['own camera'][3]:.3f}",
                      flush=True)
        return
    stages = args.stage or ["model", "size", "data", "test"]
    WEIGHTS.mkdir(exist_ok=True)

    if "model" in stages:
        for m in args.models:
            run("model", f"{m}_960", m, 960, epochs=args.epochs)
    if "size" in stages:
        best = winner({"model"})
        for px in (640, 1280):
            run("size", f"{best['model']}_{px}", best["model"], px, epochs=args.epochs)
    if "data" in stages:
        best = winner({"model", "size"})
        m, px = best["model"], int(best["imgsz"])
        run("data", f"{m}_{px}_no_sim", m, px, data="data_real_only.yaml", epochs=args.epochs)
        run("data", f"{m}_{px}_rotate", m, px, epochs=args.epochs, degrees=25.0, flipud=0.5)
        run("data", f"{m}_{px}_copy_paste", m, px, epochs=args.epochs, copy_paste=0.3)
    report()
    if "test" in stages:
        test_winner()


if __name__ == "__main__":
    main()
