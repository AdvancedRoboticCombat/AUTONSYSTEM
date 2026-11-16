"""
Build the clean train / valid / test split from the contaminated dataset.

    python ZACVISIONQUEST/build_dataset.py

Reads   ZACVISIONQUEST/Zac_vis_final_segmented_dataset/{train,valid}
Writes  ZACVISIONQUEST/dataset/{train,valid,test}/{images,labels}, data*.yaml, manifest.csv

The original folder is not touched. Images are hard links, so the clean set
takes no extra disk space.

What was wrong with the original split, and what this does about it:

  1. The same video was in train and valid (frames a fraction of a second
     apart). Here every image is assigned to the video or photo set it came
     from, and a whole source goes to exactly one split.
  2. The same fight was filed under several names (a compilation and the
     original clip, for example). Those sources are tied together in
     SAME_FOOTAGE, found by image similarity (check_leaks.py repeats the check).
  3. 3,580 files were byte-for-byte copies (the _dupN / _copyN files), many
     of them on both sides of the split. One copy of each image is kept.
  4. Roboflow wrote about three augmented versions of each frame. Training
     keeps them all; valid and test keep one version per frame so a frame is
     not scored three times.
  5. Some label files list the same object twice. Each row is kept once.
"""
import csv
import hashlib
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "Zac_vis_final_segmented_dataset"
OUT = HERE / "dataset"
NAMES = ["robot", "nemesis"]

# Sources that show the same footage under different names. Each row is one group.
SAME_FOOTAGE = {
    "tv_orbitron_mix": ["COMBINED_Dataset", "DATASET_orbitron", "seq_1", "seq_2", "seq_3"],
    "tv_ripperoni_endgame": [
        "DATASET_ripperonivsendgame",
        "BattleBots-Fight-of-the-Week-RIPperoni-vs_-End-Game-from-World-Championship-VII",
        "BattleBot-Gets-LAUNCHED-Across-The-Arena-_-Vengeance-in-Vegas-2-_-BattleBots",
    ],
    "tv_blacksmith_mix": [
        "videoplayback-1", "seq_7", "seq_8-", "PELEAS-DE-ROBOTS-A-CONTROL-REMOTO",
        "STAY-DOWN-STAY-DOWN-Blacksmith-vs_-Minotaur-BattleBots", "Yeti-vs_-Lucky-BattleBots",
    ],
    "tv_basement_tape": ["seq_9-", "BattleBots-Basement-Tape-2020-Ghost-Raptor-vs-Chomp"],
    "photos_facebook": ["photos_n", "photos_IMG", "photos_FBIMG", "photos_fighter", "photos_nCopie"],
    "photos_web": ["photos_OIP", "photos_images", "photos_th"],
}

# Where each group goes. Anything not named here is a small independent source
# and is dealt out between valid and test by a hash of its name.
PINNED = {
    # our own camera. NEM_04 and NEM_08 are the same session (same robots parked
    # in the same corner), so they stay together.
    "NEM_08": "train",
    "NEM_04": "train",
    # the one recorded Nemesis fight: never in train. First half valid, second half test.
    "NEM_V_WitchDoctor": "time_split",
    # overhead greyscale fights: two different fights, one seen in training, one not
    "gray_fight4": "train",
    "gray_fight3": "test",
    "bilibili_Av52106355": "train",
    "tv_orbitron_mix": "train",
    "tv_ripperoni_endgame": "train",
    "tv_blacksmith_mix": "train",
    "tv_basement_tape": "valid",
    "photos_web": "valid",
    "photos_facebook": "test",
}
TIME_SPLIT_GAP = 30          # video frames (1 s) dropped between the valid half and the test half


def parse(stem):
    """(source video or photo set, frame id inside it) for an image file name."""
    s = re.sub(r"\.rf\.[0-9a-f]{32}$", "", stem)                # Roboflow augmentation hash
    s = re.sub(r"(_dup\d+)?(_copy\d+)?$", "", s)                # oversampling copies
    s = re.sub(r"^segment_mask_dataset_may26_(train|test|valid)__", "", s)
    s = re.sub(r"_(jpe?g|png|webp)$", "", s, flags=re.I)
    m = re.match(r"(?:two_bots_fight(\d)_\d+_\d+|collisions_fight(\d))_frame_(\d+)$", s)
    if m:
        return f"gray_fight{m.group(1) or m.group(2)}", s
    m = re.match(r"sam2_collisions_(\d)_\d+__frame_(\d+)$", s)
    if m:
        return f"gray_fight{m.group(1)}", s
    m = re.match(r"\d+_MVI_\d+-(Av\d+)-P(\d+)-_mp4-(\d+)$", s)
    if m:
        return f"bilibili_{m.group(1)}", s
    m = re.match(r"(.+)_mp4-(\d+)$", s)
    if m:
        return m.group(1), s
    m = re.match(r"(.+)__frame_(\d+)$", s)
    if m:
        return m.group(1), s
    m = re.match(r"(\d+-?)_(\d{10})$", s)
    if m:
        return f"seq_{m.group(1)}", s
    if re.fullmatch(r"[0-9a-f]{32}", s):
        return "photos_hashname", s
    return "photos_" + (re.sub(r"[\d_\-]+", "", s) or "numbered"), s


def frame_number(frame):
    return int(re.search(r"frame_(\d+)$", frame).group(1))


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_yamls():
    """The data files training reads. They differ only in what is trained on; valid and test are always the same."""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    tail = f"val: valid/images\ntest: test/images\nnames:\n{names}\n"
    (OUT / "data.yaml").write_text(
        f"# real footage + Unity robots pasted onto our own arena (run unity_to_yolo.py to fill sim/)\npath: {OUT}\n"
        f"train:\n  - train/images\n  - sim/images\n{tail}")
    (OUT / "data_real_only.yaml").write_text(f"# real footage only\npath: {OUT}\ntrain: train/images\n{tail}")
    (OUT / "data_sim_asrendered.yaml").write_text(
        f"# NOT for a model we keep: the Unity renders as rendered, whose background is the held-out fight.\n"
        f"# Only used to measure how much that leak is worth.\npath: {OUT}\n"
        f"train:\n  - train/images\n  - sim_asrendered/images\n{tail}")
    (OUT / "data_own_camera.yaml").write_text(
        f"# scoring only: the held-out Nemesis fight from our own camera\npath: {OUT}\n"
        f"train: train/images\nval: own_camera_valid.txt\ntest: own_camera_test.txt\nnames:\n{names}\n")


def main():
    group_of = {src: group for group, members in SAME_FOOTAGE.items() for src in members}

    # 1. every file, with its source group, frame and content hash
    files = []
    for split in ("train", "valid"):
        for img in sorted((SRC / split / "images").glob("*.jpg")):
            label = SRC / split / "labels" / (img.stem + ".txt")
            if not label.exists():
                continue
            source, frame = parse(img.stem)
            files.append(dict(old_split=split, img=img, label=label, source=source,
                              group=group_of.get(source, source), frame=frame, md5=md5(img)))
    print(f"{len(files)} labelled images in the original train + valid")

    # 2. one file per distinct image (shortest name wins: the plain frame, not its copies)
    by_hash = defaultdict(list)
    for f in files:
        by_hash[f["md5"]].append(f)
    distinct = [min(v, key=lambda f: (len(f["img"].stem), f["img"].stem)) for v in by_hash.values()]
    print(f"{len(distinct)} distinct images ({len(files) - len(distinct)} byte-identical copies dropped)")

    # 3. a split for every group
    groups = defaultdict(list)
    for f in distinct:
        groups[f["group"]].append(f)
    split_of = {}
    for group in sorted(groups):
        if group in PINNED:
            split_of[group] = PINNED[group]
        else:
            split_of[group] = "valid" if int(hashlib.md5(group.encode()).hexdigest(), 16) % 2 == 0 else "test"

    chosen, dropped_gap = [], 0
    for group, members in groups.items():
        where = split_of[group]
        if where == "time_split":
            numbers = sorted({frame_number(f["frame"]) for f in members})
            cut = numbers[len(numbers) // 2]
            for f in members:
                n = frame_number(f["frame"])
                if n < cut - TIME_SPLIT_GAP // 2:
                    chosen.append((f, "valid"))
                elif n > cut + TIME_SPLIT_GAP // 2:
                    chosen.append((f, "test"))
                else:
                    dropped_gap += 1
            continue
        if where == "train":
            chosen += [(f, "train") for f in members]
            continue
        # valid / test: one image per frame
        per_frame = defaultdict(list)
        for f in members:
            per_frame[f["frame"]].append(f)
        chosen += [(min(v, key=lambda f: f["img"].stem), where) for v in per_frame.values()]

    # 4. write
    if OUT.exists():
        for split in ("train", "valid", "test"):
            shutil.rmtree(OUT / split, ignore_errors=True)
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    rows, repeated_rows = [], 0
    for f, split in sorted(chosen, key=lambda c: (c[1], c[0]["img"].stem)):
        img_out = OUT / split / "images" / f["img"].name
        os.link(f["img"], img_out)
        # a few label files list the same object twice; keep each row once, in order
        lines = [l.strip() for l in f["label"].read_text().splitlines() if l.strip()]
        unique = list(dict.fromkeys(lines))
        repeated_rows += len(lines) - len(unique)
        (OUT / split / "labels" / f["label"].name).write_text("\n".join(unique) + "\n")
        rows.append(dict(split=split, file=f["img"].name, group=f["group"], source=f["source"], frame=f["frame"],
                         was_in=f["old_split"], md5=f["md5"]))
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # own-camera subsets, scored on their own because that is what the robot will see
    for split in ("valid", "test"):
        own = [f"./{split}/images/{r['file']}" for r in rows if r["split"] == split and r["group"] == "NEM_V_WitchDoctor"]
        (OUT / f"own_camera_{split}.txt").write_text("\n".join(own) + "\n")
    write_yamls()

    # 5. prove it
    seen_group, seen_hash = defaultdict(set), defaultdict(set)
    for r in rows:
        seen_group[r["group"]].add(r["split"])
        seen_hash[r["md5"]].add(r["split"])
    shared_groups = {g: s for g, s in seen_group.items() if len(s) > 1 and split_of[g] != "time_split"}
    shared_images = sum(len(s) > 1 for s in seen_hash.values())
    assert not shared_groups, shared_groups
    assert shared_images == 0

    print(f"\n{'split':6s} {'images':>7s} {'frames':>7s} {'sources':>8s} {'robot':>7s} {'nemesis':>8s}")
    for split in ("train", "valid", "test"):
        mine = [r for r in rows if r["split"] == split]
        classes = Counter()
        for r in mine:
            for line in (OUT / split / "labels" / (Path(r["file"]).stem + ".txt")).read_text().splitlines():
                if line.strip():
                    classes[int(line.split()[0])] += 1
        print(f"{split:6s} {len(mine):7d} {len({(r['group'], r['frame']) for r in mine}):7d} "
              f"{len({r['group'] for r in mine}):8d} {classes[0]:7d} {classes[1]:8d}")
    print(f"\n{dropped_gap} frames dropped at the valid/test boundary of the held-out fight")
    print(f"{repeated_rows} repeated label rows removed")
    print("sources in more than one split: 0   identical images in more than one split: 0")
    for split in ("valid", "test"):
        counts = Counter(r["group"] for r in rows if r["split"] == split)
        print(f"{split}: " + ", ".join(f"{g} {c}" for g, c in counts.most_common()))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
