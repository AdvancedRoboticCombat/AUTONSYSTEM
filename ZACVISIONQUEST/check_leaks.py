"""
Independent check that valid and test share no footage with what is trained on.

    python ZACVISIONQUEST/check_leaks.py

build_dataset.py splits by file name. This looks at the pictures instead. "Train"
here means everything a model is trained on: dataset/train and the Unity images in
dataset/sim.

  same picture     the nearest train image by raw pixels (48x48 grey thumbnails), also
                   mirrored, upside down and rotated half a turn. Catches a held-out
                   frame that is also in train, including on a fixed camera where
                   every frame has the same background.
  same footage     the nearest train image by an ImageNet network, mirrored or not.
                   Catches the same TV frame after Roboflow's flips, noise and colour
                   shifts. Fixed-camera sources are skipped here: their frames look
                   alike because the arena does.
  same scene       for fixed cameras: the background of each held-out recording (the
                   median of its frames, which removes the robots) against the
                   background of each fixed-camera train source. Catches training
                   images made from a held-out recording's background, which is what
                   the Unity renders are before unity_to_yolo.py re-pastes them.

Anything listed under "suspects" should be looked at; move its source into the
right group in build_dataset.py and rebuild.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

OUT = Path(__file__).resolve().parent / "dataset"
FIXED_CAMERA = {"NEM_08", "NEM_04", "NEM_V_WitchDoctor", "gray_fight3", "gray_fight4", "unity"}
SAME_PICTURE = 1.5       # mean grey-level difference (0-255) below which two thumbnails are the same picture
SAME_FOOTAGE = 0.915     # network similarity above which two TV frames are the same footage
SAME_SCENE = 4.0         # mean grey-level difference below which two backgrounds are the same recording session
TURNS = {"as is": lambda t: t, "mirrored": lambda t: t.flip(-1), "upside down": lambda t: t.flip(-2),
         "rotated half a turn": lambda t: t.flip(-1).flip(-2)}


def path_of(r):
    return OUT / r["split"] / "images" / r["file"]


class Images(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = cv2.imread(str(path_of(self.rows[i])))
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA)
        net_in = cv2.cvtColor(cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(net_in).permute(2, 0, 1), torch.from_numpy(thumb).float()


def background(rows, n=40):
    """The scene with the robots taken out: the median of a spread of a source's frames, as a 192x144 grey picture."""
    rows = [rows[i] for i in np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)]
    frames = [cv2.resize(cv2.imread(str(path_of(r)), cv2.IMREAD_GRAYSCALE), (192, 144), interpolation=cv2.INTER_AREA) for r in rows]
    return torch.from_numpy(np.median(np.stack(frames), axis=0)).float()


def nearest_turn(a, b):
    """(smallest mean difference between two pictures over the four turns, which turn)."""
    return min((float((a - turn(b)).abs().mean()), name) for name, turn in TURNS.items())


@torch.no_grad()
def main():
    rows = list(csv.DictReader(open(OUT / "manifest.csv")))
    # the Unity images are trained on too, but are not in the manifest
    rows += [dict(split="sim", file=f.name, group="unity") for f in sorted((OUT / "sim" / "images").glob("*.jpg"))]
    extra = [dict(split="sim_asrendered", file=f.name, group="unity as rendered")
             for f in sorted((OUT / "sim_asrendered" / "images").glob("*.jpg"))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    net.fc = torch.nn.Identity()
    net = net.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    plain, mirrored, thumbs = [], [], []
    for x, t in torch.utils.data.DataLoader(Images(rows), batch_size=96, num_workers=12):
        x = (x.to(dev).float() / 255 - mean) / std
        plain.append(torch.nn.functional.normalize(net(x), dim=1))
        mirrored.append(torch.nn.functional.normalize(net(torch.flip(x, dims=[3])), dim=1))
        thumbs.append(t.to(dev))
    plain, mirrored, thumbs = torch.cat(plain), torch.cat(mirrored), torch.cat(thumbs)

    split = np.array([r["split"] for r in rows])
    train = torch.from_numpy(np.where((split == "train") | (split == "sim"))[0]).to(dev)
    print(f"trained on: {int((split == 'train').sum())} real images, {int((split == 'sim').sum())} Unity images")
    suspects = []
    for name in ("valid", "test"):
        held = np.where(split == name)[0]
        pix, sim, pix_j, sim_j = [], [], [], []
        for i in held:
            d = torch.stack([(thumbs[train] - turn(thumbs[i])).abs().mean((1, 2)) for turn in TURNS.values()]).min(0).values
            s = torch.maximum(plain[train] @ plain[i], mirrored[train] @ plain[i])
            pix.append(float(d.min())); pix_j.append(int(train[d.argmin()]))
            sim.append(float(s.max())); sim_j.append(int(train[s.argmax()]))
        pix, sim = np.array(pix), np.array(sim)
        moving = np.array([rows[i]["group"] not in FIXED_CAMERA for i in held])
        print(f"{name}: {len(held)} images")
        print(f"  nearest train picture, grey-level difference: smallest {pix.min():.1f}, median {np.median(pix):.1f}"
              f"   (same picture would be under {SAME_PICTURE})")
        print(f"  nearest train footage, similarity (moving-camera sources): largest {sim[moving].max():.3f}, "
              f"median {np.median(sim[moving]):.3f}   (same footage would be over {SAME_FOOTAGE})")
        for k, i in enumerate(held):
            if pix[k] < SAME_PICTURE:
                suspects.append((name, rows[i]["file"], "same picture as", rows[pix_j[k]]["file"], round(pix[k], 2)))
            elif moving[k] and sim[k] > SAME_FOOTAGE:
                suspects.append((name, rows[i]["file"], "same footage as", rows[sim_j[k]]["file"], round(sim[k], 3)))

    # same scene: backgrounds of the fixed-camera recordings
    def by_group(pool, splits):
        groups = {}
        for r in pool:
            if r["split"] in splits and r["group"] in FIXED_CAMERA | {"unity as rendered"}:
                groups.setdefault(r["group"], []).append(r)
        return {g: background(v) for g, v in groups.items()}
    held_scenes = by_group(rows, {"valid", "test"})
    train_scenes = by_group(rows, {"train", "sim"})
    print(f"backgrounds of held-out recordings against backgrounds trained on   (same session would be under {SAME_SCENE})")
    for h, hb in held_scenes.items():
        for t, tb in train_scenes.items():
            d, turn = nearest_turn(hb, tb)
            print(f"  {h:18s} vs {t:18s} {d:5.1f}  ({turn})")
            if d < SAME_SCENE:
                suspects.append(("valid/test", h, "same scene as train source", t, round(d, 2)))
    for t, tb in by_group(extra, {"sim_asrendered"}).items():
        for h, hb in held_scenes.items():
            d, turn = nearest_turn(hb, tb)
            if d < SAME_SCENE:
                print(f"  not trained on, shown for comparison: {t} is the same scene as {h} ({d:.1f}, {turn}).")
                print("  That is why unity_to_yolo.py pastes the Unity robots onto a training session's arena.")

    print(f"\nsuspects: {len(suspects)}")
    for s in suspects:
        print("  ", *s)


if __name__ == "__main__":
    main()
