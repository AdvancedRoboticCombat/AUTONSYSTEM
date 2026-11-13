"""
Independent check that valid and test share no footage with train.

    python ZACVISIONQUEST/check_leaks.py

build_dataset.py splits by file name. This looks at the pictures instead:

  same picture     the nearest train image by raw pixels (48x48 grey thumbnails).
                   Catches a held-out frame that is also in train, including on
                   a fixed camera where every frame has the same background.
  same footage     the nearest train image by an ImageNet network, mirrored or
                   not. Catches the same TV frame after Roboflow's flips, noise
                   and colour shifts. Fixed-camera sources are skipped here:
                   their frames look alike because the arena does.

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
FIXED_CAMERA = {"NEM_08", "NEM_04", "NEM_V_WitchDoctor", "gray_fight3", "gray_fight4"}
SAME_PICTURE = 1.5       # mean grey-level difference (0-255) below which two thumbnails are the same picture
SAME_FOOTAGE = 0.915     # network similarity above which two TV frames are the same footage


class Images(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(OUT / r["split"] / "images" / r["file"]))
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA)
        net_in = cv2.cvtColor(cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(net_in).permute(2, 0, 1), torch.from_numpy(thumb).float().flatten()


@torch.no_grad()
def main():
    rows = list(csv.DictReader(open(OUT / "manifest.csv")))
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
    train = torch.from_numpy(np.where(split == "train")[0]).to(dev)
    suspects = []
    for name in ("valid", "test"):
        held = np.where(split == name)[0]
        pix, sim, pix_j, sim_j = [], [], [], []
        for i in held:
            d = (thumbs[train] - thumbs[i]).abs().mean(1)
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
    print(f"\nsuspects: {len(suspects)}")
    for s in suspects:
        print("  ", *s)


if __name__ == "__main__":
    main()
