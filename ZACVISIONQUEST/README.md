# ZACVISIONQUEST: the robot detector

Trains the segmentation model the dashboard uses to find Nemesis and the
opponent. Two classes: `0 robot`, `1 nemesis`.

Run everything from the repo root with the repo environment
(`source scripts/_common.sh` or `source .venv/bin/activate; export PYTHONNOUSERSITE=1`).

| Step | Command | Result |
|---|---|---|
| 1. Clean split | `python ZACVISIONQUEST/build_dataset.py` | `dataset/{train,valid,test}` |
| 2. Check it | `python ZACVISIONQUEST/check_leaks.py` | must print `suspects: 0` |
| 3. Unity renders | `python ZACVISIONQUEST/unity_to_yolo.py --renders DIR [DIR ...]` | `dataset/sim` |
| 4. Sweep | `python ZACVISIONQUEST/sweep.py` | `runs/sweep/results.csv` |
| Results table | `python ZACVISIONQUEST/sweep.py --report` | |

## Folders

| Folder | What it is |
|---|---|
| `Zac_vis_final_segmented_dataset/` | The original, contaminated dataset. Never modified. Its `Datasets/` holds the Unity clips. |
| `dataset/` | The clean dataset (below). Images are hard links to the originals, so it costs no disk. |
| `weights/` | Pretrained weights downloaded for the sweep, and SAM 2 for labelling renders. |
| `runs/sweep/` | One folder per trained model, `results.csv`, and `test_result.txt` for the winner. |

All four are in `.gitignore`. The scripts are what git keeps.

## The clean dataset

```
dataset/
  train/   real footage for training
  valid/   real footage for choosing between models
  test/    real footage looked at once, for the final number
  sim/     Unity renders, training only
  data.yaml              train + sim -> valid / test
  data_real_only.yaml    train       -> valid / test
  data_own_camera.yaml   scores only the held-out Nemesis fight
  manifest.csv           every image: split, source group, frame, where it was before
```

### What was wrong with the original

| Problem | Size | Fix |
|---|---|---|
| Same video in train and valid | 1,307 of 1,331 valid images came from a video that was also in train | A whole source goes to one split |
| Same fight under different file names | 6 groups of sources (a compilation and the clip it was cut from, for example) | Tied together in `SAME_FOOTAGE` in `build_dataset.py` |
| Byte-identical copies (`_dupN`, `_copyN`) | 3,580 of 13,306 files | One copy kept |
| Roboflow's augmented versions of one frame | about 3 per frame | Train keeps all; valid and test keep one |
| Label rows listed twice | 19 rows | Each kept once |

### How the split is made

- **Our camera.** `NEM_04` and `NEM_08` are the same session (same robots parked
  in the same corner), so both are in train. `NEM_V_WitchDoctor` is the only
  recorded Nemesis fight and is never in train: its first half is valid, its
  second half is test, with one second dropped between them.
- **Greyscale overhead fights.** Fight 4 is in train, fight 3 is in test.
- **TV footage and photos.** Three large linked groups and the Bilibili set are
  in train. The rest is dealt out between valid and test, whole sources at a
  time.

Valid and test both come from the same Nemesis fight, so they are independent
of train but not of each other. A second recorded fight would fix that: put
its name in `PINNED` as `"test"` and make `NEM_V_WitchDoctor` `"valid"`.

### Unity renders

The Unity export has video and yaw only, no masks. `unity_to_yolo.py` rebuilds
the masks: robots are rendered over a still photo of the arena, so the outline
is whatever differs from the empty arena, and SAM 2 follows each robot through
the clip to keep track of which is Nemesis, including while they touch. Frames
where the outline and the tracker disagree are skipped. Run it with
`--preview 60` and look at `dataset/sim/preview/` before trusting a new batch.

Exporting instance masks from Unity itself would be exact and would replace
all of this.

## Scoring

Every model is scored on valid:

| Column | Meaning |
|---|---|
| score | mask mAP50-95 over both classes, all of valid. Models are ranked by this. |
| nemesis | mask mAP50-95 of the `nemesis` class |
| own cam | mask mAP50-95 on the held-out Nemesis fight alone |
| ms | GPU time for one 1920x1440 camera frame |

Valid has 465 images, so differences under about 0.01 are noise.
