# ZACVISIONQUEST: the robot detector

Trains the segmentation model the dashboard uses to find Nemesis and the
opponent. Two classes: `0 robot`, `1 nemesis`.

Run everything from the repo root with the repo environment
(`source scripts/_common.sh` or `source .venv/bin/activate; export PYTHONNOUSERSITE=1`).

| Step | Command | Result |
|---|---|---|
| 1. Clean split | `python ZACVISIONQUEST/build_dataset.py` | `dataset/{train,valid,test}` |
| 2. Unity images | `python ZACVISIONQUEST/unity_to_yolo.py --preview 80` | `dataset/sim`, and a count of Nemesis per clip |
| 3. Check it | `python ZACVISIONQUEST/check_leaks.py` | must print `suspects: 0` |
| 4. Train and compare | `python ZACVISIONQUEST/sweep.py` | `runs/sweep/results.csv` |
| Results table | `python ZACVISIONQUEST/sweep.py --report` | |
| Score any model | `python ZACVISIONQUEST/sweep.py --score MODEL.pt` | how it does on the held-out fight |
| The final number | `python ZACVISIONQUEST/sweep.py --test` | `runs/sweep/test_result.txt` |

## Fight day: a new-looking Nemesis or a new camera angle

Record about 20 minutes of the new Nemesis driving around the arena **alone**, from the fight-day
camera position, then:

    python ZACVISIONQUEST/USETHISFORFINETUNE.py VIDEO.mp4 --build-only   # check dataset/finetune_VIDEO/preview.jpg
    python ZACVISIONQUEST/USETHISFORFINETUNE.py VIDEO.mp4 --skip-build   # fine-tune, about 30 to 60 minutes

It finds Nemesis in the frames, pastes opponents (TV footage and Unity robots) next to it, on top of it
and under it, mixes in old training images, fine-tunes the best model, and scores old and new model on
the held-out end of the video and on the old fight. Pass `--orient` to match the dashboard's Orientation
(a raw camera recording is upside down: `--orient rotate-180`). It warns, and says not to use the weights,
if the robot it labelled Nemesis looks like an opponent to the current model or if the new model starts
mixing up Nemesis and the opponent. Any other robot driving in that video breaks it.

## Folders

| Folder | What it is |
|---|---|
| `Zac_vis_final_segmented_dataset/` | The original, contaminated dataset. Never modified. Its `Datasets/` holds the short Unity clips. |
| `unity_renders/` | The five three-minute Unity clips of Nemesis against Witch Doctor (hard links, no extra disk). |
| `dataset/` | The clean dataset (below). Real images are hard links to the originals. |
| `weights/` | Pretrained weights for the sweep, and SAM 2 for labelling the Unity clips. |
| `runs/sweep/` | One folder per trained model, `results.csv`, and `test_result.txt` for the winner. |

All five are in `.gitignore`. The scripts are what git keeps.

## The clean dataset

```
dataset/
  train/           real footage for training
  valid/           real footage for choosing between models
  test/            real footage looked at once, for the final number
  sim/             Unity robots pasted onto our own empty arena, training only
  sim_asrendered/  the same Unity frames as rendered. Not for training: see "Unity images"
  data.yaml                 train + sim -> valid / test
  data_real_only.yaml       train       -> valid / test
  data_sim_asrendered.yaml  only for measuring the leak described below
  data_own_camera.yaml      scores only the held-out Nemesis fight
  manifest.csv              every real image: split, source group, frame, where it was before
```

### What was wrong with the original

| Problem | Size | Fix |
|---|---|---|
| Same video in train and valid | 1,307 of 1,331 valid images came from a video that was also in train | A whole source goes to one split |
| Same fight under different file names | 6 groups of sources (a compilation and the clip it was cut from, for example) | Tied together in `SAME_FOOTAGE` in `build_dataset.py` |
| Byte-identical copies (`_dupN`, `_copyN`) | 3,580 of 13,306 files | One copy kept |
| Roboflow's augmented versions of one frame | about 3 per frame | Train keeps all; valid and test keep one |
| Label rows listed twice | 19 rows | Each kept once |
| Unity renders use the held-out fight as their background | every render | Robots re-pasted onto a training session's arena |

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

`NEM_V_WitchDoctor` is the first 80 seconds of `battle_gamma08_fight.mp4`. The
remaining 90 seconds of that video are the same fight, so they must stay out of
training too.

### Unity images

`unity_to_yolo.py` turns the Unity clips into training images. Two things in it
are easy to get wrong, and it checks both.

**The background.** Unity renders the robots over a still photo of the arena,
and that photo is a frame of the held-out fight. Trained on as they are, the
renders would show the model the exact background of every valid and test
image. So the robots are cut out and pasted onto an empty arena made from
`NEM_08`, a session that is in train anyway. The six robots parked in that
session stay in the picture with their labels. The untouched renders are kept
in `sim_asrendered/` only so the size of the leak can be measured.

**Which robot is Nemesis.** The Unity export has video and yaw only, no masks.
The outline of the robots is whatever differs from the empty arena. Nemesis is
the one with red and gold paint. While the robots touch, SAM 2 follows each of
them to divide the joint outline, and the paint has to agree with it. A frame
is skipped when the answer is not credible, for example when an outline is the
wrong size for a robot standing at that spot. The script ends with a count per
clip and these checks:

- every two-robot image has exactly one `nemesis` and one other robot
- a second opinion on identity (Nemesis is the darker robot) agrees with the paint
- for a clip with one robot, whether that robot is Nemesis
- any clip where fewer than a quarter of the frames could be labelled

Look at `dataset/sim/preview/` after a new batch. The paint rule assumes the
opponent is not red or gold; the Unity project also has red robots (Malice,
Jackpot, Tombstone), and clips with them would mostly be skipped. Exporting
instance masks from Unity itself would be exact and would replace all of this.

### Why the Unity images matter

In every real frame from our camera that is in train, Nemesis drives around
alone while six other robots sit parked in a corner. A model trained on that
alone learns that a robot out on the open floor is Nemesis, and in a fight it
calls the opponent Nemesis too. The Unity images are the only training images
with another robot moving in our arena.

## Scoring

Every model is scored on valid:

| Column | Meaning |
|---|---|
| fight | Share of the held-out fight's frames in which Nemesis and the opponent are both found, each with the right class, and nothing else is reported. Models are ranked by this. |
| nemesis, opponent | Share of those frames in which that robot is found with the right class |
| extra | Detections per frame that are not a robot, or are a second copy of one |
| mixed up | Share of frames in which a robot is found but given the other robot's class |
| own cam | Mask mAP50-95 on the held-out fight. Breaks ties in `fight`. |
| all | Mask mAP50-95 over all of valid. Mostly TV footage: it shows whether the model still sees robots it has never met. |
| ms | GPU time for one 1920x1440 camera frame |

A robot counts as found when a detection's mask overlaps its labelled mask by
half or more, at the confidence the dashboard uses (0.25).

The fight has 164 valid frames, so `fight` moves in steps of 0.006 and
differences under 0.02 are noise.

Each run keeps the epoch that does best on the fight. The library's own
`best.pt` is chosen on all of valid, which is mostly TV footage, and the two
disagree.

The optimizer is fixed in `sweep.py`. Left on automatic, the library switches
optimizer at 10,000 training steps, which this dataset straddles, so adding the
Unity images would have changed the optimizer as well as the data.
