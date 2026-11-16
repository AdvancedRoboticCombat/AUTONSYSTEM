# Camera, lens and arena notes

What was learned on 2026-11-13 from the recorded Witch Doctor fight
(`~/battle_gamma08_fight.mp4`, 1920x1440, 170 s). Keep this up to date when
any of it is measured again.

## Two calibrations, with different lifetimes

| | What it fixes | Redo it when |
|---|---|---|
| Lens calibration | The bend the lens puts in the picture | The lens, its focus or zoom, or the capture resolution / binning changes. Otherwise once. |
| Floor mapping (homography) | Where the camera is and where it points | Every time the camera is moved or re-aimed. Every event. |

## Lens: to do before the next event

- The fight video shows strong barrel distortion: a point at the left or right
  edge of the frame sits about 37% closer to the centre than it should.
- The saved calibration (`nemesis_dashboard/vision/camera_calibration.npz`,
  1920x1440, reprojection error 1.4 px) does **not** match that. It describes a
  mild bend the other way. It is either from another lens setting or a poor
  calibration. Do not trust it for this camera as it was on the day of the fight.
- **To do:** redo the checkerboard calibration with the lens and settings that
  will be used (README, "Camera and arena setup", step 2), and aim for an error
  well under 1 px.
- **To do:** make the dashboard undistort a pixel before the floor mapping. In
  fight-video mode it applies a plain homography today.

For the fight video only, the lens was estimated from the picture itself
(assuming the lens centre is the image centre):

```
c = (959.5, 719.5)      r = |p - c| / 1200      k = -0.42   (plausible -0.38 to -0.46)
undistorted pixel = c + (p - c) / (1 + k * r^2)
```

Floor mapping on undistorted pixels, pixels to feet:

```
[[-5.918444234e-02, -9.438974447e-02,  8.139830809e+01],
 [ 6.048621905e-04, -2.038704572e-01,  1.200437497e+02],
 [ 2.046843624e-05, -3.778955752e-03,  1.0            ]]
```

| | Typical error | Worst |
|---|---|---|
| Plain homography (dashboard today) | 0.75 ft | about 2 ft, near the side walls and close to the camera |
| Lens corrected, then homography | 0.1 to 0.2 ft | 0.25 ft at the measured marks |

## Arena floor: what is 4 ft and what is not

- The small yellow floor dots are on a 4 ft grid. They are the scale. With
  them as 4 ft the far wall measures 47.4 ft against 48.
- The painted blocks are **not** 4 ft. Centre blue and orange blocks: about
  6 ft wide by 7 ft deep each. Blue and orange start squares: about 8 ft wide.
- The yellow killsaw outlines are about 1.5 by 2.3 ft.
- Only the two far corners are in view. The nearest 5 to 7 ft of the floor is
  below the frame, and the far-right corner is cut off by a diagonal bumper.
- `nemesis_dashboard/fight_video_homography.json` holds 137 points of the 4 ft
  grid for this video. `ZACVISIONQUEST/runs/fight_video/homography_check.jpg`
  shows the grid drawn over the arena.

## Axes used for the fight video

(0, 0) is the far-left floor corner, x runs right along the far wall, y runs
toward the camera, both in feet. Seen from above that is clockwise. The policy
wants counter-clockwise, so one axis has to be mirrored (for example
y' = 48 - y) before these positions feed the robot.

A robot's position is taken near the bottom of its outline, which is about a
foot closer to the camera than its centre.

## Speeds measured in the fight

First 85 s, distance covered over half a second, lens-corrected positions.

| | Typical | 90% of the time below | Fastest seen |
|---|---|---|---|
| Nemesis | 5.7 ft/s | 10.7 ft/s | 15.5 ft/s (10.6 mph) |
| Witch Doctor | 5.2 ft/s | 11.7 ft/s | 17.6 ft/s (12.0 mph) |

"Fastest seen" is the fastest each robot happened to go in one fight, not a
measured top speed. The simulator's nominal top speed for Nemesis is 22.6 ft/s,
so the fight reached 0.68 of it.

Changed in `auton_training` because of this (2026-11-13 and 14):

| Setting | Was | Now |
|---|---|---|
| `SAW_DR["v_max"]` (config.py), our top speed as a share of `V_MAX_FT` | 0.50 to 1.10 (11.3 to 24.9 ft/s) | 0.39 to 0.98 (8.8 to 22.2 ft/s) |
| `DR_V_MAX` (config.py), same for the classic rules | 0.65 to 1.05 | 0.49 to 0.88 |
| `EN_MPH_MIX` slow group (saw_field.py), enemy top speed | 9 to 12 mph | 9 to 13 mph (there was a gap between 12 and 13) |

The simulator's default for a number is the middle of its range, so both
top-speed ranges are now centred on the measured 15.5 ft/s and keep their old
width. `V_MAX_FT` itself (22.64 ft/s) was left alone on purpose: the policy's
speed inputs and the tread mixer are scaled by it, so changing it would break
every trained checkpoint.

Things to keep in mind:

- 15.5 ft/s is the fastest seen in one fight. If the robot's real top speed is
  above 22.2 ft/s the new range no longer covers it.
- Policies trained before this change never saw a robot slower than 11.3 ft/s.
- Scores from evaluations run before this change were made under the old ranges.
- Top speed is the only number measured so far. Everything else in the
  simulator is still an estimate until `SIM2REAL` has been run on the robot.
