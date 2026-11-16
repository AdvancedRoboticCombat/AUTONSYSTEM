# SIM2REAL: measure the real robot, then make the simulator match it

Drives Nemesis alone (no opponent needed), records every command next to the
camera video and the IMU on one clock, and turns the recording into the numbers
the training simulator needs. The goal is that a drive in `auton_training`
and the same drive on the floor look the same before the fight.

It does not copy the fight system. It runs the same camera, detector,
homography, radio and PPM output the fight uses, because those are the things
being measured, and replaces only the policy with a test pilot.

```
camera -> dashboard (YOLO, homography) -> /nemesis/centroid --+
robot IMU -> /nemesis/imu ------------------------------------+--> test pilot --> /control/channels --> PPM --> Nemesis
                                                              |         (or the real policy, flown against a phantom opponent)
                                                              +--> recorder --> runs/<name>/
runs/<name>/ --> analyze.py --> the robot's numbers, a simulator replay of the same commands, suggested ranges
```

## Pieces

| File | What it does |
|---|---|
| `maneuvers.py` | The test drives: named, timed throttle and turn commands |
| `test_pilot.py` | ROS node. Plays a list of maneuvers on `/control/channels`, with the safety rules below |
| `settings.yaml` | The floor (arena, keep-out boxes, `margin_ft`), the pilot's other limits and the mixer's `half_track_ft`. Check it against the floor before a session |
| `phantom_opponent.py` | ROS node. Publishes a made-up `/opponent/centroid` so the real `saw_policy_node` can be flown with no enemy in the arena |
| `recorder.py` | ROS node. Writes one run folder (format below) |
| `fake_robot.py` | ROS node. Stands in for robot, camera and IMU: turns `/control/channels` into `/nemesis/centroid` and `/nemesis/imu` using the simulator's drive model with numbers you choose. For dry runs and tests |
| `analyze.py` | Reads a run folder: fits the robot's numbers, replays the commands in the simulator, writes the report |
| `run.sh` | Starts a session: `--dry` (fake robot, nothing leaves the PC) or the real robot |
| `runs/` | One folder per session. Not in git |

## The clock

Everything is stamped with Unix time in seconds (`time.time()`) on the one PC
that runs the whole system. Commands, positions, IMU samples and video frames
can then be lined up by that number alone.

## A run folder

`runs/<YYYYmmdd_HHMMSS>_<name>/`

| File | One row per | Columns |
|---|---|---|
| `meta.json` | | `name`, `started` (Unix s), `source` (`robot` or `fake`), `arena_ft` (`[width, height]`), `maneuvers` (the list played, with their settings: one entry per start, as fitted to the floor, with `t_start` and the position and heading it started from), `mixer` (`half_track_ft`, `tread_k`), `limits` (the safety settings), and for a fake run `truth` (the numbers the fake robot used; the drive model's and the delays' names are those of `plant.py`) |
| `commands.csv` | message on `/control/channels` | `t, left, right, blade, arm` (each -1..1, as sent). Under override these are the driver's sticks |
| `events.csv` | thing that happened | `t, event, maneuver, detail`. Events: `maneuver_start`, `maneuver_end`, `abort`, `override_on`, `override_off`, `waiting`. The override events are every change of `/nemesis/override`, noted by the recorder; the rest come from the pilot. Rows are in order of arrival: sort by `t` |
| `position.csv` | message on `/nemesis/centroid` | `t, t_stamp, x_ft, y_ft, coasting` (`t` = when it arrived, `t_stamp` = the message's own stamp, `coasting` = 1 when the tracker was extrapolating) |
| `opponent.csv` | message on `/opponent/centroid` | same columns (empty when there is none) |
| `imu.csv` | message on `/nemesis/imu` | `t, t_stamp, yaw_rad, gyro_z, accel_x, accel_y` |
| `video.mp4`, `frames.csv` | camera frame | `frame, t` (when the frame was grabbed). Written by the dashboard backend when it is started with `--record DIR`; absent in a dry run |
| `logs/` | | What each program printed. `test_pilot.log` says in words what the pilot did and what it waited for |

A flight of the policy (`run.sh --policy`) has one entry in `maneuvers` (the
policy, its settings file and the phantom), the mixer and `stale_s` of
`saw_policy.yaml`, and only override events.

Arena coordinates are the policy's: feet from a corner, x and y along two
walls, heading counter-clockwise from +x seen from above.

## Commands

The pilot thinks in `throttle` and `turn` (each -1..1) and mixes them into
treads exactly as the fight code does (`auton_training/deploy.py`):

```
tread_k = half_track_ft * YAW_RATE_MAX / V_MAX_FT
scale   = max(1, |throttle| + tread_k * |turn|)
left    = (throttle - tread_k * turn) / scale
right   = (throttle + tread_k * turn) / scale
```

It publishes `[left, right, blade, arm]` at the simulator's control rate
(every 0.03 s). Blade is held at -1 (stopped) and arm at 0, always: weapon
tests are not built.

## The test drives and what each one measures

| Maneuver | Command | Measures (name in `auton_training/config.py`) |
|---|---|---|
| `still` | nothing for 2 s | position and IMU noise |
| `step` | throttle 0 to a level, hold, back to 0. Levels 0.3, 0.6, 1.0, forward and reverse | top speed (`v_max`), how fast it gets there (`tau`, traction), reverse speed, dead zone (`deadband`), delay from command to movement (`act_lag` + `obs_lag`) |
| `spin` | turn 0 to a level, hold, back to 0, both ways, no throttle | turn rate (`yaw`, `yaw_gain`), turn acceleration (`alpha`), left/right difference |
| `arc` | steady throttle with steady turn | speed lost while turning (`bleed`), sideways slip (`grip`) |
| `straight` | steady throttle, no turn | drift to one side (`pull`) |
| `doublet` | throttle forward then back in quick succession | delay and response, cross-checked |

`python SIM2REAL/maneuvers.py` lists every maneuver, the floor it needs and the
sessions (`standard`: all of the above; `quick`: five of them).

How long and how hard each one runs is set from the free floor in front of the
robot, so it fits a small test floor as well as the 48 ft arena. The pilot
works out how far the maneuver could take a robot as fast as the simulator
allows (the top of its training ranges, 22.2 ft/s and 8.6 rad/s today, at full
speed from the first instant until 0.5 s after the command ends). A straight
drive needs that distance along its line, in a fan either side of the heading;
one that turns needs it all round. If it does not fit, the hold is shortened,
down to 0.5 s (1.5 s for `straight`). If it still does not fit, the level is
lowered, but only as far as `min_share` in `settings.yaml` allows. That is 1.0
as shipped: a full-throttle step stays a full-throttle step and the pilot
waits for the driver to make room. On a small floor set it lower.

## Safety rules of the test pilot

The floor is in `settings.yaml`. As shipped it is the 48 ft BattleBox as the
simulator has it (`auton_training/saw_walls.py`): the band along the y = 0
wall and the three wall blocks are keep-out boxes. Anything else on the real
floor (kill saws) has to be added there as a box. The robot's centre keeps
`margin_ft` (5 ft) from every wall and box.

1. It publishes nothing while `/nemesis/override` is true (the driver has the
   robot), and nothing until it has been false at least once. A maneuver that
   was running is aborted, and the pilot goes silent at once.
2. It stops when our position or the IMU is older than `stale_s`, or the
   override switch itself has been silent that long. A maneuver that was
   running is aborted: the pilot sends zero treads for `brake_s` (1 s), then
   nothing, so the robot is told to stop at once and not after the 0.5 s the
   PPM output takes to notice silence. Between maneuvers it just sends nothing.
3. Before a maneuver it checks the robot is standing still and that the
   maneuver, at the fastest speed the simulator allows, stays inside the arena
   by `margin_ft` and outside every keep-out box. If not, it waits and says
   why. When everything is right it waits another `settle_s` (3 s), which is
   the driver's time to take the robot first, and then starts.
4. During a maneuver it aborts (zero treads for `brake_s`, then nothing) when
   the robot is within `margin_ft` of a wall or a keep-out box, or will be
   within `stop_s` (0.4 s) at the speed the camera sees it moving. An aborted
   maneuver is played again from its start, `tries` (2) times in all.
5. Between maneuvers the driver puts the robot back with the sticks (override
   on, drive, override off). The pilot does not drive itself back.
6. Weapon channels stay off (weapon tests are not built).

Heading. The camera gives none, and the pre-check needs one. It starts as
`initial_heading_deg` (or `--heading`) at the moment the pilot first gets the
robot, so set the robot down facing that way. With the IMU it then follows the
IMU as `saw_policy_node` does (`deploy.HeadingEstimator`). That estimator's
check of the heading against the way the robot pulls away is armed again at
the start of every step and straight drive, so a robot set down facing the
wrong way is put right on its first step above 0.3 throttle.

### Without the IMU (`--no-imu`)

For as long as the IMU link is down. Rule 2 then ignores the IMU, "standing
still" is judged from the camera alone, and the heading comes from the
operator's starting heading and the direction of travel:

- Each time the robot pulls away from rest and covers 1.5 ft in a straight
  line, the heading becomes that line (turned round if the treads were told to
  go backwards). This uses every command on `/control/channels`, the driver's
  sticks included.
- In between, the doubt in the heading grows by the most the robot could have
  turned on the turn commands it was given. After any spin or arc, and after
  the driver has turned the robot round, the heading is unknown.
- The pre-check allows for the doubt: a wider fan either side of the heading,
  and with the heading unknown a straight drive needs its room all round. If
  it has not got that, the pilot first plays a short slow hop (`heading_probe`)
  to find the heading, if that has room all round; otherwise it waits.
- So when you put the robot back, finish by driving it straight for 2 ft or
  more, forward or backward. That tells the pilot which way it faces.

A spin on the spot cannot be measured without the IMU (the camera sees no
turning); it is still played and recorded on video.

## The two ways to use it

- **Measure the robot:** `run.sh` plays the test drives; `analyze.py` fits the numbers.
- **Check the policy:** `run.sh --policy` flies the real `saw_policy_node`
  against a phantom opponent and records it the same way; `analyze.py` replays
  the same commands in the simulator and shows where the two paths part.

```sh
SIM2REAL/run.sh --dry                     # rehearsal: fake robot, nothing leaves the PC (also --dry --policy, --dry --no-imu)
SIM2REAL/run.sh --dry --fake "--plant v_max=0.7 tau=0.15 --obs-lag 0.12"    # a fake robot with other numbers
SIM2REAL/run.sh                           # the real robot; asks before it starts anything. Not yet run on hardware
SIM2REAL/run.sh --no-imu --heading 90     # the real robot without the IMU, set down facing +y
SIM2REAL/run.sh --help                    # every option
```

The dry run includes a stand-in for the driver: when the pilot waits for room,
it takes the fake robot (override on), drives it to where the maneuver fits
and gives it back, so a whole session runs unattended.
