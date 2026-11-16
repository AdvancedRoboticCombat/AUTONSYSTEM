# Ideas not built yet

## Blind replay: drive the simulator's commands with no camera

Put the robot on a flat 48 x 48 ft floor with no camera. The simulator (or the
policy flying in the simulator) produces the commands, the real robot gets the
same commands, and nothing tells the PC where the real robot is. Then compare
where the simulator says it should be with where it physically ends up.
Discussed 2026-11-14; not coded.

**What it shows.** How fast the real robot drifts away from the simulated one.
Every small error adds up over time:

- Speed: the simulator's old nominal top speed was 22.6 ft/s and the fight
  suggested about 15.5. At full throttle for 2 s that is 45 ft simulated
  against about 31 ft on the floor.
- Turning: 5% less turn than simulated on a 180 degree turn leaves the robot
  9 degrees off, which is 3 ft to one side after another 20 ft.

**The useful score.** After calibration: how many seconds can the robot drive
blind before it is 1 ft off the simulated path?

**Limits.**

- Short runs only: 2 to 5 seconds, reduced throttle first, driver on the
  override. A robot that does not know where it is reaches a wall in about 3 s
  at speed. A whole fight replayed blind means nothing after the first seconds.
- Something has to measure where it ends up. By eye: tape a mark where the
  simulator says it stops and measure the miss. Better: a phone filming from
  above. Without either, a speed error, a turn error and a delay all look the
  same.
- The floor matters. On anything but the arena's steel floor, top speed and
  delay carry over; turning and slip mostly do not.

**How it fits.** It is the comparison `analyze.py` makes, without the camera:
the camera separates the causes, blind replay gives the total. Two uses:

1. A quick check wherever there is no camera rig (straight runs and a couple
   of turns, with a tape measure and a phone).
2. The final exam after calibration: fit the numbers with the camera, then
   drive blind and see how long the robot stays on the simulated path.

**To build it.** It is the dry run (`run.sh --dry`, fake robot + pilot or
policy) with the commands also going out to the real robot through the PPM
output, plus hard limits on run length and throttle.

## Heading from the video

Read the robot's heading from its outline in the camera picture. That would
measure spin rate and turn acceleration without the IMU, which is the one
thing the no-IMU mode cannot do. Not built or tested.
