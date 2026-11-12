"""
dodge_master configuration.

Everything tunable lives here: arena, the robot's nominal physics (from the
ROS2 node), the ranges those physics are randomized over, sensor model,
fight geometry, rewards, and the sizes of the policy network.

Nominal robot parameters (ROS2 node):
    v_max_mps        = 6.9   m/s   -> 22.64 ft/s
    a_lat_max_mps2   = 30.0  m/s^2 -> 98.43 ft/s^2  (controller's centripetal limit)
    yawrate_max_rps  = 7.5   rad/s
    mass             = 250   lb
These are estimates. Anything the real robot does not know exactly is drawn
per episode from the DR_* ranges below, so the policy has to work across all
of them instead of memorising one set of numbers.
"""
import math
import os
from pathlib import Path

import torch

# ─── Device ───────────────────────────────────────────────────────────────
# DODGE_DEVICE=cpu forces a device, so a tool can run beside a training job
# without touching the GPU.
if os.environ.get("DODGE_DEVICE"):
    DEVICE_NAME = os.environ["DODGE_DEVICE"]
elif torch.cuda.is_available():
    DEVICE_NAME = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE_NAME = "mps"
else:
    DEVICE_NAME = "cpu"
DEVICE = torch.device(DEVICE_NAME)

# ─── Paths ────────────────────────────────────────────────────────────────
# Anchored to this file, so every script finds the arena and the checkpoints
# whatever directory it is launched from.
ROOT     = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "ppo_models"
EVAL_DIR = ROOT / "eval"

# ─── Units ────────────────────────────────────────────────────────────────
MPH2FT = 1.4667          # mph  -> ft/s
MPS2FT = 3.28084         # m/s  -> ft/s
FT2M   = 0.3048
G_FT   = 32.174          # ft/s^2

# ─── Arena & timing ───────────────────────────────────────────────────────
ARENA_FT      = 48
ROBOT_FT      = 4
WALL_MARGIN   = 0.25                         # gap between robot edge and wall
WALL_CLAMP_IN = ROBOT_FT / 2 + WALL_MARGIN   # closest the robot centre gets to a wall

DT            = 0.01                         # physics step (100 Hz)
PHYS_PER_CTRL = 3                            # physics steps per policy decision
CTRL_DT       = DT * PHYS_PER_CTRL           # 0.03 s (~33 Hz)
MAX_STEPS     = 2000                         # physics steps per episode (20 s)

HAZ_PATH      = ROOT / "battlebot_models" / "hazards.npy"

# ─── Nominal robot physics ────────────────────────────────────────────────
ROBOT_MASS_KG  = 113.4
V_MAX_FT       = 6.9 * MPS2FT                # 22.64 ft/s
A_LAT_MAX_FT   = 30.0 * MPS2FT               # 98.43 ft/s^2, yaw-rate limit a_lat / |v|
YAW_RATE_MAX   = 7.5                         # rad/s
A_LONG_MAX_FT  = 23.488 * MPS2FT             # 77.06 ft/s^2 longitudinal traction limit

# Yaw acceleration from drive torque over a solid-box inertia estimate.
_TRACK_WIDTH_M = 2.0 * FT2M
_INERTIA_KGM2  = ROBOT_MASS_KG * (1.22**2 + 1.22**2) / 12.0
ALPHA_MAX_RPS2 = (ROBOT_MASS_KG * 30.0 / 2.0) * (_TRACK_WIDTH_M / 2.0) / _INERTIA_KGM2   # ~18.4

# Lateral grip (magnets). Past the static limit the robot slides until the
# demand falls back under the kinetic limit.
GRIP_STATIC_FT     = 2.0 * G_FT
GRIP_KINETIC_FT    = 0.6 * G_FT
DRIFT_RECOVERY_TAU = 0.15                    # s, residual slide decay once grip returns

WALL_SCRUB       = 0.5    # speed kept per physics step while driving into a wall
BODY_RESTITUTION = 0.3    # body-on-body bounce

# ─── Domain randomization: our robot's plant (per episode) ────────────────
# (lo, hi). Multipliers on the nominal value unless the name ends in _S.
DR_V_MAX     = (0.65, 1.05)    # true top speed
DR_GRIP      = (0.60, 1.15)    # floor grip: scales traction and both slip limits
DR_YAW       = (0.80, 1.05)    # yaw rate actually reached at full command
DR_ALPHA     = (0.70, 2.00)    # yaw acceleration (inertia is a rough estimate)
DR_BLEED     = (0.20, 0.50)    # share of top speed lost at full yaw rate (estimate 0.35)
DR_TAU_S     = (0.03, 0.12)    # drive actuator time constant (estimate 0.05)
DR_ACT_LAG_S = (0.02, 0.07)    # command latency
DR_OBS_LAG_S = (0.03, 0.10)    # sensing latency
# Drive imperfections
DR_YAW_GAIN  = (0.85, 1.10)    # yaw rate delivered per unit of turn command
DR_PULL      = (-0.06, 0.06)   # mismatched drive sides: yaw picked up at full throttle, as a share of the yaw cap
DR_DEADBAND  = (0.00, 0.06)    # throttle travel around zero that does nothing

# Mid-fight wear: in this share of episodes the robot changes under the policy
# at a random moment (a hit, a hot motor), so it has to keep adapting instead
# of identifying the robot once at the start.
WEAR_P        = 0.25
DR_WEAR_SPEED = (0.80, 0.95)   # top speed kept
DR_WEAR_PULL  = (-0.08, 0.08)  # extra pull to one side

# ─── Sensor model (per episode noise levels) ──────────────────────────────
# The policy only ever sees what the real robot has: its own pose, the
# enemy's position, its own gyro and wheel speed. No enemy heading.
DR_POS_NOISE_FT    = (0.02, 0.20)   # own position, white, 1 sigma
DR_HDG_NOISE       = (0.005, 0.04)  # own heading, rad
DR_EN_POS_NOISE_FT = (0.03, 0.30)   # enemy position
DR_DROPOUT_P       = (0.00, 0.05)   # chance a position frame is missed (last one is held)
# Errors that persist from frame to frame, which averaging cannot remove:
# a calibration offset, a heading that creeps, a tracker that sits off-centre.
DR_POS_BIAS_FT     = (0.00, 0.30)   # slowly wandering offset in our own position, 1 sigma
DR_EN_POS_BIAS_FT  = (0.00, 0.50)   # and in the enemy's (it comes from a separate tracker)
DR_HDG_BIAS        = (0.00, 0.05)   # and in our heading, rad
BIAS_TAU_S         = 4.0            # how slowly those offsets wander
DR_OUTLIER_P       = (0.00, 0.01)   # chance per frame that the enemy fix jumps
OUTLIER_FT         = 3.0            # by up to this much
LAG_JITTER_STEPS   = 1              # frame-to-frame variation of the sensing delay, in physics steps
DR_SPEED_GAIN      = (0.93, 1.07)   # wheel-odometry scale error
SPEED_NOISE_FT     = 0.4            # wheel-speed noise, ft/s
GYRO_NOISE         = 0.05           # yaw-rate noise, rad/s
ENEMY_VEL_WINDOW   = 3              # control steps the enemy-velocity estimate is differenced over

# Obstacle rays: distance to the nearest wall or hazard along 8 body-frame
# directions, so the policy does not have to memorise the arena.
N_RAYS       = 8
RAY_RANGE_FT = 12.0
RAY_STEP_FT  = 0.75

# ─── Fight geometry (weapon rules) ────────────────────────────────────────
OUR_WEAPON_REACH     = 0.5    # tip protrudes this far beyond the body edge
OUR_WEAPON_TIP_RANGE = 0.6    # tip collision radius
WIN_HIT_MIN_FWD      = 2.0    # ft/s forward for the front tip to do damage
REAR_HIT_MIN_REVERSE = 3.0    # ft/s in reverse for the rear tip

# True: touching a wall ends the episode as a loss for us (the old rule).
# False: walls are solid for both robots and cost a penalty, like a real arena.
OUR_WALL_CONTACT_ENDS_EPISODE = False

# ─── Rewards ──────────────────────────────────────────────────────────────
# Outcomes are worth about +-1; everything dense is small next to them.
GAMMA        = 0.995
GAMMA_SAW    = 0.998   # saw rules: a 3 minute fight is 6,000 control steps and a swing pays off seconds later
R_WIN        = 1.0     # our weapon lands cleanly
R_LOSS       = 1.0     # their weapon lands cleanly
R_CLASH      = 1.0     # weapon tips meet (we come off worse against a spinner)
R_MUTUAL     = 0.5     # both weapons land at once
R_HAZARD     = 1.0     # we drove into a hazard (or a wall, if that rule is on)
TIME_COST_PER_S  = 0.04   # steady pressure to finish the exchange; a timeout is not terminal
R_WALL_HIT       = 0.30   # x (impact speed / V_MAX) when we run into a wall
R_WALL_TOUCH_PER_S = 0.20 # while we stay pressed against one
R_ACTION_RATE    = 0.0002 # x |change in action|^2 per control step, keeps commands smooth

# Potential-based shaping (gamma*phi' - phi): guides exploration without
# changing which policy is best, so it cannot be farmed.
K_FLANK     = 0.30     # combat: in front of their weapon is bad, on their side with our weapon aimed is good
FLANK_FAR   = 18.0     # ft, shaping fades in below this range
FLANK_NEAR  = 6.0      # ft, and is at full strength below this
K_APPROACH  = 0.50     # drills: close in on the target with the weapon pointed at it

# ─── Policy network ───────────────────────────────────────────────────────
OBS_DIM = 16 + N_RAYS   # see environment.Observer.observe for the layout
ACT_DIM = 2             # throttle, turn
HIDDEN  = 512           # trunk width and LSTM size

# ─── Rulesets ─────────────────────────────────────────────────────────────
# "classic"  the original duel: a weapon tip at each end of both robots, and
#            the first touch ends the fight. Checkpoints are named ZACAI_*.
# "saw"      our real weapon (see saw_env.py): a saw on an arm that is a
#            vertical spinner behind the robot while parked and a hammer saw
#            over the front when fired. Nothing ends a fight but the clock.
#            Checkpoints are named SAW_*.
RULES = ("classic", "saw")
DEFAULT_NAME = {"classic": "ZACAI", "saw": "SAW"}

# ─── Saw rules: the hammer saw ────────────────────────────────────────────
# None of this has been measured on the robot yet (docs/assumptions.md). The
# DR_* ranges are redrawn every episode; replace them with measurements as
# they come in.
SAW_OBS_DIM = OBS_DIM + 20    # + previous arm and blade commands, arm position, blade speed, own half length and width (30),
                              # then swings left, time left, gate open, the opponent card (kind, size) and the opponent
                              # axis and nose readings (44). The layout is in saw_env.SawEnv._observe.
SAW_ACT_DIM = 4               # throttle, turn, arm (fire / park), blade speed

# Our robot's plant and sensors under the saw rules. Wider than the classic
# ranges above wherever nobody could say what the real robot does: whether it
# has magnets, how fast it is, how the speed controllers follow a command, how
# late the camera and the radio are. Keys are the names in environment.RANGES.
SAW_DR = dict(
    v_max=(0.50, 1.10),        # top speed was never measured
    grip=(0.30, 1.15),         # 0.3 to 0.5 is rubber on steel with no magnets
    yaw=(0.55, 1.10),          # treads scrub when they turn
    alpha=(0.30, 2.00),
    bleed=(0.05, 0.50),        # scrub only: what the treads cannot deliver at once is modelled on its own (DR_TREAD_SHARE)
    tau=(0.03, 0.40),          # a speed controller on duty cycle follows far more slowly than one holding rpm
    act_lag=(0.02, 0.12),      # the command goes out over a radio
    obs_lag=(0.03, 0.14),      # camera, detector, homography, then the same radio
    yaw_gain=(0.75, 1.15),
    pull=(-0.08, 0.08),
    deadband=(0.00, 0.08),
    pos_noise=(0.03, 0.35),    # "a few inches off"
    en_pos_noise=(0.05, 0.45),
    pos_bias=(0.00, 0.40),
    en_pos_bias=(0.00, 0.60),
    hdg_noise=(0.005, 0.06),
    hdg_bias=(0.00, 0.20),     # the camera gives no heading (docs/engine_breakdown.md): it will be a gyro-based estimate that wanders
    dropout=(0.00, 0.08),
)
DR_HDG_KICK      = (0.00, 0.30)   # rad the heading estimate jumps (1 sigma) on a full-strength hit, before it settles again
DR_REVERSE       = (0.75, 1.00)   # top speed in reverse as a share of forward (the vert is used backing up)
DR_CMD_DROP_P    = (0.00, 0.03)   # chance per control step that a command is lost on the radio; the last one stands
DR_TRACK_LOSS_HZ = (0.00, 0.08)   # times per second the camera loses a robot ...
TRACK_LOSS_S     = (0.15, 1.50)   # ... for this long; the tracker goes on reporting where it last saw it
TRACK_LOSS_WHO   = (0.60, 0.25)   # share of those losses that are the enemy only, and us only; the rest are both
TRACK_LOSS_TOUCHING = 5.0         # times likelier while the robots are in contact and their outlines merge
NO_ARM_READING_P   = 0.70         # share of fights with no arm reading (it reads -1): the telemetry firmware has no arm sensor
NO_BLADE_READING_P = 0.50         # and with no blade speed reading, drawn separately: it carries one speed controller's rpm.
                                  # saw-v8: 0.60 and 0.30 before. The robot as wired today has neither reading.
DR_TREAD_SHARE   = (0.65, 0.85)   # half the track width as a share of our half width (a 0.92 m track on a 4 ft body is 0.755).
                                  # Each tread has a top speed, so throttle + turn beyond it are scaled back together.

# Robot sizes, ft from the centre to the edge. Ours is not settled and theirs
# differ from fight to fight, so both are drawn per fight. The policy is told
# its own size and has to cope with theirs. (The classic rules: 2 by 2.)
DR_OUR_HALF_LEN  = (1.50, 2.25)
DR_OUR_HALF_WID  = (1.40, 2.10)
DR_EN_HALF_LEN   = (1.25, 2.50)
DR_EN_HALF_WID   = (1.25, 2.50)

ARM_SWITCH       = 0.33           # arm command above +this fires, below -this parks, in between keeps doing what it was
ARM_PARKED       = 0.08           # arm travel (0 parked behind, 1 down over the front) under which the saw is a vert
ARM_DOWN         = 0.92           # and over which the hammer has landed
# saw-v7. The saw cuts as it comes down: for this long after the arm lands, and
# as hard as the swing was long. A saw that is already down is not a weapon.
# Under saw-v6 it was, at the worth of a hit from above, and by update 320 the
# policy laid the saw on the floor in front and drove it into them: 62% of its
# blade damage came from bites with the arm down for over a second, 8% of its
# bites with the arm landing on them. The team's technique is "charge up, hit,
# then back up and try again".
STRIKE_S         = 0.10
HAMMER_REACH     = 0.5            # ft beyond the front edge where the saw comes down (nominal)
DR_HAMMER_REACH  = (0.4, 1.0)     # saw-v8: drawn per fight, because nobody has measured it and it decides what can be hit
DR_ARM_FIRE_S    = (0.28, 0.45)   # swing time, back to front: 0.35 s on the robot (a motor through a Belleville slip clutch)
DR_ARM_RETRACT_S = (0.30, 0.70)   # and front to back (not measured)
# The robot is set up for a fight as a hammer saw or as a vert, never both, and
# the team runs it as a hammer saw only ("make it only hammer", 2026-11-10).
# Parked, the blade then touches nothing: the hammer is the whole weapon.
VERT_SETUP_P     = 0.0            # share of fights set up as a vert instead: arm locked, the parked blade is the weapon
# Full swings the arm has in it over a 3 minute fight, as (share of fights, range). The team said "probably like
# 10". saw-v8: most fights get 8 to 12, and the rest more, in case 10 is not a hard limit. It was a flat 10 to 30
# under saw-v7, and the policy trained on twice the swings it will have. A shorter fight gets its share, never
# less than one. The drills do not count swings at all.
ARM_SWINGS_3MIN  = ((0.60, (8.0, 12.0)), (0.25, (12.0, 20.0)), (0.15, (20.0, 30.0)))
ARM_JAM_P        = 0.0            # share of fights in which the arm is dead and stays parked. It was 0.20 while a dead
                                  # arm still left the vert. On a hammer saw it leaves nothing, and the driver takes over.
ARM_JAM_P_DRILLS = 0.0            # (the drills had it at 0.50, to make the policy practise the vert)
# The blade is 15.5 in across, weighs 17 lb and turns 7,696 rpm, reached in
# 1.68 s. As a solid disc that is 49 kJ and 120 N m s of angular momentum: a
# real spinner, not a thin saw.
DR_BLADE_TAU_S   = (0.45, 0.95)   # blade spin-up time constant: 1.68 s to full speed is 2 to 3.5 of these
DR_BLADE_KEEP    = (0.20, 0.50)   # share of its speed the blade keeps through a bite (not measured): a hit has to be
                                  # charged again, see DR_BITE_FLOOR
DR_BLADE_SAG     = (0.0, 0.6)     # share of spin-up torque lost at full drive throttle (the drive and the saw share a battery)
DR_BLADE_GYRO    = (0.30, 0.70)   # share of yaw rate lost at full blade speed. Turning at w rad/s the blade rolls the robot
                                  # with 120 w N m, and 250 lb on a half track of 0.28 to 0.54 m holds a tread down with 310
                                  # to 600 N m: one side lifts at 2.6 to 5 rad/s, a third to two thirds of the full turn rate.
# The BattleBots design rules hold any external moving part to 370 ft/s at its
# tip, and at 7,696 rpm this blade's is 520. Asked whether that applies to
# this robot the team said "maybe", so each fight draws the share of 7,696 rpm
# the blade is allowed. Held to the rule it turns 5,470 rpm and carries
# 24.5 kJ, not 48.5. Blade speed in the simulator is a share of that top.
DR_BLADE_TOP     = (0.71, 1.00)   # share of 7,696 rpm the blade tops out at
BLADE_MIN_BITE   = 0.25           # below this share of full speed the saw does not cut
BLADE_FLOOR_KEEP = 0.5            # share of blade speed kept when the hammer comes down on the floor
BLADE_NOISE      = 0.03           # blade-speed telemetry noise, share of full speed
ARM_NOISE        = 0.02           # arm-position sensor noise, share of travel

# What the scripted opponents carry, one column per kind. A vert is a blade
# on the nose with forks beside it. A drum is the same across the whole
# front. A horizontal spinner is a bar or disc that reaches well past the nose
# and often past the front corners; it hits harder, throws its target
# sideways, kicks back at its owner, and takes long to spin back up.
# The numbers are set from the heavyweight field (docs/decisions/0004, "What
# the field looks like"): the mix is the spinners of the last two televised
# seasons; power is stored energy over our blade's 49 kJ (verts 25 to 70 kJ,
# drums 10 to 55, horizontals 30 to 75 with shells near 95); spin-up is 3 s for
# a drum, 5 for a big vert, 8 for a big horizontal, which is about three time
# constants. Most of those figures are builders' and fans', not measured.
# saw-v6: the field's mix is 46 / 24 / 30. Half are horizontals here because
# the first robot we meet is one: Traves T., a 75 to 78 lb, 36 in bar at the
# very front of a long thin frame, 250 mph at the tip (about 74 kJ, 1.5 of
# ours), slow to spin up. That is inside the horizontal column below.
#                    vert            drum            horizontal
EN_WEAPON_SHARE  = (0.30,           0.20,           0.50)
DR_EN_POWER      = ((0.50, 1.50),   (0.25, 1.20),   (0.60, 1.80))   # damage per bite at full speed, ours = 1
DR_EN_TAU_S      = ((1.0, 2.2),     (0.7, 1.5),     (1.8, 3.5))     # spin-up time constant
# A blade on a cross-wise axle rolls the robot when it turns, theirs as much as
# ours (DR_BLADE_GYRO): a tread lifts once turn rate x angular momentum passes
# weight x half the track. Angular momentum is 2 x energy / spin rate. At a
# 250 mph tip a vert's 8 to 12 in disc turns at 370 to 560 rad/s, so 25 to 70 kJ
# is 90 to 380 N m s against our 120: a tread lifts at 0.8 to 6.7 rad/s of a
# 5.6 to 8.6 rad/s top rate. A drum is small and fast (20 to 175 N m s). A
# horizontal's axle is upright and turning costs it nothing. Estimates, not
# measured on anybody's robot.
DR_EN_GYRO       = ((0.35, 0.80),   (0.00, 0.45),   (0.0, 0.0))     # share of yaw rate lost at full blade speed
DR_EN_KEEP       = ((0.35, 0.65),   (0.50, 0.80),   (0.15, 0.45))   # share of speed kept through a bite
EN_RECOIL        = (0.3,            0.3,            0.6)            # share of the throw that comes back at them
DR_EN_NOSE_FT    = ((0.3, 0.6),     (0.2, 0.4),     (0.9, 1.6))     # how far the blade reaches past the front edge
DR_VERT_WIDE     = (0.45, 0.90)     # half-width of what a vert's front hits with, as a share of the body's: the blade is
                                    # narrow, but the forks beside it feed whatever they catch into it
DR_DRUM_WIDE     = (0.65, 0.95)     # half-width of a drum, as a share of the body's
DR_HORIZ_HUB     = (0.3, 1.1)       # where a horizontal's hub sits, as a share of the way from centre to front edge
EN_BLADE_FT      = 0.55             # contact radius of a vert or drum blade
EN_FRONT_DISCS   = 7                # contact discs across a vert's or a drum's front: close enough that nothing fits between
# A spinner's weapon side is off limits. Our blade scores nothing on the
# front of their body, on its front corners, or anywhere near what their
# blade sweeps; if their blade is cutting, ours takes the hit. The margin is
# how far back from their front edge, and how far out from their blade, that
# holds: a real blade is wider than a point on a plan, and the camera is not
# exact about where either robot is.
DR_FRONT_MARGIN_FT = (0.2, 0.6)     # 0.4 to 1.0 under saw-v5; the team asked for looser rules with the hammer only
# Our own forks are not on the hammer end (the team, 2026-11-10), so nothing
# of ours is armoured against a spinner: its blade does a full hit wherever it
# lands on us.
DR_ENEMY_POWER   = (0.25, 1.80)     # the whole spread of DR_EN_POWER, for the tools that sweep it
EN_BLADE_START   = (0.2, 1.0)       # their blade speed as a fight opens: mostly up to speed, sometimes still spinning up

# Damage, in units of one full-speed vert hit
DMG_VERT         = 1.0            # rear saw bite, x blade speed^2
DMG_HAMMER_SAW   = 1.5            # saw bite from above, x blade speed^2. It is the same blade with the same energy
                                  # as the vert's, it lands on top armour, the thinnest on any robot, and the floor
                                  # is behind the target, so none of it goes into throwing them. An estimate (it was
                                  # 0.6 when the saw was taken for a thin blade).
DMG_HAMMER_BLOW  = 0.2            # the arm landing on them, blade or no blade
DMG_CLASH        = 0.5            # saw meets saw (a self-play copy): each side takes this x the other's blade energy.
                                  # Against a spinner there is no exchange: see DR_FRONT_MARGIN_FT.
REBITE_S         = 0.40           # a weapon held in contact bites this often
# A hit has to be charged. What a bite is worth is the energy the blade
# carries over what armour shrugs off: with a floor of 0.4, a blade at 70% of
# its speed has half its energy and does a sixth of a full hit, and one below
# 63% does none. Every bite takes the blade's speed all the same, so a blade
# tapped on a target again and again never gets back to where it counts. The
# way to hurt is to spin up, hit, back off and do it again. The same holds
# for their blades on us.
DR_BITE_FLOOR    = (0.30, 0.50)   # share of a blade's full energy that does no damage; drawn for each robot's armour
BLADE_ON_BIAS    = 1.5            # a new saw policy starts out asking for the blade (the action head's bias). Left at 0 it
                                  # asks for half speed on average, where no bite is worth anything and nothing teaches it better.
CLASH_KEEP       = 0.2            # share of blade speed left after it meets a weapon, or a spinner's front

# What a hit does to the robot that takes it
KNOCK_VERT_FT_S  = 12.0           # speed a full vert hit throws them at
KNOCK_SPIN_FT_S  = 22.0           # speed a full hit from a spinner throws us at (10 before saw-v9: the team
                                  # watched the replays and a bar's hits did not look like a bar's hits)
DMG_SPINNER      = 1.0            # a spinner's bite on us, x blade energy (the team asked for more throw, not more damage)
HORIZ_SIDEWAYS   = 0.8            # share of a horizontal's throw that goes along its blade's travel (the rest pushes away)
KNOCK_CLASH_FT_S = 5.0            # each robot, apart, after weapon meets weapon
KNOCK_RECOIL     = 0.3            # share of a hit's throw that comes back at the robot that landed it
KNOCK_YAW        = 4.0            # rad/s of spin from a full hit square on a corner
STUN_S           = 0.40           # seconds a full hit leaves the robot without drive
STUN_TRACTION    = 0.25           # share of traction left during that time
DR_SQUEEZE_YAW   = (0.3, 0.8)     # share of yaw authority left while jammed between a wall and the other robot
SQUEEZE_GAP_FT   = 0.25           # a robot this close to a wall with the other robot on it counts as jammed
WALL_FRICTION    = 0.3            # along-wall speed lost per unit of speed into the wall

# Hazards, scoring and starts under the saw rules
HAZARD_DMG_PER_S = 1.0            # damage per second while any part of our body is over a hazard
HAZARD_DMG_ENTER = 0.3            # and on driving onto one
# Rewards. A fight lasts the full 20 s here, about ten times a classic one,
# so everything charged per second is set to keep a fight's total near +-1:
# winning by four full hits is worth +1.
R_DAMAGE         = 0.25           # reward per unit of damage dealt
# The team, 2026-11-10: "i dont think it NEEDS to win, its more like we want
# to get some attacks in and get proof of concept". The scorecard still counts
# a unit taken as a unit taken. The reward counts what their weapon does to us
# at this share of it: going in for a hit is what gets a robot hit, and at
# equal weight the run before learned to stay out of reach as the drivers got
# better (engaged 75% of the time at p = 0.40, 54% at p = 0.80).
R_TAKEN_SHARE    = 0.5
R_FOUL           = 0.01           # x blow, charged when a swing lands on the floor or on their weapon side. Small on
                                  # purpose: an average swing has to stay worth taking, or the policy stops swinging.
ATTACK_IN_DMG    = 0.5            # blade damage in one fight that counts as having got an attack in. One fully charged
                                  # hit does 1.5, or 0.75 in a fight where the blade is capped at 71% of its speed
# saw-v9, the team, 2026-11-12: "after each hit [get] as far away as possible and reposition" instead of
# staying on them, and "moving as fast as possible literally at all times".
RETREAT_S        = 2.5            # seconds after a swing lands (on anything) in which backing off is paid
RETREAT_FT       = 12.0           # ... out to this far from them (PASSIVE_FT is 14: backing off is not hiding)
R_RETREAT_PER_FT = 0.01           # per foot of distance gained in that time (and charged per foot lost)
R_RAM_PER_S      = 0.10           # charged while our body is on theirs in that time
R_SLOW_PER_S     = 0.01           # x (1 - ground speed / top speed), charged all fight unless we have been thrown
R_SAW_WALL_HIT   = 0.10           # x (impact speed / V_MAX) when we run into a wall
R_SAW_WALL_PER_S = 0.05           # while we stay against one
PTS_MARGIN       = 0.25           # points a fight has to be ahead by at the clock to count as won
SPAWN_CLEAR_FT   = 4.3            # start at least this far from a hazard, cell to cell. The largest body reaches 3.1 ft
                                  # from its centre, and at 4.0 its corner could open a fight 0.02 ft over one.
# A real fight is three minutes and is decided on aggression as much as on
# damage. Training mixes fight lengths so the policy's memory has run a whole
# one. Keeping away from the opponent is charged by the second, in the reward
# and on the scorecard alike, at a rate that makes it the worst plan there is:
# a whole 20 s fight spent out of reach costs as much as taking five full hits
# for none, which is worse than the worst record a robot that fights has shown.
# saw-v8: 60 / 30 / 10% before. A tenth of the fights (and far less of the experience near a bell) were whole
# ones, and the proof of concept is a whole fight.
FIGHT_LEN_S      = ((20.0, 0.30), (60.0, 0.40), (180.0, 0.30))   # (seconds, share of training fights)
BELL_SHORT_P     = 0.5            # share of the 20 s and 60 s fights that start from the bell (every 180 s fight does):
                                  # both robots at rest, facing each other, weapons barely turning
BELL_FACE_RAD    = 0.5            # from the bell each robot points at the other within this many radians
PRESS_V_CAP_MPH  = 9.0            # a start with them pressing on us has them moving at most this fast
R_PASSIVE_PER_S  = 0.06           # charged while further than PASSIVE_FT from an armed opponent ...
PASSIVE_FT       = 14.0
PASSIVE_CLOSE_FT_S = 3.0          # ... unless we are driving at them at least this fast (either end first)
PASSIVE_PTS_PER_S = R_PASSIVE_PER_S / R_DAMAGE   # the same charge on the scorecard, in damage: 0.24 a second
# Share of fights that start in a spot the policy has to get out of (or cash in)
SPAWN_PINNED     = 0.12           # we are against a wall with them pressing on us
SPAWN_CORNERED   = 0.08           # we are in a corner with them closing
SPAWN_CONTACT    = 0.10           # they are on our body out in the open
SPAWN_PINNING    = 0.10           # they are against a wall with us on them

# saw-v8. What the camera can say about the way the opponent points, drawn per fight: nothing, the line of its
# long axis (either end may be the front), or the axis and which end is the nose. The team's tracker gives no
# orientation (their answer, 2026-11-11), so every training fight is the first kind. The other two stay in
# the code for the day it does, and are reached through Scenario(heading_mode=...).
HDG_SENSE_P         = (1.0, 0.0, 0.0)      # none / axis only / axis and nose
HDG_SENSE_SIGMA_DEG = (5.0, 25.0)          # noise on that reading, one sigma, drawn per fight
HDG_SENSE_DROP_P    = 0.10                 # share of camera frames in which the reading is missing
CAMERA_PERIOD_STEPS = (2, 8)               # physics steps between camera frames, drawn per fight (12.5 to 50 Hz).
                                           # The fix is held, noise and all, until the next frame.
SAW_OBS_RING        = 32                   # saw observer history, physics steps: lag 14 + jitter 1 + frame age 7, and room
PRIV_DIM            = 40                   # true-state features the training-only critic sees (saw_env.SawEnv.privileged)


def angle_wrap(a):
    """Wrap an angle (float or tensor) to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi
