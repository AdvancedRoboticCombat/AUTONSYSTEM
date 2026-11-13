# Real-robot bring-up for Nemesis: every ROS node of the autonomy stack.
#
# The camera + dashboard backend (nemesis_dashboard/main.py) is NOT started
# here: it needs a GUI for camera setup. scripts/all.sh starts both together;
# scripts/dashboard.sh starts only the camera/dashboard side.
#
# Data flow:
#   dashboard backend     ──► /nemesis/centroid, /opponent/centroid, /nemesis/override
#   telemetry receiver    ──► /nemesis/imu                (LoRa receiver on USB serial)
#   saw_policy_node       ──► /control/channels           (while override is off)
#                             reads the centroids and the IMU directly
#   radiomaster_reader    ──► /radiomaster/channels
#   channel_mux           ──► /control/channels           (while override is on)
#   sim2real (PPM)        ◄── /control/channels ──► audio jack ──► Zorro trainer port
#
# Usage:
#   ros2 launch nemesis_gazebo nemesis.launch.py
#   ros2 launch nemesis_gazebo nemesis.launch.py policy:=false ppm:=false   # dry run
#
# Arguments:
#   policy       true/false: run the policy                        (default true)
#   checkpoint   saw policy checkpoint, absolute or relative to auton_training/
#                (default: the one named in the settings file)
#   params       settings file for the saw policy
#                (default nemesis_policy_engine/config/saw_policy.yaml)
#   weapon       true/false: send blade and arm commands           (default false: drive only)
#   ppm          true/false: send /control/channels to the transmitter   (default true)
#   imu          true/false: run the IMU telemetry receiver        (default true)
#   radio        true/false: read the Radiomaster for manual override    (default true)
#   plotter      true/false: run the Kalman filter and its live plotter  (default false)

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Repository root. With `colcon build --symlink-install` this file is a symlink
# into the source tree, so realpath finds it; NEMESIS_ROOT overrides that.
ROOT = os.environ.get("NEMESIS_ROOT") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)

# Plain scripts need the unified venv (torch, sounddevice, pySerialTransfer).
_VENV_PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
PYTHON = _VENV_PYTHON if os.path.exists(_VENV_PYTHON) else "python3"


def generate_launch_description():
    args = [
        DeclareLaunchArgument("policy", default_value="true", description="Run the policy."),
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="Saw checkpoint (empty: the one in the settings file)."),
        DeclareLaunchArgument("params",
                              default_value=os.path.join(ROOT, "nemesis_policy_engine", "config", "saw_policy.yaml"),
                              description="Settings file for the saw policy."),
        DeclareLaunchArgument("weapon", default_value="false",
                              description="Send blade and arm commands (false: drive only)."),
        DeclareLaunchArgument("ppm", default_value="true", description="Run the PPM (audio) output to the transmitter."),
        DeclareLaunchArgument("imu", default_value="true", description="Run the IMU telemetry receiver."),
        DeclareLaunchArgument("radio", default_value="true", description="Read the Radiomaster for manual override."),
        DeclareLaunchArgument("plotter", default_value="false", description="Run the Kalman filter and its live plotter."),
    ]

    # RC input. Publishes /radiomaster/channels.
    radiomaster = Node(
        package="radiomaster_reader",
        executable="read_tx",
        name="radiomaster_reader",
        output="screen",
        condition=IfCondition(LaunchConfiguration("radio")),
    )
    # If the transmitter is not plugged in, keep going without manual override.
    radiomaster_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=radiomaster,
            on_exit=[LogInfo(msg="[WARN] radiomaster_reader exited (TX not connected?). "
                                 "Manual override will not drive the robot.")],
        )
    )

    # Forwards /radiomaster/channels to /control/channels while /nemesis/override is true.
    channel_mux = Node(
        package="nemesis_control",
        executable="channel_mux",
        name="channel_mux",
        output="screen",
    )

    # IMU over the LoRa receiver. Publishes /nemesis/imu. Without it the
    # Kalman filter runs camera-only.
    imu = ExecuteProcess(
        cmd=[PYTHON, os.path.join(ROOT, "telemetry", "ros_node", "telemetry_receiver_ros.py")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("imu")),
    )

    # /control/channels -> PPM on the default audio output -> transmitter.
    # Sends neutral whenever commands stop arriving.
    ppm = ExecuteProcess(
        cmd=[PYTHON, "-u", "sim2real.py"],
        cwd=os.path.join(ROOT, "nemesis_control", "src"),
        # The blade and arm channels go out only when the weapon is switched on
        additional_env={"NEMESIS_PPM_WEAPON": LaunchConfiguration("weapon")},
        output="screen",
        condition=IfCondition(LaunchConfiguration("ppm")),
    )

    # Smoothed pose and velocity, for looking at in the plotter. The policy does not use it.
    kalman = Node(
        package="nemesis_tracking",
        executable="kalman_node",
        name="kalman_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("plotter")),
    )

    kalman_plotter = ExecuteProcess(
        cmd=[PYTHON, os.path.join(ROOT, "nemesis_tracking", "nemesis_tracking", "kalman_plotter.py")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("plotter")),
    )

    # The hammer-saw policy trained in auton_training. Reads the camera
    # centroids and the IMU itself; needs the IMU.
    saw_policy = Node(
        package="nemesis_policy_engine",
        executable="saw_policy_node",
        name="saw_policy_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params"),
            {"weapon_enabled": ParameterValue(LaunchConfiguration("weapon"), value_type=bool),
             # Empty leaves the settings file's choice of checkpoint alone
             "checkpoint_override": ParameterValue(LaunchConfiguration("checkpoint"), value_type=str),
             "auton_dir": os.path.join(ROOT, "auton_training")},
        ],
        condition=IfCondition(LaunchConfiguration("policy")),
    )

    # Startup sequence
    # t=0s — manual path (mux, radio, PPM) and sensors first, so the driver has
    #        control before anything autonomous starts.
    # t=5s — the policy.
    return LaunchDescription([
        *args,
        # ~/.local can hold a numpy 2 that breaks the system OpenCV / torch mix.
        SetEnvironmentVariable("PYTHONNOUSERSITE", "1"),
        channel_mux,
        radiomaster,
        radiomaster_exit_handler,
        ppm,
        imu,
        TimerAction(period=2.0, actions=[kalman, kalman_plotter]),
        TimerAction(period=5.0, actions=[saw_policy]),
    ])
