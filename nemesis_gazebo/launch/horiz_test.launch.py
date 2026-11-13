# horiz_test.launch.py: A Gazebo sim to test spawning of generic horizontal opponent

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # where tf are the robots?
    nem = get_package_share_directory("nemesis_gazebo")
    opp = get_package_share_directory("nemesis_gazebo")
    worlds = get_package_share_directory("nemesis_gazebo")
    os.environ["GZ_SIM_RESOURCE_PATH"] = (
        os.path.dirname(nem) + ":" +
        os.path.dirname(opp) + ":" +
        os.path.dirname(worlds) + ":" +
        os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    )

    # World
    world = PathJoinSubstitution([
        FindPackageShare("nemesis_gazebo"),
        "worlds",
        "battlebox.sdf", # Specify your world file here
    ])

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        output="screen"
    )

    # Spawn URDF
    urdf = PathJoinSubstitution([
        FindPackageShare("nemesis_gazebo"),
        "models", "generic_horiz_urdf", "urdf",
        "generic_horiz_urdf.urdf",
    ])

    spawn = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "morberizer",
            "-file", urdf,
            "-x", "0", "-y", "0", "-z", "0.1"
        ],
        output="screen"
    )

    # Bridge wheel velocity commands ROS -> Gazebo
    bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/opponent/left_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/right_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/weapon_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
        ],
        output="screen"
    )

    # ----------------------------------
    # Teleop Control
    # ----------------------------------
    # Run radiomaster reader package
    radiomaster = ExecuteProcess(
        cmd=["ros2", "run", "radiomaster_reader", "read_tx"],
        output="screen"
    )

    # Run rc_to_wheels_opponent node
    rc_to_wheels_opponent = ExecuteProcess(
        cmd=["ros2", "run", "nemesis_gazebo", "rc_to_wheels_opponent"],
        output="screen"
    )
    # ----------------------------------

    # Delay bringup until Gazebo is ready
    bringup = TimerAction(
        period=2.0,
        actions=[spawn, bridge, radiomaster, rc_to_wheels_opponent]
    )

    return LaunchDescription([gazebo, bringup])
