# sim.launch.py: The basic Nemesis drive sim in Gazebo

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
        "models", "nemesis_urdf_final", "urdf",
        "nemesis_urdf_final.urdf",
    ])

    spawn = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "nemesis",
            "-file", urdf,
            "-x", "0", "-y", "0", "-z", "0.1"
        ],
        output="screen"
    )

    # Bridge wheel velocity commands ROS -> Gazebo
    bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/nemesis/left_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/nemesis/right_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/nemesis/arm_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/nemesis/weapon_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
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

    # Run rc_to_wheels node
    rc_to_wheels = ExecuteProcess(
        cmd=["ros2", "run", "nemesis_gazebo", "rc_to_wheels"],
        output="screen"
    )
    # ----------------------------------

    # Delay bringup until Gazebo is ready
    bringup = TimerAction(
        period=2.0,
        actions=[spawn, bridge, radiomaster, rc_to_wheels]
    )

    return LaunchDescription([gazebo, bringup])
