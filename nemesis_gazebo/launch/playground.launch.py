# Summons Nemesis and an opponent, with Nemesis with teleop control

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
        "battlebox.sdf",
    ])

    gazebo = ExecuteProcess(
        cmd=[
            "gz", "sim", "-r",
            PathJoinSubstitution([
                FindPackageShare("nemesis_gazebo"),
                "worlds",
                "battlebox.sdf",
            ])
        ],
        output="screen"
    )


    # -------------------------
    # URDF paths
    # -------------------------
    nemesis_urdf = PathJoinSubstitution([
        FindPackageShare("nemesis_gazebo"),
        "models", "nemesis_urdf_final", "urdf",
        "nemesis_urdf_final.urdf",
    ])

    opponent_urdf = PathJoinSubstitution([
        FindPackageShare("nemesis_gazebo"),
        "models", "generic_horiz_urdf", "urdf",
        "generic_horiz_urdf.urdf",  
    ])

    # -------------------------
    # Spawn both
    # -------------------------
    spawn_nemesis = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "nemesis",
            "-file", nemesis_urdf,
            "-x", "-5.5", "-y", "0", "-z", "0.1"
        ],
        output="screen"
    )

    spawn_opponent = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "opponent",
            "-file", opponent_urdf,
            "-x", "5.5", "-y", "0", "-z", "0.1",   # Starting position offset here
            "-Y", "-3.14"
        ],
        output="screen"
    )

    # -------------------------
    # Bridges (ROS -> Gazebo)
    # -------------------------
    bridge_nemesis = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/nemesis/left_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/nemesis/right_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/nemesis/arm_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/nemesis/weapon_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
        ],
        output="screen"
    )

    # -------------------------
    # Teleop Nemesis
    # -------------------------
    radiomaster = ExecuteProcess(
        cmd=["ros2", "run", "radiomaster_reader", "read_tx"],
        output="screen"
    )

    rc_to_wheels = ExecuteProcess(
        cmd=["ros2", "run", "nemesis_gazebo", "rc_to_wheels"],
        output="screen"
    )

    bringup = TimerAction(
        period=2.0,
        actions=[
            spawn_nemesis, spawn_opponent,
            bridge_nemesis,
            radiomaster,
            rc_to_wheels,
        ]
    )

    return LaunchDescription([gazebo, bringup])
