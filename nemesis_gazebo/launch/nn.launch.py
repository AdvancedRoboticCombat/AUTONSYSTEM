
# Summons Nemesis and an opponent in Gazebo: the opponent is driven by radio, Nemesis by the
# hammer-saw policy (saw_policy_node), the same node that runs on the real robot.
# gz_pose_to_centroid stands in for the camera, Gazebo's IMU for the robot's.
# Includes channel_mux so Nemesis can be manually overridden by publishing /nemesis/override.
#
#   ros2 launch nemesis_gazebo nn.launch.py
#   ros2 launch nemesis_gazebo nn.launch.py weapon:=true      also drive the blade and arm joints
#   ros2 launch nemesis_gazebo nn.launch.py headless:=true    no Gazebo window
#   ros2 launch nemesis_gazebo nn.launch.py checkpoint:=ppo_models/my_run/SAW_upd100.pt

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
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

    args = [
        DeclareLaunchArgument("headless", default_value="false", description="Run Gazebo without its window."),
        DeclareLaunchArgument("weapon", default_value="false", description="Drive the blade and arm joints too."),
        DeclareLaunchArgument("checkpoint", default_value="",
                              description="Saw checkpoint, absolute or relative to auton_training/ (empty: the default)."),
    ]

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("headless")),
    )
    gazebo_headless = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-s", world],
        output="screen",
        condition=IfCondition(LaunchConfiguration("headless")),
    )

    move_camera = ExecuteProcess(
        cmd=[
            'gz', 'service',
            '-s', '/gui/move_to/pose',
            '--reqtype', 'gz.msgs.GUICamera',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req',
            'pose: {position: {x: 0.0, y: -11.0, z: 2.5} orientation: {x: 0.0, y: 0.0, z: 0.7, w: 0.7}}'
            # 'pose: {position: {x: 0.0, y: 0.0, z: 20.5} orientation: {x: 0.0, y: 0.7, z: 0.0, w: 0.7}}'
        ],
        output='screen'
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
        "models", "generic_vert_urdf", "urdf",
        "generic_vert_urdf.urdf",  
    ])

    # -------------------------
    # Spawn both
    # -------------------------
    spawn_nemesis = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "nemesis",
            "-file", nemesis_urdf,
            "-x", "3.5", "-y", "2", "-z", "0.1"
        ],
        output="screen"
    )

    spawn_opponent = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_sim", "create",
            "-name", "opponent",
            "-file", opponent_urdf,
            "-x", "-3.5", "-y", "-2.5", "-z", "0.1",   # Starting position offset here
            "-Y", "3.14",
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

    bridge_opponent = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/opponent/left_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/right_wheels/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/left_wheels/cmd_vel_f@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/right_wheels/cmd_vel_f@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/arm_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
            "/opponent/weapon_motor/cmd_vel@std_msgs/msg/Float64@gz.msgs.Double",
        ],
        output="screen"
    )
    # -------------------------
    # Bridges (Gazebo -> ROS)
    # -------------------------
    # Bridge Gazebo ground-truth poses to intermediate topics;
    # gz_pose_to_centroid republishes them as the camera's /nemesis/centroid and /opponent/centroid.
    bridge_poses = ExecuteProcess(
    cmd=[
        "ros2", "run", "ros_gz_bridge", "parameter_bridge",
        "/model/nemesis/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
        "/model/opponent/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
        "--ros-args",
        "-r", "/model/nemesis/pose:=/nemesis/gz_pose",
        "-r", "/model/opponent/pose:=/opponent/gz_pose",
    ],
    output="screen",
    emulate_tty=True,
    )

    bridge_imus = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/world/battlebox/model/nemesis/link/base_link/sensor/imu/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "--ros-args",
            "-r", "/world/battlebox/model/nemesis/link/base_link/sensor/imu/imu:=/nemesis/imu",
        ],
        output="screen",
        emulate_tty=True,
    )

    # Stands in for the camera: Gazebo poses as the dashboard's centroid topics.
    gz_pose_to_centroid = ExecuteProcess(
        cmd=["ros2", "run", "nemesis_gazebo", "gz_pose_to_centroid"],
        output="screen",
    )
    
    # -------------------------
    # Teleop / controllers
    # -------------------------
    radiomaster = ExecuteProcess(
        cmd=["ros2", "run", "radiomaster_reader", "read_tx"],
        output="screen"
    )

    channel_mux = Node(
        package="nemesis_control",
        executable="channel_mux",
        name="channel_mux",
        output="screen",
    )

    rc_to_wheels_opponent = ExecuteProcess(
        cmd=["ros2", "run", "nemesis_gazebo", "rc_to_wheels_opponent"],
        output="screen"
    )

    control_to_wheels = ExecuteProcess(
        cmd=["ros2", "run", "nemesis_gazebo", "control_to_wheels"],
        output="screen"
    )

    # The same policy node as on the robot. Gazebo's frame is metres from the arena centre,
    # counter-clockwise positive, and Nemesis spawns pointing along +x. The arena is taken as a
    # plain 48 ft square: the simulator's wall bulges may not sit where the mesh has them.
    repo = os.environ.get("NEMESIS_ROOT") or os.path.normpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
    policy_engine_node = Node(
        package="nemesis_policy_engine",
        executable="saw_policy_node",
        name="saw_policy_node",
        output="screen",
        parameters=[
            os.path.join(repo, "nemesis_policy_engine", "config", "saw_policy.yaml"),
            {"units": "m", "origin": "center", "y_down": False,
             "arena_width_ft": 48.0, "arena_height_ft": 48.0,
             "initial_heading": "0", "half_track_ft": 1.48,
             "weapon_enabled": ParameterValue(LaunchConfiguration("weapon"), value_type=bool),
             "checkpoint_override": ParameterValue(LaunchConfiguration("checkpoint"), value_type=str),
             "auton_dir": os.path.join(repo, "auton_training")},
        ],
    )

    bringup = TimerAction(
        period=2.0,
        actions=[
            spawn_nemesis, spawn_opponent,
            bridge_nemesis, bridge_opponent,
            radiomaster,
            channel_mux,
            rc_to_wheels_opponent,
            control_to_wheels,
            move_camera,
        ]
    )

    posebridge = TimerAction(
        period=4.0,
        actions=[
            bridge_poses,
            bridge_imus,
            gz_pose_to_centroid,
        ]
    )

    # The policy starts last, once poses and the IMU are flowing
    return LaunchDescription([*args, gazebo, gazebo_headless, bringup, posebridge,
                              TimerAction(period=6.0, actions=[policy_engine_node])])
