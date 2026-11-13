# nemesis_gazebo
A driving dynamic simulator for Nemesis within Gazebo

## Dependencies
Have ROS2 Jazzy installed. The robot models live in this package under `models/`
(`nemesis_urdf_final`, `generic_horiz_urdf`, `generic_vert_urdf`: URDF, meshes, joint names).
The launch files also start nodes from `radiomaster_reader`, `nemesis_control` and
`nemesis_policy_engine`, which sit beside this package in the repository.

To look at a model in RViz2 (needs `ros-jazzy-urdf-tutorial`):
`ros2 launch urdf_tutorial display.launch.py model:=$(ros2 pkg prefix nemesis_gazebo)/share/nemesis_gazebo/models/nemesis_urdf_final/urdf/nemesis_urdf_final.urdf`

Install Gazebo Harmonic:
```
sudo apt update
sudo apt install -y gz-harmonic
sudo apt install -y ros-jazzy-ros-gz
```
## Install
```
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```
## Launch
As of 12/17/2025, this package only supports driving a virtual Nemesis manually with a Radiomaster Zorro. Future revisions will allow the user to select between manual/autonomous control of Nemesis/an opponent.
UPDATE: As of 12/25/2025, this package now supports multiple sim modes!
UPDATE: As of 01/26/2025, Nemesis is controlled automatically by the nemesis_path_engine node. 

To launch Nemesis driving sim:
`ros2 launch nemesis_gazebo sim.launch.py`

To launch generic horizontal spinner driving sim:
`ros2 launch nemesis_gazebo horiz_test.launch.py`

To launch generic vertical spinner driving sim:
`ros2 launch nemesis_gazebo vert_test.launch.py`

To drive a generic horizontal spinner against Nemesis:
`ros2 launch nemesis_gazebo fight.launch.py`

To drive Nemesis against a ragdoll horizontal spinner:
`ros2 launch nemesis_gazebo playground.launch.py`

TO launch neural-net-controlled Nemesis in sim:
`ros2 launch nemesis_gazebo nn.launch.py`

## Control
Both robots subscribe to topics dictating their motor velocities.

Nemesis subscribes to the topics under `/nemesis/`

The opponent subscribes to `/opponent/`

## Path Visualization
In order to visualize the projected path that nemesis is currently following: 
1) Run `ros2 run rviz2 rviz2` in a new terminal 
3) In rviz2 window, change `Fixed Frame` from `map` to `battlebox`
2) Select `Add` 
3) Select `By topic`
4) Add `nemesis/path_engine/path/Path` 
5) Add `nemesis/pose/Pose`
6) Add `opponent/pose/Pose`
7) Add `nemesis/path_engine/visualize/force_field_radius_marker`

## What the physics is made of
The robots and the arena are drawn from their CAD meshes but collide as simple shapes:
- arena (`worlds/battlebox.sdf`): a flat floor at z = 0 and four box walls 7.28 m from the centre;
- robots (`models/*/urdf`): a box for the body, a cylinder per wheel, a box or cylinder for the weapon.

Wheels grip (friction 1.0), bodies and walls slide (0.3), and Nemesis's saw blade barely grips (0.05)
so that touching something with it spinning does not launch the robot. The arm joint is driven at
8.4 rad/s, the real 0.35 s swing. Mesh-on-mesh collision is what made the robots bob, slide and
catch; if a model's shape changes, update the sizes in its URDF's `<collision>` blocks.
