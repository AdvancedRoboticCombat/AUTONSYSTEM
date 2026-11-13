# nemesis_control

A pure pursuit controller for Nemesis.

Inputs: Robot state (x, y, yaw), path waypoints (Completed as of 01/23/26)
* [nemesis_gazebo](https://github.com/AdvancedRoboticCombat/nemesis_gazebo.git) simulates IMU and vision inputs for simulation control

Outputs: Left wheel and right wheel throttle command (emulating Tx controller)

## Prerequisites

Ensure you have the following installed:
* ROS2 (jazzy)

## Installation

1. Navigate to the `src` directory of your ROS2 workspace

2. Clone this repository

3. Build the package from the root of your workspace: `colcon build --symlink-install` 

4. Run the command: `source install/setup.bash`
