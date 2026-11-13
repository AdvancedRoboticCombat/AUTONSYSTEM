# radiomaster_reader

This package enables the reading of channels 1-4 of a Radiomaster Zorro (or any other similar EdgeTX controller) through SDL2

## Dependencies
- ROS 2 Jazzy
- SDL2

Install SDL2:
```bash
sudo apt install libsdl2-dev
```

## Installation 

1. Navigate to the `src` directory of your ROS2 workspace

2. Clone this repository

3. Build the package from the root of your workspace: 
`colcon build --symlink-install`

4. Run the command: 
`source install/setup.bash`

## Usage 
Plug in a Radiomaster Zorro (or similar) into your computer's USB port.

Run:
`ros2 run radiomaster_reader read_tx`

The joystick states from ch. 1-4 will be published to the `radiomaster/channels` topic at 200 hz

