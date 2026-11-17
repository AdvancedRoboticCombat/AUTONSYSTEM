"""Nemesis dashboard backend: camera, vision, tracking, ROS and the web server.

    config.py      settings (edit the constants there)
    state.py       the latest camera frame and the numbers the page shows
    tracker.py     constant-velocity tracker in arena coordinates
    recorder.py    --record: frames to an mp4 and their grab times to a csv
    pipeline.py    the capture, fight-video, vision and stream loops
    ros_bridge.py  the ROS node (the only module that imports rclpy)
    server.py      the page and its WebSocket
    app.py         Dashboard: ties the above together
"""
