"""Nemesis dashboard backend.

One process: reads the overhead camera, finds both robots, turns pixels into
arena positions, publishes them to ROS, and serves the web dashboard (page and
WebSocket) on http://localhost:8080. The code is in backend/; its settings in
backend/config.py.

Usage:
    python3 main.py                                   # camera, YOLO, ROS
    python3 main.py --fight-video [VIDEO]             # recorded fight instead of the camera
    python3 main.py --record DIR                      # also save the frames and their grab times
    python3 main.py --no-ros --vision none --fight-video VIDEO
                                                      # no ROS, no detector: page and video only
"""

import argparse
import os

from backend.config import FIGHT_VIDEO_PATH, HOST, PORT, STREAM_WIDTH, Config
from vision import VISION_METHODS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Nemesis dashboard backend.")
    parser.add_argument(
        "--vision",
        default="yolo",
        choices=VISION_METHODS,
        help="Vision/detection method to run (default: yolo; 'none' finds nothing).",
    )
    parser.add_argument(
        "--fight-video",
        nargs="?",
        const=FIGHT_VIDEO_PATH,
        metavar="VIDEO",
        help=f"Play a recorded fight through the detector instead of the camera (default: {FIGHT_VIDEO_PATH}).",
    )
    parser.add_argument(
        "--record",
        metavar="DIR",
        help="Save every grabbed frame to DIR/video.mp4 and its grab time (Unix seconds) to DIR/frames.csv.",
    )
    parser.add_argument(
        "--record-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Size of the recorded video relative to the camera frame (default: 1.0, full size).",
    )
    parser.add_argument(
        "--stream-width",
        type=int,
        default=STREAM_WIDTH,
        metavar="PX",
        help=f"Shrink frames wider than this before sending them to the page (default: {STREAM_WIDTH}; 0 = camera size).",
    )
    parser.add_argument("--port", type=int, default=PORT, help=f"Port of the dashboard (default: {PORT}).")
    parser.add_argument(
        "--host",
        default=HOST,
        help=f"Address to listen on (default: {HOST}, this computer only). Anything else needs --allow-host.",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="Another name or address the page may be opened under, e.g. this computer's IP. Repeatable. "
             "Opens the OVERRIDE button to everyone who can reach that address.",
    )
    parser.add_argument("--no-ros", action="store_true", help="Run without ROS: positions go to the page only.")
    return parser.parse_args(argv)


def config_from_args(argv=None) -> Config:
    args = parse_args(argv)
    config = Config(
        vision=args.vision,
        stream_width=max(0, args.stream_width),
        host=args.host,
        port=args.port,
        allow_hosts=tuple(args.allow_host),
        use_ros=not args.no_ros,
        record_scale=args.record_scale,
    )
    if args.fight_video:
        config.start_with_fight_video = True
        config.fight_video_path = os.path.abspath(args.fight_video)
    if args.record:
        config.record_dir = os.path.abspath(args.record)
    return config


def main(argv=None):
    config = config_from_args(argv)
    from backend.app import run  # after the arguments: --help stays instant
    run(config)


if __name__ == "__main__":
    main()
