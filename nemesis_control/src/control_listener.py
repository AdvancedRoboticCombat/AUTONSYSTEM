#!/usr/bin/env python3
import rclpy
import threading
import time
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

# A command older than this is treated as "no command" (see get_channels).
STALE_SEC = 0.5

# Global variables
global_ch1 = None  
global_ch2 = None
global_ch3 = -1.0       # blade: stopped unless a message says otherwise
global_ch4 = 0.0        # arm: no command
global_ch_time = None
global_ch_lock = threading.Lock()

class ControlSubscriberNode(Node):
    def __init__(self):
        super().__init__("control_subscriber_node")
        self.control_sub_ = self.create_subscription(
            Float32MultiArray,
            "/control/channels",
            self.control_channels_callback,
            10)
        
    def control_channels_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        global global_ch1, global_ch2, global_ch3, global_ch4, global_ch_time
        with global_ch_lock:
            global_ch1 = float(msg.data[0])
            global_ch2 = float(msg.data[1])
            global_ch3 = float(msg.data[2]) if len(msg.data) > 2 else -1.0
            global_ch4 = float(msg.data[3]) if len(msg.data) > 3 else 0.0
            global_ch_time = time.monotonic()

def get_channels():
    """Latest (left, right, blade, arm) from /control/channels.

    Returns four Nones if nothing has arrived yet or the last message is
    older than STALE_SEC, so the caller can fall back to neutral instead of
    repeating an old command forever.
    """
    with global_ch_lock:
        if global_ch_time is None or time.monotonic() - global_ch_time > STALE_SEC:
            return None, None, None, None
        return global_ch1, global_ch2, global_ch3, global_ch4

def main(args=None):
    rclpy.init(args=args)

    control_subscriber_node = ControlSubscriberNode()

    rclpy.spin(control_subscriber_node)

    control_subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
