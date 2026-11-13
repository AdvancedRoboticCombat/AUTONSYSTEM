#!/usr/bin/env python3
"""
Stands in for the overhead camera in Gazebo: turns the simulator's
ground-truth model poses into the centroid topics the dashboard backend
publishes on the real robot, so saw_policy_node runs unchanged.

Subscribes
    /nemesis/gz_pose   (geometry_msgs/PoseStamped)  - Gazebo model pose
    /opponent/gz_pose  (geometry_msgs/PoseStamped)

Publishes
    /nemesis/centroid   (geometry_msgs/PointStamped)  x, y in metres from the arena centre
    /opponent/centroid  (geometry_msgs/PointStamped)

Positions are stamped with this node's clock, as the dashboard stamps its
own: the policy judges how old a fix is against the wall clock, and Gazebo's
stamps are simulation time.
"""

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class GzPoseToCentroid(Node):
    def __init__(self):
        super().__init__("gz_pose_to_centroid")
        self._nem_pub = self.create_publisher(PointStamped, "/nemesis/centroid", 10)
        self._opp_pub = self.create_publisher(PointStamped, "/opponent/centroid", 10)
        self.create_subscription(PoseStamped, "/nemesis/gz_pose", lambda m: self._relay(m, self._nem_pub), 10)
        self.create_subscription(PoseStamped, "/opponent/gz_pose", lambda m: self._relay(m, self._opp_pub), 10)
        self.get_logger().info("gz_pose_to_centroid node started")

    def _relay(self, msg: PoseStamped, pub):
        out = PointStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "arena"
        out.point.x = msg.pose.position.x
        out.point.y = msg.pose.position.y
        out.point.z = 0.0            # a real fix, not a coasted one
        pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GzPoseToCentroid()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
