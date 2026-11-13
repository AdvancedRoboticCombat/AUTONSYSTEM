#!/usr/bin/env python3
import struct
import sys
import threading
import time
from queue import Empty, Queue

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

# IMPORTANT: This implementation uses pySerialTransfer for all serial I/O.
from pySerialTransfer import pySerialTransfer as txfer

# === CONFIG - change PORT for your Linux device ===
PORT = "/dev/ttyUSB0"  # e.g. "/dev/ttyUSB0" or "/dev/ttyACM0"
BAUDRATE = 115200      # kept for logging only; pySerialTransfer uses the underlying serial default

# The robot sends a QuaternionPacket (transmitter/transmitter.ino):
#   uint32 timestamp (ms since boot), then floats qw qx qy qz, gx gy gz, ax ay az,
#   and on newer firmware one more float, the weapon VESC's ERPM.
# The first field is an integer, not a float: reading the packet as 11 floats
# shifts every value by one (the "gyro z" was then really gyro y).
PACKET_FORMAT = "<I10f"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)        # 44 bytes
PACKET_SIZE_RPM = PACKET_SIZE + 4                   # 48 bytes with the ERPM
PACKET_SIZES = (PACKET_SIZE, PACKET_SIZE_RPM)


def parse_packet(data: bytes):
    """Unpack a QuaternionPacket (44 bytes, or 48 with the ERPM). Returns dict or None on failure."""
    if len(data) not in PACKET_SIZES:
        return None
    try:
        unpacked = struct.unpack(PACKET_FORMAT, data[:PACKET_SIZE])
        packet = {
            "timestamp_ms": unpacked[0],
            "quaternion": {
                "w": unpacked[1],
                "x": unpacked[2],
                "y": unpacked[3],
                "z": unpacked[4],
            },
            "gyroscope": {"x": unpacked[5], "y": unpacked[6], "z": unpacked[7]},
            "accelerometer": {"x": unpacked[8], "y": unpacked[9], "z": unpacked[10]},
            "erpm": struct.unpack("<f", data[PACKET_SIZE:])[0] if len(data) == PACKET_SIZE_RPM else None,
        }
        return packet
    except struct.error as e:
        print(f"[parse_packet] struct error: {e}")
        return None


class SerialReader(threading.Thread):
    """
    Uses pySerialTransfer to receive framed packets reliably.
    The thread pushes parsed packets to packet_queue.
    """

    def __init__(self, port: str, packet_queue: Queue):
        super().__init__(daemon=True)
        self.port = port
        self.packet_queue = packet_queue
        self.running = True
        self.link = None

    def run(self):
        try:
            # Construct the pySerialTransfer link with the device path
            self.link = txfer.SerialTransfer(self.port)
            # Open the underlying serial connection
            self.link.open()
            # As recommended by pySerialTransfer examples: allow Arduino to reset
            # (the library's README/example shows a short sleep after open).
            time.sleep(2.0)

            print(f"[SerialReader] Opened pySerialTransfer on {self.port}")
            print(f"[SerialReader] Expecting payload size: {PACKET_SIZE} or {PACKET_SIZE_RPM} bytes")

            # Main receive loop
            while self.running:
                # The library maintains an internal read buffer and will parse packets for us.
                # Use link.available() to wait for a full parsed packet.
                if not self.link.available():
                    # check for error status while waiting (example pattern from docs)
                    if self.link.status.value < 0:
                        # map known status errors to readable output
                        status = self.link.status
                        if status == txfer.Status.CRC_ERROR:
                            print("[SerialReader] ERROR: CRC_ERROR")
                        elif status == txfer.Status.PAYLOAD_ERROR:
                            print("[SerialReader] ERROR: PAYLOAD_ERROR")
                        elif status == txfer.Status.STOP_BYTE_ERROR:
                            print("[SerialReader] ERROR: STOP_BYTE_ERROR")
                        else:
                            print(f"[SerialReader] ERROR: {status.name}")
                        # continue loop and wait for next packet
                        time.sleep(0.001)
                        continue
                    # Nothing available yet - small sleep so this loop isn't busy-waiting
                    time.sleep(0.001)
                    continue

                # At this point a full packet has been parsed by pySerialTransfer
                rec_size = self.link.bytes_read
                # rx_buff is the underlying buffer that holds the parsed payload
                payload = bytes(self.link.rx_buff[:rec_size])

                if rec_size in PACKET_SIZES:
                    packet = parse_packet(payload)
                    if packet:
                        self.packet_queue.put(packet)
                    else:
                        print(f"[SerialReader] Failed to unpack payload (hex): {payload.hex()}")
                elif rec_size > 0:
                    # It's possible the Arduino (SerialTransfer on Arduino side) sent a differently-sized packet;
                    # report it here to help debugging.
                    print(
                        f"[SerialReader] Unexpected packet size: {rec_size} bytes (expected {PACKET_SIZE} or {PACKET_SIZE_RPM})"
                    )
                    print(f"[SerialReader] Payload (hex): {payload.hex()}")

                # Important: after handling, call link.close() only at shutdown.
                # pySerialTransfer manages its internal rx/tx buffers between iterations.

        except Exception as e:
            print(f"[SerialReader] Exception: {e}")
        finally:
            try:
                if self.link:
                    self.link.close()
                    print("[SerialReader] pySerialTransfer closed")
            except Exception:
                pass

    def stop(self):
        self.running = False


class ImuPublisher(Node):
    def __init__(self):
        super().__init__("imu_publisher_node")
        self.publisher_ = self.create_publisher(Imu, "/nemesis/imu", 10)
        # Weapon speed controller ERPM, when the firmware sends it
        self.erpm_publisher_ = self.create_publisher(Float64, "/nemesis/weapon/erpm", 10)
        self.packet_queue = Queue(maxsize=200)
        self.serial_reader = SerialReader(PORT, self.packet_queue)
        self.serial_reader.start()

        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.packet_count = 0
        self.get_logger().info("IMU Publisher Node started (pySerialTransfer)")

    def timer_callback(self):
        try:
            while True:
                packet = self.packet_queue.get_nowait()
                self.publish_imu_message(packet)
        except Empty:
            pass

    def publish_imu_message(self, packet):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        msg.orientation.w = packet["quaternion"]["w"]
        msg.orientation.x = packet["quaternion"]["x"]
        msg.orientation.y = packet["quaternion"]["y"]
        msg.orientation.z = packet["quaternion"]["z"]

        msg.angular_velocity.x = packet["gyroscope"]["x"]
        msg.angular_velocity.y = packet["gyroscope"]["y"]
        msg.angular_velocity.z = packet["gyroscope"]["z"]

        msg.linear_acceleration.x = packet["accelerometer"]["x"]
        msg.linear_acceleration.y = packet["accelerometer"]["y"]
        msg.linear_acceleration.z = packet["accelerometer"]["z"]

        self.publisher_.publish(msg)
        if packet["erpm"] is not None:
            self.erpm_publisher_.publish(Float64(data=float(packet["erpm"])))
        self.packet_count += 1

        if self.packet_count % 50 == 0:
            self.get_logger().info(
                f"Published {self.packet_count} IMU messages | "
                f"Quat: [{msg.orientation.w:.3f}, {msg.orientation.x:.3f}, "
                f"{msg.orientation.y:.3f}, {msg.orientation.z:.3f}]"
            )

    def destroy_node(self):
        self.get_logger().info("Shutting down IMU Publisher Node...")
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()