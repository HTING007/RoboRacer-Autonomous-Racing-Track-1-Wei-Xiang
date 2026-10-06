#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and
`/drive` as the output topic and everything else is yours to change - rewrite
this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then asks for a slow constant speed with the wheels straight. It
will set off from the grid and into the first thing in front of it. That is
deliberate and it is the whole point: **there is no algorithm here and no
algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the workspace, the topics and
your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

What you are allowed to read (see docs/06-rules.md):
    /scan               LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom   ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map            the static map
What you publish:
    /drive              AckermannDriveStamped - you ask for a SPEED, and the
                        simulator closes that loop for you
    /driver/...         anything of your own, for visualisation
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        # ros2 run team_driver driver --ros-args -p crawl_speed:=2.0
        # Add your own as you go; config/driver_params.yaml loads them.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('crawl_speed', 2.8)        # [m/s]
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here

        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        self._marker_divisor = 0
        self.get_logger().warn(
            'team_driver is up, but this template has no driving logic: it will '
            'crawl straight ahead until it hits something. Implement plan().')

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (about 40 Hz).
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """Decide what the car should do, given the latest scan.

        Returns `(steering, speed)`: a steering angle in radians, and a speed
        in m/s. Unlike Track 2, the speed is a genuine request - the simulator
        runs the controller that achieves it - so you can think in the units
        your algorithm naturally produces.

        Right now it returns "straight ahead, slowly", which is not driving:
        it ignores `ranges` entirely, so the car will hold its heading off the
        grid and put itself into the first wall it meets. Replace the whole
        method.

        You have more to work with than the scan. `self.position`, `self.yaw`
        and `self.speed` are ground truth from /ego_racecar/odom and are yours
        to use, the occupancy grid is published on /map, and nothing stops you
        subscribing to more topics, loading a line you computed offline, or
        running a policy you trained. See docs/04-algorithms.md for the
        approaches and what each one needs.
        """
        # Only consider the forward 180-degree field of view.
        front_mask = np.abs(angles) <= (math.pi / 2)

        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        # Temporary tuning value for this experiment.
        # A beam is considered usable only if it has at least this much clearance.
        safe_distance = 1.5  # [m]

        safe = front_ranges > safe_distance

        # Find contiguous True regions in the safe/unsafe mask.
        padded = np.concatenate(([False], safe, [False]))
        changes = np.diff(padded.astype(np.int8))

        gap_starts = np.where(changes == 1)[0]
        gap_ends = np.where(changes == -1)[0]  # end is exclusive

        if len(gap_starts) == 0:
            # No safe forward gap: stop rather than blindly drive into an obstacle.
            return 0.0, 0.0

        gap_widths = gap_ends - gap_starts
        best_gap = np.argmax(gap_widths)

        gap_start = gap_starts[best_gap]
        gap_end = gap_ends[best_gap]

        # Aim at the angular centre of the widest safe gap.
        target_index = (gap_start + gap_end - 1) // 2
        target_angle = front_angles[target_index]

        # Do not command the full target angle directly.
        steering = 0.5 * target_angle

        # Respect the simulator steering limit with some margin.
        steering = np.clip(steering, -0.35, 0.35)

        # Reduce speed when the required direction becomes aggressive.
        # Keep full speed on straights and progressively slow for turns.
        target_deg_abs = abs(math.degrees(target_angle))

        if target_deg_abs <= 10.0:
            speed = self.crawl_speed

        elif target_deg_abs >= 30.0:
            speed = 2.6

        else:
            # Linear interpolation from crawl_speed at 10 deg
            # to 2.5 m/s at 30 deg.
            alpha = (target_deg_abs - 10.0) / 20.0
            speed = (
                self.crawl_speed
                + alpha * (2.6 - self.crawl_speed)
            )

        if self._marker_divisor == 0:
            target_deg = math.degrees(target_angle)

            self.get_logger().info(
                f'gap_width={gap_widths[best_gap]} beams, '
                f'target={target_deg:+.1f} deg, '
                f'steer={math.degrees(float(steering)):+.1f} deg'
            )

            # Diagnostic only:
            # When the chosen direction suddenly points far sideways,
            # print every candidate safe gap so we can understand why.
            if abs(target_deg) > 45.0:
                gap_info = []

                for i, (start, end, width) in enumerate(
                    zip(gap_starts, gap_ends, gap_widths)
                ):
                    centre_index = (start + end - 1) // 2

                    centre_deg = math.degrees(front_angles[centre_index])
                    start_deg = math.degrees(front_angles[start])
                    end_deg = math.degrees(front_angles[end - 1])

                    chosen = "*" if i == best_gap else " "

                    gap_info.append(
                        f'{chosen}gap{i}: '
                        f'width={int(width)}, '
                        f'centre={centre_deg:+.1f}deg, '
                        f'span=[{start_deg:+.1f},{end_deg:+.1f}]'
                    )

                self.get_logger().info(
                    f'gap_width={gap_widths[best_gap]} beams, '
                    f'target={math.degrees(target_angle):+.1f} deg, '
                    f'steer={math.degrees(float(steering)):+.1f} deg, '
                    f'speed={speed:.2f} m/s'
                )

        return steering, speed

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
