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
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
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
        self.declare_parameter('crawl_speed', 2.85)        # [m/s] fallback crawl speed
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here
        self.declare_parameter('wheelbase', 0.33)           # [m]
        self.declare_parameter('route_file', 'icra26_shortest_clear45.csv')
        self.declare_parameter('lookahead_min', 1.00)       # [m] min lookahead in corners
        self.declare_parameter('lookahead_max', 2.30)       # [m] max lookahead on straights
        self.declare_parameter('lookahead_ratio', 0.34)     # L = lookahead_ratio * v + lookahead_min
        self.declare_parameter('max_speed', 15.0)            # [m/s] top straight speed
        self.declare_parameter('min_speed', 2.5)            # [m/s] minimum cornering speed
        self.declare_parameter('a_lat_max', 8.6)            # [m/s^2] max lateral acceleration
        self.declare_parameter('a_brake', 5.8)              # [m/s^2] braking deceleration limit
        self.declare_parameter('a_accel', 5.0)              # [m/s^2] forward acceleration limit
        self.declare_parameter('max_steer', 0.38)           # [rad] steering angle cap
        self.declare_parameter('steering_rate_limit', 3.2)  # [rad/s] maximum steering slew rate

        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.lookahead_min = float(self.get_parameter('lookahead_min').value)
        self.lookahead_max = float(self.get_parameter('lookahead_max').value)
        self.lookahead_ratio = float(self.get_parameter('lookahead_ratio').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.a_lat_max = float(self.get_parameter('a_lat_max').value)
        self.a_brake = float(self.get_parameter('a_brake').value)
        self.a_accel = float(self.get_parameter('a_accel').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.steering_rate_limit = float(self.get_parameter('steering_rate_limit').value)

        # Latest known pose and speed from odometry
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        # Controller tracking state
        self.prev_steering = 0.0
        self.prev_nearest_idx = None

        # Load route and precompute optimal curvature-bounded speed profile
        self.route = None
        self.segment_lengths = None
        self.speed_profile = None
        self.curvatures = None

        route_filename = self.get_parameter('route_file').value
        package_share = get_package_share_directory('team_driver')
        route_path = os.path.join(package_share, 'config', route_filename)

        if not os.path.isfile(route_path):
            fallback_path = os.path.join(package_share, 'config', 'icra26_shortest_clear45.csv')
            if os.path.isfile(fallback_path):
                route_path = fallback_path

        try:
            route = np.loadtxt(
                route_path,
                delimiter=',',
                comments='#',
                dtype=np.float64,
            )
            if route.ndim != 2 or route.shape[1] < 2:
                raise ValueError('route CSV must contain x,y columns')

            self.route = route[:, :2]
            self.segment_lengths = np.hypot(
                np.diff(self.route[:, 0], append=self.route[0, 0]),
                np.diff(self.route[:, 1], append=self.route[0, 1]),
            )
            self.speed_profile, self.curvatures = self.compute_speed_profile(self.route)

            self.get_logger().info(
                f'Loaded route: {len(self.route)} points from {route_path}. '
                f'Speed profile: min={self.speed_profile.min():.2f}m/s, '
                f'max={self.speed_profile.max():.2f}m/s, '
                f'mean={self.speed_profile.mean():.2f}m/s'
            )
        except Exception as exc:
            self.get_logger().error(
                f'Could not load Pure Pursuit route: {exc}. '
                f'Falling back to Follow-the-Gap.'
            )

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

            if getattr(self, 'position', None) is not None:
                x, y = self.position
                self.get_logger().info(
                    f'POSE x={x:.2f}, y={y:.2f}, '
                    f'yaw={math.degrees(self.yaw):+.1f} deg, '
                    f'speed={self.speed:.2f} m/s'
                )

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
        """Primary planner: Pure Pursuit with Follow-the-Gap fallback."""

        if self.position is not None and self.route is not None:
            result = self.plan_pure_pursuit()

            if result is not None:
                return result

        # Safety fallback if pose/route/target is unavailable.
        return self.plan_gap(ranges, angles)


    def compute_speed_profile(self, route):
        """Compute optimal curvature-bounded speed profile with forward/backward passes."""
        n = len(route)
        k = 3  # smoothing offset for curvature estimation
        curvatures = np.zeros(n)

        for i in range(n):
            p_prev = route[(i - k) % n]
            p_curr = route[i]
            p_next = route[(i + k) % n]

            a = float(np.hypot(*(p_curr - p_prev)))
            b = float(np.hypot(*(p_next - p_curr)))
            c = float(np.hypot(*(p_next - p_prev)))

            area = 0.5 * abs(
                (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1])
                - (p_curr[1] - p_prev[1]) * (p_next[0] - p_prev[0])
            )
            if a * b * c > 1e-6:
                curvatures[i] = 4.0 * area / (a * b * c)
            else:
                curvatures[i] = 0.0

        # Raw curvature-limited cornering speeds: v = sqrt(a_lat / kappa)
        speeds = np.full(n, self.max_speed)
        for i in range(n):
            if curvatures[i] > 1e-3:
                v_crit = math.sqrt(self.a_lat_max / curvatures[i])
                speeds[i] = min(self.max_speed, max(self.min_speed, v_crit))

        ds = np.hypot(
            np.diff(route[:, 0], append=route[0, 0]),
            np.diff(route[:, 1], append=route[0, 1]),
        )

        # Backward braking pass: ensure the car slows down BEFORE corners
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                next_i = (i + 1) % n
                v_allowed = math.sqrt(speeds[next_i]**2 + 2.0 * self.a_brake * ds[i])
                if speeds[i] > v_allowed:
                    speeds[i] = v_allowed

        # Forward acceleration pass: accelerate realistically out of turns
        for _ in range(2):
            for i in range(n):
                next_i = (i + 1) % n
                v_allowed = math.sqrt(speeds[i]**2 + 2.0 * self.a_accel * ds[i])
                if speeds[next_i] > v_allowed:
                    speeds[next_i] = v_allowed

        return speeds, curvatures

    def plan_pure_pursuit(self):
        """Track the precomputed closed route using Adaptive Pure Pursuit."""
        if self.position is None or self.route is None or self.speed_profile is None:
            return None

        x, y = self.position
        n = len(self.route)
        if n < 2:
            return None

        # --------------------------------------------------------------
        # 1. Local search for closest route point (prevents track hopping)
        # --------------------------------------------------------------
        if self.prev_nearest_idx is not None:
            search_window = 40
            indices = (self.prev_nearest_idx - 10 + np.arange(search_window)) % n
            dx_w = self.route[indices, 0] - x
            dy_w = self.route[indices, 1] - y
            dist_sq_w = dx_w * dx_w + dy_w * dy_w
            best_local = int(np.argmin(dist_sq_w))
            nearest_index = int(indices[best_local])
            cte_sq = float(dist_sq_w[best_local])

            # Global fallback if tracking was somehow lost (>2.5m error)
            if cte_sq > 6.25:
                dx_all = self.route[:, 0] - x
                dy_all = self.route[:, 1] - y
                dist_sq = dx_all * dx_all + dy_all * dy_all
                nearest_index = int(np.argmin(dist_sq))
                cte_sq = float(dist_sq[nearest_index])
        else:
            dx_all = self.route[:, 0] - x
            dy_all = self.route[:, 1] - y
            dist_sq = dx_all * dx_all + dy_all * dy_all
            nearest_index = int(np.argmin(dist_sq))
            cte_sq = float(dist_sq[nearest_index])

        self.prev_nearest_idx = nearest_index

        # --------------------------------------------------------------
        # 2. Dynamic Lookahead Distance (speed-dependent, stops zigzagging)
        # --------------------------------------------------------------
        current_v = max(self.speed, 1.0)
        local_curv = float(self.curvatures[nearest_index]) if self.curvatures is not None else 0.0
        base_L = self.lookahead_ratio * current_v + self.lookahead_min
        lookahead = float(np.clip(
            base_L / (1.0 + 0.35 * local_curv),
            self.lookahead_min,
            self.lookahead_max,
        ))

        # --------------------------------------------------------------
        # 3. Find lookahead point by path arc length (prevents track hopping)
        # --------------------------------------------------------------
        target_point = None
        target_index = nearest_index
        accum = 0.0
        curr = nearest_index

        for _ in range(n):
            nxt = (curr + 1) % n
            seg_len = float(self.segment_lengths[curr])
            if accum + seg_len >= lookahead:
                alpha = (lookahead - accum) / seg_len if seg_len > 1e-4 else 0.5
                alpha = float(np.clip(alpha, 0.0, 1.0))
                p_curr = self.route[curr]
                p_next = self.route[nxt]
                target_point = (1.0 - alpha) * p_curr + alpha * p_next
                target_index = nxt
                break
            accum += seg_len
            curr = nxt

        if target_point is None:
            fwd = (nearest_index + 5) % n
            target_point = self.route[fwd]
            target_index = fwd

        # --------------------------------------------------------------
        # 4. Transform target into vehicle coordinates
        # --------------------------------------------------------------
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        dx = float(target_point[0] - x)
        dy = float(target_point[1] - y)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        lookahead_sq = local_x * local_x + local_y * local_y
        if lookahead_sq < 1e-6:
            return None

        # --------------------------------------------------------------
        # 5. Pure Pursuit Curvature and Steering with Slew-Rate Limiting
        # --------------------------------------------------------------
        curvature = 2.0 * local_y / lookahead_sq
        raw_steering = math.atan(self.wheelbase * curvature)
        target_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))

        # Steering slew-rate limiter: smooths out high-frequency jerking
        dt = 0.025
        max_delta_steer = self.steering_rate_limit * dt
        steering = float(np.clip(
            target_steering,
            self.prev_steering - max_delta_steer,
            self.prev_steering + max_delta_steer,
        ))
        self.prev_steering = steering

        # --------------------------------------------------------------
        # 6. Dynamic Speed Command from Speed Profile
        # --------------------------------------------------------------
        base_speed = min(self.speed_profile[nearest_index], self.speed_profile[target_index])

        # Cross-track error damping: gently reduce speed if drifting wide
        cte = math.sqrt(cte_sq)
        if cte > 0.30:
            speed_factor = float(np.clip(1.0 - 0.6 * (cte - 0.30), 0.65, 1.0))
            speed = base_speed * speed_factor
        else:
            speed = base_speed

        if self._marker_divisor == 0:
            self.get_logger().info(
                f'PP near={nearest_index}, tgt={target_index}, '
                f'cte={cte:.2f}m, L={lookahead:.2f}m, '
                f'steer={math.degrees(steering):+.1f}deg, '
                f'speed={speed:.2f}m/s'
            )

        return steering, speed
    
    def plan_gap(self, ranges, angles):
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

        # Diagnostic: inspect all available gaps at the bottom-left route split.
        if (
            getattr(self, 'position', None) is not None
            and self._marker_divisor == 0
        ):
            x, y = self.position

            if x < -7.3 and 8.8 < y < 10.0:
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
                        f'w={int(width)}, '
                        f'centre={centre_deg:+.1f}deg, '
                        f'span=[{start_deg:+.1f},{end_deg:+.1f}]'
                    )

                self.get_logger().warn(
                    f'FORK_SCAN x={x:.2f}, y={y:.2f}, '
                    f'yaw={math.degrees(self.yaw):+.1f}deg | '
                    + ' | '.join(gap_info)
                )

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
            speed = 2.65

        else:
            # Linear interpolation from crawl_speed at 10 deg
            # to 2.65 m/s at 30 deg.
            alpha = (target_deg_abs - 10.0) / 20.0
            speed = (
                self.crawl_speed
                + alpha * (2.65 - self.crawl_speed)
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
