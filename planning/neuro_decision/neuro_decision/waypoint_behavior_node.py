import glob
import math
import os
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, String


class WaypointBehaviorNode(Node):
    def __init__(self):
        super().__init__('waypoint_behavior_node')

        egg = glob.glob(os.path.expanduser('~/carla_sim/PythonAPI/carla/dist/carla-*.egg'))
        if egg:
            sys.path.append(egg[0])

        try:
            import carla
            self.carla = carla
        except ImportError as e:
            raise RuntimeError('CARLA Python API를 찾지 못했습니다.') from e

        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('role_name', 'ego_vehicle')

        self.declare_parameter('desired_speed_straight_mps', 2.20)
        self.declare_parameter('desired_speed_gentle_turn_mps', 1.25)
        self.declare_parameter('desired_speed_sharp_turn_mps', 0.95)
        self.declare_parameter('lookahead_straight_m', 8.0)
        self.declare_parameter('lookahead_turn_m', 4.2)
        self.declare_parameter('route_step_m', 1.5)
        self.declare_parameter('route_length_points', 100)
        self.declare_parameter('averaging_window_straight', 8)
        self.declare_parameter('averaging_window_turn', 3)
        self.declare_parameter('turn_threshold_abs_local_y_small', 0.35)
        self.declare_parameter('turn_threshold_abs_local_y_large', 0.85)
        self.declare_parameter('target_y_clamp_m', 1.8)
        self.declare_parameter('center_offset_m', -0.10)
        self.declare_parameter('target_smoothing_alpha', 0.60)
        self.declare_parameter('control_period_s', 0.1)

        self.carla_host = str(self.get_parameter('carla_host').value)
        self.carla_port = int(self.get_parameter('carla_port').value)
        self.role_name = str(self.get_parameter('role_name').value)

        self.desired_speed_straight_mps = float(self.get_parameter('desired_speed_straight_mps').value)
        self.desired_speed_gentle_turn_mps = float(self.get_parameter('desired_speed_gentle_turn_mps').value)
        self.desired_speed_sharp_turn_mps = float(self.get_parameter('desired_speed_sharp_turn_mps').value)
        self.lookahead_straight_m = float(self.get_parameter('lookahead_straight_m').value)
        self.lookahead_turn_m = float(self.get_parameter('lookahead_turn_m').value)
        self.route_step_m = float(self.get_parameter('route_step_m').value)
        self.route_length_points = int(self.get_parameter('route_length_points').value)
        self.averaging_window_straight = max(1, int(self.get_parameter('averaging_window_straight').value))
        self.averaging_window_turn = max(1, int(self.get_parameter('averaging_window_turn').value))
        self.turn_threshold_small = float(self.get_parameter('turn_threshold_abs_local_y_small').value)
        self.turn_threshold_large = float(self.get_parameter('turn_threshold_abs_local_y_large').value)
        self.target_y_clamp_m = float(self.get_parameter('target_y_clamp_m').value)
        self.center_offset_m = float(self.get_parameter('center_offset_m').value)
        self.target_smoothing_alpha = float(self.get_parameter('target_smoothing_alpha').value)
        self.control_period_s = float(self.get_parameter('control_period_s').value)

        self.desired_speed_pub = self.create_publisher(Float64, '/desired_speed', 10)
        self.target_point_pub = self.create_publisher(Point, '/target_point', 10)
        self.state_pub = self.create_publisher(String, '/behavior_state', 10)
        self.debug_pub = self.create_publisher(String, '/behavior_debug_text', 10)

        self.client = self.carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()

        self.ego_vehicle = None
        self.route_world = []
        self.filtered_target_x = None
        self.filtered_target_y = None
        self.filtered_desired_speed = 1.0

        self.timer = self.create_timer(self.control_period_s, self.control_loop)

        self.get_logger().info('🛣️ waypoint_behavior_node 시작: global waypoint route 기반 추종')

    def find_ego_vehicle(self):
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == self.role_name:
                return actor
        return None

    def yaw_diff_deg(self, a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    def world_to_vehicle_local(self, ego_transform, target_location):
        ego_loc = ego_transform.location
        dx = target_location.x - ego_loc.x
        dy = target_location.y - ego_loc.y

        yaw_rad = math.radians(ego_transform.rotation.yaw)

        local_x = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
        local_y = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy

        return float(local_x), float(local_y)

    def publish_stop(self, reason='stop'):
        speed_msg = Float64()
        speed_msg.data = 0.0
        self.desired_speed_pub.publish(speed_msg)

        target_msg = Point()
        target_msg.x = 0.0
        target_msg.y = 0.0
        target_msg.z = 0.0
        self.target_point_pub.publish(target_msg)

        state_msg = String()
        state_msg.data = 'STOP'
        self.state_pub.publish(state_msg)

        debug_msg = String()
        debug_msg.data = f'state=STOP | reason={reason}'
        self.debug_pub.publish(debug_msg)

    def choose_best_next_wp(self, current_wp):
        next_wps = current_wp.next(self.route_step_m)
        if not next_wps:
            return None

        current_yaw = current_wp.transform.rotation.yaw
        best_wp = min(
            next_wps,
            key=lambda wp: self.yaw_diff_deg(wp.transform.rotation.yaw, current_yaw)
        )
        return best_wp

    def build_route_from_ego(self):
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location

        current_wp = self.carla_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving
        )

        if current_wp is None:
            return False

        route = [current_wp.transform.location]
        wp = current_wp

        for _ in range(self.route_length_points):
            next_wp = self.choose_best_next_wp(wp)
            if next_wp is None:
                break
            route.append(next_wp.transform.location)
            wp = next_wp

        if len(route) < 5:
            return False

        self.route_world = route
        return True

    def prune_passed_points(self, ego_transform):
        pruned = []
        for loc in self.route_world:
            local_x, _ = self.world_to_vehicle_local(ego_transform, loc)
            if local_x > -2.0:
                pruned.append(loc)
        self.route_world = pruned

    def calculate_dynamic_speed(self, local_y):
        abs_y = abs(float(local_y))
        if abs_y < self.turn_threshold_small:
            return self.desired_speed_straight_mps
        elif abs_y < self.turn_threshold_large:
            return self.desired_speed_gentle_turn_mps
        else:
            return self.desired_speed_sharp_turn_mps

    def choose_target_from_route(self, ego_transform):
        if not self.route_world:
            return None

        forward_points = []
        for loc in self.route_world:
            local_x, local_y = self.world_to_vehicle_local(ego_transform, loc)
            if local_x > 0.5:
                forward_points.append((loc, local_x, local_y))

        if not forward_points:
            return None

        # 현재 근처 곡률(local_y)로 코너 판단
        nearby_abs_y = [abs(p[2]) for p in forward_points[:3]]
        is_in_corner = len(nearby_abs_y) > 0 and max(nearby_abs_y) > self.turn_threshold_small

        # 코너면 짧은 lookahead, 직선면 긴 lookahead
        target_lookahead = self.lookahead_turn_m if is_in_corner else self.lookahead_straight_m

        candidate_idx = None
        for i, (_, local_x, _) in enumerate(forward_points):
            if local_x >= target_lookahead:
                candidate_idx = i
                break

        if candidate_idx is None:
            candidate_idx = max(0, len(forward_points) - 1)

        # 직선: 더 작은 평균 범위 (5), 코너: 더 작은 평균 범위 (3)
        if is_in_corner:
            avg_window = self.averaging_window_turn
        else:
            avg_window = self.averaging_window_straight
        
        start_idx = max(0, candidate_idx - 1)
        end_idx = min(len(forward_points), candidate_idx + avg_window)
        selected = forward_points[start_idx:end_idx]

        if len(selected) == 1:
            avg_x = selected[0][1]
            avg_y = selected[0][2]
        else:
            weighted_sum_x = 0.0
            weighted_sum_y = 0.0
            weight_sum = 0.0
            for i, (_, local_x, local_y) in enumerate(selected):
                weight = 1.0 + (i * 0.25)
                weighted_sum_x += weight * local_x
                weighted_sum_y += weight * local_y
                weight_sum += weight

            avg_x = weighted_sum_x / weight_sum
            avg_y = weighted_sum_y / weight_sum

        # 직선에서는 center_offset 최소화 (중앙 강조)
        if not is_in_corner:
            corrected_y = avg_y * 0.95  # 95% 중앙 추종
        else:
            corrected_y = avg_y - self.center_offset_m
        
        return is_in_corner, avg_x, corrected_y

    def control_loop(self):
        self.world = self.client.get_world()

        if self.ego_vehicle is None:
            self.ego_vehicle = self.find_ego_vehicle()

        if self.ego_vehicle is None:
            self.publish_stop(reason='ego_vehicle_not_found')
            return

        ego_transform = self.ego_vehicle.get_transform()

        if len(self.route_world) < 10:
            ok = self.build_route_from_ego()
            if not ok:
                self.publish_stop(reason='route_build_failed')
                return

        self.prune_passed_points(ego_transform)

        if len(self.route_world) < 10:
            ok = self.build_route_from_ego()
            if not ok:
                self.publish_stop(reason='route_rebuild_failed')
                return

        target = self.choose_target_from_route(ego_transform)
        if target is None:
            ok = self.build_route_from_ego()
            if not ok:
                self.publish_stop(reason='no_forward_target')
                return

            target = self.choose_target_from_route(ego_transform)
            if target is None:
                self.publish_stop(reason='no_forward_target_after_rebuild')
                return

        is_in_corner, local_x, local_y = target

        local_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, local_y))

        alpha = max(0.0, min(1.0, self.target_smoothing_alpha))
        if self.filtered_target_x is None or self.filtered_target_y is None:
            self.filtered_target_x = local_x
            self.filtered_target_y = local_y
        else:
            self.filtered_target_x = ((1.0 - alpha) * self.filtered_target_x) + (alpha * local_x)
            self.filtered_target_y = ((1.0 - alpha) * self.filtered_target_y) + (alpha * local_y)

        local_x = float(self.filtered_target_x)
        local_y = float(self.filtered_target_y)

        raw_desired_speed = self.calculate_dynamic_speed(local_y)
        speed_alpha = 0.10
        self.filtered_desired_speed = ((1.0 - speed_alpha) * self.filtered_desired_speed) + (speed_alpha * raw_desired_speed)

        speed_msg = Float64()
        speed_msg.data = float(self.filtered_desired_speed)
        self.desired_speed_pub.publish(speed_msg)

        target_msg = Point()
        target_msg.x = float(local_x)
        target_msg.y = float(local_y)
        target_msg.z = 0.0
        self.target_point_pub.publish(target_msg)

        state_msg = String()
        state_msg.data = 'LANE_KEEPING'
        self.state_pub.publish(state_msg)

        debug_msg = String()
        debug_msg.data = (
            f'state=LANE_KEEPING | '
            f'mode={"STRAIGHT" if not is_in_corner else "CORNER"} | '
            f'curvature={abs(local_y):.2f} | '
            f'desired_speed={self.filtered_desired_speed:.2f} | '
            f'local_target=({local_x:.2f},{local_y:.2f}) | '
            f'route_points={len(self.route_world)}'
        )
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointBehaviorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
