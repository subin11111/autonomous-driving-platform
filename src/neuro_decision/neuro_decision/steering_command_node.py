import math
from time import monotonic

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, String


class SteeringCommandNode(Node):
    def __init__(self):
        super().__init__('steering_command_node')

        # ===== 기본 파라미터 =====
        self.declare_parameter('wheelbase', 2.9)
        self.declare_parameter('max_steering_angle_rad', 0.75)
        self.declare_parameter('control_period_s', 0.05)
        self.declare_parameter('command_timeout_s', 1.0)
        self.declare_parameter('steer_ema_alpha', 0.18)
        self.declare_parameter('target_y_clamp_m', 1.40)

        # ===== 상태별 조향 gain =====
        self.declare_parameter('lk_steering_gain', 0.55)
        self.declare_parameter('lk_steer_delta_per_cycle', 0.035)
        self.declare_parameter('lk_steer_ema_alpha', 0.15)

        self.declare_parameter('fv_steering_gain', 0.56)
        self.declare_parameter('fv_steer_delta_per_cycle', 0.035)
        self.declare_parameter('fv_steer_ema_alpha', 0.15)

        self.declare_parameter('plc_steering_gain', 0.60)
        self.declare_parameter('plc_steer_delta_per_cycle', 0.045)
        self.declare_parameter('plc_steer_ema_alpha', 0.20)

        self.declare_parameter('lc_steering_gain', 0.65)
        self.declare_parameter('lc_steer_delta_per_cycle', 0.055)
        self.declare_parameter('lc_steer_ema_alpha', 0.22)

        self.declare_parameter('rtl_steering_gain', 0.52)
        self.declare_parameter('rtl_steer_delta_per_cycle', 0.040)
        self.declare_parameter('rtl_steer_ema_alpha', 0.18)

        # ===== 파라미터 로드 =====
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle_rad = float(self.get_parameter('max_steering_angle_rad').value)
        self.control_period_s = float(self.get_parameter('control_period_s').value)
        self.command_timeout_s = float(self.get_parameter('command_timeout_s').value)
        self.steer_ema_alpha = float(self.get_parameter('steer_ema_alpha').value)
        self.target_y_clamp_m = float(self.get_parameter('target_y_clamp_m').value)

        self.steering_profiles = {
            'LANE_KEEPING': {
                'gain': float(self.get_parameter('lk_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('lk_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('lk_steer_ema_alpha').value),
            },
            'FOLLOW_VEHICLE': {
                'gain': float(self.get_parameter('fv_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('fv_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('fv_steer_ema_alpha').value),
            },
            'PREPARE_LANE_CHANGE_LEFT': {
                'gain': float(self.get_parameter('plc_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('plc_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('plc_steer_ema_alpha').value),
            },
            'PREPARE_LANE_CHANGE_RIGHT': {
                'gain': float(self.get_parameter('plc_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('plc_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('plc_steer_ema_alpha').value),
            },
            'LANE_CHANGE_LEFT': {
                'gain': float(self.get_parameter('lc_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('lc_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('lc_steer_ema_alpha').value),
            },
            'LANE_CHANGE_RIGHT': {
                'gain': float(self.get_parameter('lc_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('lc_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('lc_steer_ema_alpha').value),
            },
            'RETURN_TO_LANE': {
                'gain': float(self.get_parameter('rtl_steering_gain').value),
                'delta_per_cycle': float(self.get_parameter('rtl_steer_delta_per_cycle').value),
                'ema_alpha': float(self.get_parameter('rtl_steer_ema_alpha').value),
            },
            'STOP': {
                'gain': 0.0,
                'delta_per_cycle': 0.0,
                'ema_alpha': 0.2,
            },
            'EMERGENCY_STOP': {
                'gain': 0.0,
                'delta_per_cycle': 0.0,
                'ema_alpha': 0.2,
            },
        }

        # ===== 입력 구독 =====
        self.target_point_sub = self.create_subscription(Point, '/target_point', self.target_point_callback, 10)
        self.behavior_state_sub = self.create_subscription(String, '/behavior_state', self.behavior_state_callback, 10)

        # ===== 출력 발행 =====
        self.desired_steering_angle_pub = self.create_publisher(Float64, '/desired_steering_angle_rad', 10)
        self.desired_steering_normalized_pub = self.create_publisher(Float64, '/desired_steering_normalized', 10)
        self.debug_pub = self.create_publisher(String, '/steering_command_debug_text', 10)

        # ===== 내부 상태 =====
        self.target_x = 0.0
        self.target_y = 0.0
        self.behavior_state = 'STOP'
        self.last_behavior_state = 'STOP'

        self.latest_raw_steering_angle_rad = 0.0
        self.latest_raw_steer_normalized = 0.0
        self.filtered_steering_angle_rad = 0.0
        self.filtered_steer_normalized = 0.0
        self.steer_filter_initialized = False
        self.state_changed = False

        now = monotonic()
        self.last_target_update = now
        self.last_state_update = now

        self.timer = self.create_timer(self.control_period_s, self.publish_steering_commands)

        self.get_logger().info('steering_command_node: steering-only interface node ready')

    def target_point_callback(self, msg: Point):
        self.target_x = float(msg.x)
        self.target_y = float(msg.y)
        self.last_target_update = monotonic()

    def behavior_state_callback(self, msg: String):
        new_state = msg.data.strip().upper()
        if new_state != self.behavior_state:
            self.state_changed = True
            self.last_behavior_state = self.behavior_state
            self.behavior_state = new_state
            self.steer_filter_initialized = False
            if self.behavior_state in ('STOP', 'EMERGENCY_STOP'):
                self.filtered_steering_angle_rad = 0.0
                self.filtered_steer_normalized = 0.0
                self.latest_raw_steering_angle_rad = 0.0
                self.latest_raw_steer_normalized = 0.0
            self.get_logger().info(f'State change: {self.last_behavior_state} -> {self.behavior_state}')
        else:
            self.state_changed = False
        self.last_state_update = monotonic()

    def get_steering_profile_for_state(self):
        if self.behavior_state in self.steering_profiles:
            return self.steering_profiles[self.behavior_state]
        return self.steering_profiles['LANE_KEEPING']

    def compute_steering_angle_rad(self, target_x, target_y):
        target_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, float(target_y)))
        l_d_square = (float(target_x) ** 2) + (target_y ** 2)

        if l_d_square <= 1e-6:
            return 0.0

        return math.atan2(2.0 * self.wheelbase * target_y, l_d_square)

    def compute_steer_ratio_from_angle(self, steering_angle_rad):
        if self.max_steering_angle_rad <= 1e-6:
            return 0.0
        return steering_angle_rad / self.max_steering_angle_rad

    def publish_steering_commands(self):
        now = monotonic()
        target_stale = (now - self.last_target_update) > self.command_timeout_s
        state_stale = (now - self.last_state_update) > self.command_timeout_s

        profile = self.get_steering_profile_for_state()
        steering_gain = float(profile['gain'])
        delta_per_cycle = float(profile['delta_per_cycle'])
        ema_alpha = max(0.0, min(1.0, float(profile['ema_alpha'])))

        if target_stale or state_stale:
            self.latest_raw_steering_angle_rad = 0.0
            self.latest_raw_steer_normalized = 0.0
            self.filtered_steering_angle_rad = 0.0
            self.filtered_steer_normalized = 0.0
            self.steer_filter_initialized = False
            status = 'SAFE_STEER_ZERO'
        else:
            if self.behavior_state in ('STOP', 'EMERGENCY_STOP'):
                raw_steering_angle_rad = 0.0
                raw_steer_normalized = 0.0
            else:
                raw_steering_angle_rad = self.compute_steering_angle_rad(self.target_x, self.target_y)
                raw_steer_normalized = self.compute_steer_ratio_from_angle(raw_steering_angle_rad)

            self.latest_raw_steering_angle_rad = raw_steering_angle_rad
            self.latest_raw_steer_normalized = max(-1.0, min(1.0, raw_steer_normalized * steering_gain))

            if not self.steer_filter_initialized:
                self.filtered_steer_normalized = self.latest_raw_steer_normalized
                self.steer_filter_initialized = True
            else:
                ema_steer = (ema_alpha * self.latest_raw_steer_normalized) + ((1.0 - ema_alpha) * self.filtered_steer_normalized)
                delta = ema_steer - self.filtered_steer_normalized
                delta = max(-delta_per_cycle, min(delta_per_cycle, delta))
                self.filtered_steer_normalized = self.filtered_steer_normalized + delta

            self.filtered_steer_normalized = max(-1.0, min(1.0, self.filtered_steer_normalized))
            self.filtered_steering_angle_rad = self.filtered_steer_normalized * self.max_steering_angle_rad
            status = 'STEER_ACTIVE'

        angle_msg = Float64()
        angle_msg.data = float(self.filtered_steering_angle_rad)
        self.desired_steering_angle_pub.publish(angle_msg)

        normalized_msg = Float64()
        normalized_msg.data = float(self.filtered_steer_normalized)
        self.desired_steering_normalized_pub.publish(normalized_msg)

        debug_msg = String()
        debug_msg.data = (
            f'state={self.behavior_state} | '
            f'status={status} | '
            f'target_x={self.target_x:.2f} | '
            f'target_y={self.target_y:.2f} | '
            f'raw_steering_angle_rad={self.latest_raw_steering_angle_rad:.4f} | '
            f'filtered_steering_angle_rad={self.filtered_steering_angle_rad:.4f} | '
            f'normalized_steer={self.filtered_steer_normalized:.4f} | '
            f'steering_gain={steering_gain:.3f} | '
            f'delta_per_cycle={delta_per_cycle:.4f} | '
            f'ema_alpha={ema_alpha:.3f} | '
            f'target_stale={target_stale} | '
            f'state_stale={state_stale} | '
            f'max_steering_angle_rad={self.max_steering_angle_rad:.3f}'
        )
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SteeringCommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()