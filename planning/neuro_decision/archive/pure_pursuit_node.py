import math
from time import monotonic

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, String
from carla_msgs.msg import CarlaEgoVehicleControl


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ===== 기본 파라미터 =====
        self.declare_parameter('wheelbase', 2.9)

        # ===== Steering 범위 제한 =====
        self.declare_parameter('max_steering_angle_rad', 0.75)

        # ===== Smoothing 및 주기 =====
        self.declare_parameter('steer_ema_alpha', 0.18)
        self.declare_parameter('control_period_s', 0.05)
        self.declare_parameter('command_timeout_s', 1.0)

        # ===== 상태별 조향 gain =====
        # LANE_KEEPING: 보수적
        self.declare_parameter('lk_steering_gain', 0.55)
        self.declare_parameter('lk_steer_delta_per_cycle', 0.035)
        self.declare_parameter('lk_steer_ema_alpha', 0.15)

        # FOLLOW_VEHICLE: lane keeping과 유사
        self.declare_parameter('fv_steering_gain', 0.56)
        self.declare_parameter('fv_steer_delta_per_cycle', 0.035)
        self.declare_parameter('fv_steer_ema_alpha', 0.15)

        # PREPARE_LANE_CHANGE: 약간 더 빠른 반응
        self.declare_parameter('plc_steering_gain', 0.60)
        self.declare_parameter('plc_steer_delta_per_cycle', 0.045)
        self.declare_parameter('plc_steer_ema_alpha', 0.20)

        # LANE_CHANGE: 강한 반응
        self.declare_parameter('lc_steering_gain', 0.65)
        self.declare_parameter('lc_steer_delta_per_cycle', 0.055)
        self.declare_parameter('lc_steer_ema_alpha', 0.22)

        # RETURN_TO_LANE: 과조향 없이 복귀
        self.declare_parameter('rtl_steering_gain', 0.52)
        self.declare_parameter('rtl_steer_delta_per_cycle', 0.040)
        self.declare_parameter('rtl_steer_ema_alpha', 0.18)

        # ===== 공통 파라미터 =====
        self.declare_parameter('target_y_clamp_m', 1.40)
        self.declare_parameter('left_turn_gain', 1.0)
        self.declare_parameter('right_turn_gain', 1.0)

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle_rad = float(self.get_parameter('max_steering_angle_rad').value)

        self.steer_ema_alpha = float(self.get_parameter('steer_ema_alpha').value)
        self.control_period_s = float(self.get_parameter('control_period_s').value)
        self.command_timeout_s = float(self.get_parameter('command_timeout_s').value)

        self.target_y_clamp_m = float(self.get_parameter('target_y_clamp_m').value)
        self.left_turn_gain = float(self.get_parameter('left_turn_gain').value)
        self.right_turn_gain = float(self.get_parameter('right_turn_gain').value)

        # State-specific steering profiles
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
        self.speed_sub = self.create_subscription(Float64, '/speed_command', self.speed_callback, 10)
        self.brake_sub = self.create_subscription(Float64, '/brake_command', self.brake_callback, 10)
        self.behavior_state_sub = self.create_subscription(String, '/behavior_state', self.behavior_state_callback, 10)

        # ===== 출력 발행 =====
        self.control_pub = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        self.status_pub = self.create_publisher(String, '/pure_pursuit_status', 10)

        # ===== 내부 상태 =====
        self.latest_throttle = 0.0
        self.latest_brake = 1.0
        self.latest_steer = 0.0

        self.filtered_steer = 0.0
        self.steer_filter_initialized = False

        self.target_x = 0.0
        self.target_y = 0.0
        self.behavior_state = 'STOP'

        self.last_behavior_state = 'STOP'
        self.state_changed = False

        now = monotonic()
        self.last_target_update = now
        self.last_speed_update = now
        self.last_brake_update = now
        self.last_state_update = now

        self.timer = self.create_timer(self.control_period_s, self.publish_control)

        self.get_logger().info(
            f'⚙️ pure_pursuit_node v2: state-based steering parameter adaptation'
        )

    def speed_callback(self, msg: Float64):
        self.latest_throttle = max(0.0, min(1.0, float(msg.data)))
        self.last_speed_update = monotonic()

    def brake_callback(self, msg: Float64):
        self.latest_brake = max(0.0, min(1.0, float(msg.data)))
        self.last_brake_update = monotonic()

    def target_point_callback(self, msg: Point):
        """Perception 기반 behavior_node로부터 target_point 수신"""
        self.target_x = float(msg.x)
        self.target_y = float(msg.y)
        self.last_target_update = monotonic()

        # Steering 계산 (상태별 파라미터 적용)
        profile = self.get_steering_profile_for_state()
        raw_steer = self.compute_steer_ratio(self.target_x, self.target_y, profile['gain'])

        # ===== EMA 필터 적용 =====
        ema_alpha = profile['ema_alpha']
        if not self.steer_filter_initialized:
            self.filtered_steer = raw_steer
            self.steer_filter_initialized = True
        else:
            alpha = max(0.0, min(1.0, ema_alpha))
            ema_steer = (alpha * raw_steer) + ((1.0 - alpha) * self.filtered_steer)

            # ===== Rate limit 적용 =====
            max_delta = profile['delta_per_cycle']
            delta = ema_steer - self.filtered_steer
            delta = max(-max_delta, min(max_delta, delta))
            self.filtered_steer = self.filtered_steer + delta

        self.latest_steer = max(-1.0, min(1.0, self.filtered_steer))

    def behavior_state_callback(self, msg: String):
        """Behavior state 수신 및 변화 감지"""
        new_state = msg.data.strip().upper()
        if new_state != self.behavior_state:
            self.state_changed = True
            self.last_behavior_state = self.behavior_state
            self.behavior_state = new_state
            self.get_logger().info(f'🔄 State change: {self.last_behavior_state} -> {self.behavior_state}')
            # 상태 변화 시 필터 리셋
            if self.behavior_state in ('STOP', 'EMERGENCY_STOP'):
                self.filtered_steer = 0.0
        else:
            self.state_changed = False
        self.last_state_update = monotonic()

    def get_steering_profile_for_state(self):
        """현재 state에 해당하는 steering profile 반환"""
        if self.behavior_state in self.steering_profiles:
            return self.steering_profiles[self.behavior_state]
        # Default to LANE_KEEPING
        return self.steering_profiles['LANE_KEEPING']

    def compute_steer_ratio(self, target_x, target_y, steering_gain):
        """Pure pursuit steering angle 계산 (gain 적용)"""
        target_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, target_y))

        l_d_square = (target_x ** 2) + (target_y ** 2)

        if l_d_square <= 1e-6:
            return 0.0
        if self.max_steering_angle_rad <= 1e-6:
            return 0.0

        steering_angle_rad = math.atan2(2.0 * self.wheelbase * target_y, l_d_square)
        steer_ratio = steering_angle_rad / self.max_steering_angle_rad

        # 상태별 gain 적용
        steer_ratio *= steering_gain

        return max(-1.0, min(1.0, steer_ratio))

    def publish_control(self):
        """최종 제어 신호 발행"""
        now = monotonic()

        # ===== Timeout 감시 =====
        target_stale = (now - self.last_target_update) > self.command_timeout_s
        speed_stale = (now - self.last_speed_update) > self.command_timeout_s
        brake_stale = (now - self.last_brake_update) > self.command_timeout_s
        state_stale = (now - self.last_state_update) > self.command_timeout_s

        control_msg = CarlaEgoVehicleControl()

        if target_stale or speed_stale or brake_stale or state_stale:
            # 신호 손실 시 안전 정지
            control_msg.throttle = 0.0
            control_msg.brake = 1.0
            control_msg.steer = 0.0
            status_text = (
                f'SAFE_STOP | target_stale={target_stale} | '
                f'speed_stale={speed_stale} | brake_stale={brake_stale} | state_stale={state_stale}'
            )
        else:
            # 정상 주행
            if self.latest_brake > 0.01:
                control_msg.throttle = 0.0
                control_msg.brake = self.latest_brake
            else:
                control_msg.throttle = self.latest_throttle
                control_msg.brake = 0.0

            # 필터링된 조향각 적용
            control_msg.steer = self.latest_steer

            profile = self.get_steering_profile_for_state()
            status_text = (
                f'state={self.behavior_state} | '
                f'throttle={control_msg.throttle:.2f} | '
                f'brake={control_msg.brake:.2f} | '
                f'steer={control_msg.steer:.2f} | '
                f'steer_gain={profile["gain"]:.2f} | '
                f'target=({self.target_x:.2f},{self.target_y:.2f})'
            )

        control_msg.hand_brake = False
        control_msg.reverse = False
        control_msg.manual_gear_shift = False

        self.control_pub.publish(control_msg)

        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
