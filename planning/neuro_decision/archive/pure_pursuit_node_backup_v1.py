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
        # perception 기반 target_point가 waypoint보다 민감할 수 있으므로, 
        # 최대 조향각을 충분히 크게 설정
        self.declare_parameter('max_steering_angle_rad', 0.75)

        # ===== Smoothing 및 주기 =====
        # steer_ema_alpha를 높여서 perception 노이즈 필터링 강화
        self.declare_parameter('steer_ema_alpha', 0.18)
        self.declare_parameter('control_period_s', 0.05)
        self.declare_parameter('command_timeout_s', 1.0)

        # ===== 조향 gain =====
        # steering_gain는 v3 기준으로 조정
        self.declare_parameter('steering_gain', 0.58)
        self.declare_parameter('left_turn_gain', 1.0)
        self.declare_parameter('right_turn_gain', 1.0)

        # ===== 조향 변화율 제한 =====
        # 급격한 조향 변화 방지 (perception 노이즈로부터 보호)
        self.declare_parameter('max_steer_delta_per_cycle', 0.040)

        # ===== 횡방향 target clamp =====
        # 너무 큰 측방 오차에 과민하게 반응하지 않도록 조절
        self.declare_parameter('target_y_clamp_m', 1.40)

        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle_rad = float(self.get_parameter('max_steering_angle_rad').value)

        self.steer_ema_alpha = float(self.get_parameter('steer_ema_alpha').value)
        self.control_period_s = float(self.get_parameter('control_period_s').value)
        self.command_timeout_s = float(self.get_parameter('command_timeout_s').value)

        self.steering_gain = float(self.get_parameter('steering_gain').value)
        self.left_turn_gain = float(self.get_parameter('left_turn_gain').value)
        self.right_turn_gain = float(self.get_parameter('right_turn_gain').value)

        self.max_steer_delta_per_cycle = float(self.get_parameter('max_steer_delta_per_cycle').value)
        self.target_y_clamp_m = float(self.get_parameter('target_y_clamp_m').value)

        # ===== 입력 구독 =====
        self.target_point_sub = self.create_subscription(
            Point,
            '/target_point',
            self.target_point_callback,
            10
        )
        self.speed_sub = self.create_subscription(
            Float64,
            '/speed_command',
            self.speed_callback,
            10
        )
        self.brake_sub = self.create_subscription(
            Float64,
            '/brake_command',
            self.brake_callback,
            10
        )

        # ===== 출력 발행 =====
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        self.status_pub = self.create_publisher(String, '/pure_pursuit_status', 10)

        # ===== 내부 상태 =====
        self.latest_throttle = 0.0
        self.latest_brake = 1.0
        self.latest_steer = 0.0

        self.filtered_steer = 0.0
        self.steer_filter_initialized = False

        self.target_x = 0.0
        self.target_y = 0.0

        now = monotonic()
        self.last_target_update = now
        self.last_speed_update = now
        self.last_brake_update = now

        self.timer = self.create_timer(self.control_period_s, self.publish_control)

        self.get_logger().info(
            f'⚙️ pure_pursuit_node 시작 (perception 기반): '
            f'wheelbase={self.wheelbase:.2f}m, '
            f'steer_EMA_alpha={self.steer_ema_alpha:.2f} (강화된 노이즈 필터), '
            f'steer_gain={self.steering_gain:.2f}, '
            f'max_steer={self.max_steering_angle_rad:.2f}rad, '
            f'max_steer_delta={self.max_steer_delta_per_cycle:.4f}/cycle'
        )

    def speed_callback(self, msg: Float64):
        self.latest_throttle = max(0.0, min(1.0, float(msg.data)))
        self.last_speed_update = monotonic()

    def brake_callback(self, msg: Float64):
        self.latest_brake = max(0.0, min(1.0, float(msg.data)))
        self.last_brake_update = monotonic()

    def target_point_callback(self, msg: Point):
        """
        Perception 기반 behavior_node로부터 target_point 수신
        - 이전 waypoint 기반보다 노이즈가 있을 수 있음
        - EMA 필터와 rate limit으로 노이즈 억제
        """
        self.target_x = float(msg.x)
        self.target_y = float(msg.y)
        self.last_target_update = monotonic()

        # Perception 기반 target에서 조향각 계산
        raw_steer = self.compute_steer_ratio(self.target_x, self.target_y)

        # ===== EMA 필터: 노이즈 필터링 =====
        # steer_ema_alpha가 높을수록 강한 필터 (perception 노이즈 억제)
        if not self.steer_filter_initialized:
            self.filtered_steer = raw_steer
            self.steer_filter_initialized = True
        else:
            alpha = max(0.0, min(1.0, self.steer_ema_alpha))
            ema_steer = (alpha * raw_steer) + ((1.0 - alpha) * self.filtered_steer)

            # ===== Rate limit: 급격한 조향 변화 방지 =====
            delta = ema_steer - self.filtered_steer
            delta = max(-self.max_steer_delta_per_cycle, min(self.max_steer_delta_per_cycle, delta))
            self.filtered_steer = self.filtered_steer + delta

        self.latest_steer = max(-1.0, min(1.0, self.filtered_steer))

    def compute_steer_ratio(self, target_x, target_y):
        """
        Perception 기반 target_point로부터 조향각 계산
        - target_y: 차량 기준 측방 오차 (음수=좌, 양수=우)
        - 작은 오차에는 민감하지 않고, 큰 오차에는 충분히 반응
        """
        # 너무 큰 횡방향 target은 clamp (과감한 회전 방지)
        target_y = max(-self.target_y_clamp_m, min(self.target_y_clamp_m, target_y))

        l_d_square = (target_x ** 2) + (target_y ** 2)

        if l_d_square <= 1e-6:
            return 0.0
        if self.max_steering_angle_rad <= 1e-6:
            return 0.0

        # Pure pursuit steering angle calculation
        # atan2(2*L*y, l_d^2) : L=wheelbase, y=lateral_offset, l_d=lookahead_distance
        steering_angle_rad = math.atan2(2.0 * self.wheelbase * target_y, l_d_square)

        # [-1, 1] 범위의 정규화 조향각
        steer_ratio = steering_angle_rad / self.max_steering_angle_rad

        # Steering gain 적용 (perception 기반 target의 민감도 조절)
        steer_ratio *= self.steering_gain

        return max(-1.0, min(1.0, steer_ratio))

    def publish_control(self):
        """
        최종 제어 신호 발행
        - timeout 감시로 안전성 확보
        - perception 기반 target의 불안정성에 대응하는 모든 필터 완료
        """
        now = monotonic()

        # ===== Timeout 감시: 신호 끊김 감지 =====
        target_stale = (now - self.last_target_update) > self.command_timeout_s
        speed_stale = (now - self.last_speed_update) > self.command_timeout_s
        brake_stale = (now - self.last_brake_update) > self.command_timeout_s

        control_msg = CarlaEgoVehicleControl()

        if target_stale or speed_stale or brake_stale:
            # 신호 손실 시 안전 정지
            control_msg.throttle = 0.0
            control_msg.brake = 1.0
            control_msg.steer = 0.0
            status_text = (
                f'SAFE_STOP | target_stale={target_stale} | '
                f'speed_stale={speed_stale} | brake_stale={brake_stale}'
            )
        else:
            # 정상 주행: brake 우선 적용 (throttle과 동시에 못함)
            if self.latest_brake > 0.01:
                control_msg.throttle = 0.0
                control_msg.brake = self.latest_brake
            else:
                control_msg.throttle = self.latest_throttle
                control_msg.brake = 0.0

            # 필터링된 조향각 적용
            control_msg.steer = self.latest_steer
            status_text = (
                f'RUN | throttle={control_msg.throttle:.2f} | '
                f'steer={control_msg.steer:.2f} | brake={control_msg.brake:.2f} | '
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