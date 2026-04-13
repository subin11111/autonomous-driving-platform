import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, String
from carla_msgs.msg import CarlaEgoVehicleControl


# 제어 어댑터 노드
# 입력: target_point(판단), speed/brake_command(PID)
# 출력: CARLA 차량 제어 메시지(CarlaEgoVehicleControl)
class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('wheelbase', 2.9)
        self.declare_parameter('max_steering_angle_rad', 1.22)
        self.declare_parameter('steer_ema_alpha', 0.2)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.max_steering_angle_rad = float(self.get_parameter('max_steering_angle_rad').value)
        self.steer_ema_alpha = float(self.get_parameter('steer_ema_alpha').value)
        
        # ===== [입력 구독] 상위 판단/속도제어 결과 =====
        self.target_point_sub = self.create_subscription(
            Point,
            '/target_point',
            self.target_point_callback,
            10)

        self.speed_sub = self.create_subscription(
            Float64,
            '/speed_command',
            self.speed_callback,
            10)

        # 💥 NEW: 브레이크 신호 구독
        self.brake_sub = self.create_subscription(
            Float64,
            '/brake_command',
            self.brake_callback,
            10)

        # ===== [출력 발행] CARLA 제어 명령 =====
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10)
        self.status_pub = self.create_publisher(String, '/pure_pursuit_status', 10)

        # ===== [내부 상태] 최신 입력값 저장 =====
        self.latest_throttle = 0.0
        self.latest_brake = 1.0
        self.latest_steer = 0.0
        self.filtered_steer = 0.0
        self.steer_filter_initialized = False
        self.target_x = 0.0
        self.target_y = 0.0

        self.get_logger().info(
            f'⚙️ 제어 어댑터 노드 가동: target_point 기반 조향 (wheelbase={self.wheelbase:.2f} m, '
            f'EMA alpha={self.steer_ema_alpha:.2f})')

    def speed_callback(self, msg):
        # PID 스로틀 명령 업데이트
        self.latest_throttle = max(0.0, min(1.0, msg.data))
        self.get_logger().debug(f'🛣️ 속도명령 수신: 스로틀 {self.latest_throttle:.2f}')
        self.publish_control()

    def brake_callback(self, msg):
        # PID 브레이크 명령 업데이트
        self.latest_brake = max(0.0, min(1.0, msg.data))
        self.get_logger().debug(f'🛑 브레이크명령 수신: 브레이크 {self.latest_brake:.2f}')
        self.publish_control()

    def target_point_callback(self, msg):
        # 판단 노드 목표점 업데이트 후 퓨어퍼슛으로 조향 계산
        self.target_x = float(msg.x)
        self.target_y = float(msg.y)
        raw_steer = self.compute_steer_ratio(self.target_x, self.target_y)

        if not self.steer_filter_initialized:
            self.filtered_steer = raw_steer
            self.steer_filter_initialized = True
        else:
            alpha = max(0.0, min(1.0, self.steer_ema_alpha))
            self.filtered_steer = (alpha * raw_steer) + ((1.0 - alpha) * self.filtered_steer)

        self.latest_steer = self.filtered_steer
        self.get_logger().debug(
            f'🎯 목표점 수신: x={self.target_x:.2f}, y={self.target_y:.2f}, '
            f'raw_steer={raw_steer:.2f}, filtered_steer={self.latest_steer:.2f}')
        self.publish_control()

    def compute_steer_ratio(self, target_x, target_y):
        # 목표점(target_x, target_y)을 CARLA steer 비율(-1.0~1.0)로 변환
        l_d_square = (target_x ** 2) + (target_y ** 2)
        if l_d_square <= 1e-6 or self.max_steering_angle_rad <= 1e-6:
            return 0.0

        steering_angle_rad = math.atan((2.0 * self.wheelbase * target_y) / l_d_square)
        steer_ratio = -(steering_angle_rad / self.max_steering_angle_rad)
        return max(-1.0, min(1.0, steer_ratio))

    def publish_control(self):
        # ===== [명령 합성] 스로틀/브레이크 우선순위 적용 후 차량 명령 발행 =====
        control_msg = CarlaEgoVehicleControl()

        if self.latest_brake > 0.5:
            control_msg.throttle = 0.0
            control_msg.brake = self.latest_brake
        elif self.latest_throttle <= 0.01:
            control_msg.throttle = 0.0
            control_msg.brake = 1.0
        else:
            control_msg.throttle = self.latest_throttle
            control_msg.brake = 0.0

        control_msg.steer = self.latest_steer
        self.control_pub.publish(control_msg)

        status_msg = String()
        status_msg.data = f'RUN {control_msg.throttle:.2f} {control_msg.steer:.2f} {control_msg.brake:.2f}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()