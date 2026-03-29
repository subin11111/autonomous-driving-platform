import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, String
from carla_msgs.msg import CarlaEgoVehicleControl
import math

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # 1. 뇌(behavior_node)가 쏘는 목표 좌표(X, Y) 구독
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
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

        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10)
        self.status_pub = self.create_publisher(String, '/pure_pursuit_status', 10)

        self.latest_throttle = 0.3
        self.latest_brake = 0.0  # 💥 NEW: 브레이크 상태 저장

        self.get_logger().info('⚙️ 제어 파트 (Pure Pursuit) 재가동! 뇌의 명령 대기 중...')

    def speed_callback(self, msg):
        self.latest_throttle = max(0.0, min(1.0, msg.data))
        self.get_logger().debug(f'🛣️ 속도명령 수신: 스로틀 {self.latest_throttle:.2f}')

    # 💥 NEW: 브레이크 콜백 추가
    def brake_callback(self, msg):
        self.latest_brake = max(0.0, min(1.0, msg.data))
        self.get_logger().debug(f'🛑 브레이크명령 수신: 브레이크 {self.latest_brake:.2f}')

    def goal_callback(self, msg):
        target_x = msg.pose.position.x  
        target_y = msg.pose.position.y  
        
        control_msg = CarlaEgoVehicleControl()

        # 🚨 [STOP 상태 감지] 뇌가 (0.0, 0.0)을 보냈다면? = 앞에 차가 있다!
        if target_x == 0.0 and target_y == 0.0:
            control_msg.throttle = 0.0  # 엑셀 발 떼고
            control_msg.steer = 0.0     # 핸들 정중앙
            control_msg.brake = 1.0     # 브레이크 풀파워! (1.0)

            self.get_logger().warning('🛑 [정지 명령 수신] 앞차 감지! 엑셀 오프 & 브레이크 작동!')
            self.control_pub.publish(control_msg)

            status_msg = String()
            status_msg.data = 'STOP'
            self.status_pub.publish(status_msg)
            return  # 브레이크 밟았으니 아래 조향각 계산은 건너뜀

        # 🟢 [LANE_KEEPING 상태] 목표점이 정상적으로 들어왔다면 차선 유지 공식 가동
        L = 2.9  # 휠베이스
        l_d_square = (target_x ** 2) + (target_y ** 2)
        
        if l_d_square == 0:
            return
            
        # Pure Pursuit 조향각 계산 공식
        steering_angle_rad = math.atan((2 * L * target_y) / l_d_square)
        
        # CARLA 비율로 변환 (역방향 주의)
        steer_ratio = -(steering_angle_rad / 1.22)
        steer_ratio = max(-1.0, min(1.0, steer_ratio))
        
        # 💥 개선: 브레이크 신호 적용
        # 제어 명령 포장하기
        if self.latest_brake > 0.5:  # 브레이크가 50% 이상
            control_msg.throttle = 0.0
            control_msg.brake = self.latest_brake
            self.get_logger().info(f'🛑 브레이크 적용 중: {self.latest_brake:.2f}')
        elif self.latest_throttle <= 0.01:
            control_msg.throttle = 0.0
            control_msg.brake = 1.0
            self.get_logger().info('🛑 속도명령 0으로 감지됨: 브레이크 적용')
        else:
            control_msg.throttle = self.latest_throttle
            control_msg.brake = 0.0

        control_msg.steer = steer_ratio

        self.get_logger().info(f'🚗 [주행 중] 전방: {target_x:.1f}m, 조향각: {steer_ratio:.2f}, 스로틀: {control_msg.throttle:.2f}, 브레이크: {control_msg.brake:.2f}')
        self.control_pub.publish(control_msg)

        status_msg = String()
        status_msg.data = f'RUN {control_msg.throttle:.2f} {steer_ratio:.2f} {control_msg.brake:.2f}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()