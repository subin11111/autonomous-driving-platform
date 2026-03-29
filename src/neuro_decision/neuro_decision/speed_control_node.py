import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math
from time import time

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt=0.1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class SpeedControlNode(Node):
    def __init__(self):
        super().__init__('speed_control_node')
        
        # PID 컨트롤러 초기화 (스로틀용)
        self.pid = PIDController(kp=0.5, ki=0.0, kd=0.1)
        
        # 현재 속도 구독 (CARLA 속도미터)
        self.speed_sub = self.create_subscription(
            Float64, '/carla/ego_vehicle/speedometer', self.speed_callback, 10)

        # 장애물 거리 구독 (behavior_node에서 직접 전달)
        self.obstacle_sub = self.create_subscription(
            Float64, '/obstacle_distance', self.obstacle_callback, 10)

        # 속도 명령 퍼블리시 (pure_pursuit_node에 전달)
        self.speed_pub = self.create_publisher(Float64, '/speed_command', 10)
        
        # 브레이크 신호 퍼블리시 (순수하게 브레이크용)
        self.brake_pub = self.create_publisher(Float64, '/brake_command', 10)
        
        # 현재 상태
        self.current_speed = 0.0
        self.target_speed = 0.5  # 기본 목표 속도 (0.5 = 50%)
        self.obstacle_distance = 99.0  # 장애물 거리 (behavior_node에서 추정)
        
        # 타임아웃 관리
        self.last_speed_update = time()
        self.last_obstacle_update = time()
        self.sensor_timeout = 2.0  # 2초 타임아웃
        
        # 타이머 기반 PID 제어 (100ms 주기)
        self.control_timer = self.create_timer(0.1, self.periodic_control)
        
        self.get_logger().info('🚗 속도 제어 노드 가동! PID 컨트롤러 준비 완료 (타이머 주기: 100ms)')

    def speed_callback(self, msg):
        self.current_speed = msg.data  # m/s 단위 (CARLA 기본)
        self.last_speed_update = time()
        self.get_logger().debug(f'📊 속도 수신: {self.current_speed:.2f} m/s')

    def obstacle_callback(self, msg):
        self.obstacle_distance = msg.data
        self.last_obstacle_update = time()

        if self.obstacle_distance < 10.0:
            self.target_speed = 0.0
        elif self.obstacle_distance < 20.0:
            self.target_speed = 0.2
        else:
            self.target_speed = 0.5

        self.get_logger().debug(f'🗺️ 장애물 거리: {self.obstacle_distance:.2f} m, 목표속도: {self.target_speed:.2f}')

    def periodic_control(self):
        """타이머 기반 주기적 제어 (100ms)"""
        # 📌 센서 타임아웃 확인 (주석: 비상 정지 로직 비활성화)
        # current_time = time()
        # speed_timeout = current_time - self.last_speed_update > self.sensor_timeout
        # obstacle_timeout = current_time - self.last_obstacle_update > self.sensor_timeout
        
        # if speed_timeout:
        #     self.get_logger().warning('⚠️ 속도 센서 타임아웃! (2초 이상 신호 없음)')
        # if obstacle_timeout:
        #     self.get_logger().warning('⚠️ 장애물 센서 타임아웃! (2초 이상 신호 없음)')
        
        # # 센서 타임아웃 시 안전 정지
        # if speed_timeout or obstacle_timeout:
        #     self.emergency_stop()
        #     return
        
        # 속도 제어 업데이트
        self.update_speed_control()

    def update_speed_control(self):
        # 속도 오차 계산
        error = self.target_speed - self.current_speed
        
        # PID로 스로틀 명령 계산 (0.0 ~ 1.0 범위)
        throttle_cmd = self.pid.compute(error, dt=0.1)
        throttle_cmd = max(0.0, min(1.0, throttle_cmd))  # 클램핑
        
        # 브레이크 명령
        brake_cmd = 1.0 if self.target_speed == 0.0 else 0.0
        
        # 속도 명령 퍼블리시 (Float64: 스로틀 값)
        speed_msg = Float64()
        speed_msg.data = throttle_cmd
        self.speed_pub.publish(speed_msg)
        
        # 브레이크 명령 퍼블리시 (Float64: 0.0 또는 1.0)
        brake_msg = Float64()
        brake_msg.data = brake_cmd
        self.brake_pub.publish(brake_msg)
        
        self.get_logger().debug(f'🏎️ 속도 제어: 현재 {self.current_speed:.2f} m/s, 목표 {self.target_speed:.2f}, 스로틀 {throttle_cmd:.2f}, 브레이크 {brake_cmd:.2f}')

    def emergency_stop(self):
        """비상 정지"""
        self.get_logger().error('🚨 비상 정지 작동!')
        
        # 브레이크 풀파워
        brake_msg = Float64()
        brake_msg.data = 1.0
        self.brake_pub.publish(brake_msg)
        
        # 스로틀 0
        speed_msg = Float64()
        speed_msg.data = 0.0
        self.speed_pub.publish(speed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpeedControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()