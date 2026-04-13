import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from time import time


# PID 계산기
# 입력: 속도 오차(error)
# 출력: 스로틀 명령(연속값)
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
        
        # ===== [제어기 설정] PID 게인 및 제어 주기 =====
        self.pid = PIDController(kp=0.5, ki=0.0, kd=0.1)
        self.control_dt = 0.1

        # ===== [입력 구독] 실제 속도 + 판단 노드 목표속도 =====
        self.speed_sub = self.create_subscription(
            Float64, '/carla/ego_vehicle/speedometer', self.speed_callback, 10)

        # behavior_node가 계산한 목표속도(m/s) 구독
        self.desired_speed_sub = self.create_subscription(
            Float64, '/desired_speed', self.desired_speed_callback, 10)

        # ===== [출력 발행] 하위 제어 어댑터로 전달할 명령 =====
        self.speed_pub = self.create_publisher(Float64, '/speed_command', 10)
        
        # 브레이크 신호 퍼블리시 (순수하게 브레이크용)
        self.brake_pub = self.create_publisher(Float64, '/brake_command', 10)
        
        # ===== [내부 상태] 최신 속도 및 목표 속도 =====
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.speed_ready = False
        self.target_ready = False
        self.waiting_log_printed = False
        
        # ===== [안전장치] 입력 타임아웃 시 비상정지 =====
        self.last_speed_update = time()
        self.last_desired_speed_update = time()
        self.sensor_timeout = 2.0  # 2초 타임아웃
        
        # 타이머 기반 PID 제어 (100ms 주기)
        self.control_timer = self.create_timer(0.1, self.periodic_control)
        
        self.get_logger().info('🚗 속도 제어 노드 가동! PID 컨트롤러 준비 완료 (타이머 주기: 100ms)')

    def speed_callback(self, msg):
        # CARLA 속도계 값(m/s) 갱신
        self.current_speed = msg.data  # m/s 단위 (CARLA 기본)
        self.speed_ready = True
        self.last_speed_update = time()
        self.get_logger().debug(f'📊 속도 수신: {self.current_speed:.2f} m/s')

    def desired_speed_callback(self, msg):
        # 행동 노드가 계산한 목표속도(m/s) 갱신
        self.target_speed = max(0.0, msg.data)
        self.target_ready = True
        self.last_desired_speed_update = time()
        self.get_logger().debug(f'🎯 목표속도 수신: {self.target_speed:.2f} m/s')

    def periodic_control(self):
        """타이머 기반 주기적 제어 (100ms)"""
        # 첫 메시지 수신 전에는 비상정지를 남발하지 않고 안전 대기
        if not self.speed_ready or not self.target_ready:
            if not self.waiting_log_printed:
                self.get_logger().warning(
                    '⏳ 입력 대기 중: /carla/ego_vehicle/speedometer 및 /desired_speed 첫 수신을 기다립니다.'
                )
                self.waiting_log_printed = True
            self.publish_safe_stop()
            return

        self.waiting_log_printed = False

        current_time = time()
        speed_timeout = current_time - self.last_speed_update > self.sensor_timeout
        target_timeout = current_time - self.last_desired_speed_update > self.sensor_timeout

        if speed_timeout:
            self.get_logger().warning('⚠️ 속도 센서 타임아웃! (2초 이상 신호 없음)')
        if target_timeout:
            self.get_logger().warning('⚠️ 목표 속도 타임아웃! (2초 이상 신호 없음)')

        # 센서 타임아웃 시 안전 정지
        if speed_timeout or target_timeout:
            self.emergency_stop()
            return
        
        # 속도 제어 업데이트
        self.update_speed_control()

    def update_speed_control(self):
        # ===== [핵심 제어] 목표속도-현재속도 오차를 PID로 스로틀화 =====
        # 속도 오차 계산
        error = self.target_speed - self.current_speed
        
        # PID로 스로틀 명령 계산 (0.0 ~ 1.0 범위)
        throttle_cmd = self.pid.compute(error, dt=self.control_dt)
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
        
        self.get_logger().debug(
            f'🏎️ 속도 제어: 현재 {self.current_speed:.2f} m/s, 목표 {self.target_speed:.2f} m/s, '
            f'스로틀 {throttle_cmd:.2f}, 브레이크 {brake_cmd:.2f}'
        )

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

    def publish_safe_stop(self):
        """입력 대기/정상 감속용 안전 정지"""
        brake_msg = Float64()
        brake_msg.data = 1.0
        self.brake_pub.publish(brake_msg)

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