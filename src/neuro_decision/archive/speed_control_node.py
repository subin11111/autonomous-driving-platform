from time import monotonic

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32, String


class PIDController:
    def __init__(self, kp=0.5, ki=0.0, kd=0.1, integral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt=0.1):
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        derivative = 0.0
        if dt > 1e-6:
            derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0


class SpeedControlNode(Node):
    def __init__(self):
        super().__init__('speed_control_node')

        # ===== 제어 주기 및 타임아웃 =====
        self.declare_parameter('control_period_s', 0.1)
        self.declare_parameter('sensor_timeout_s', 2.0)

        # ===== Smoothing 파라미터 =====
        self.declare_parameter('throttle_filter_alpha', 0.08)
        self.declare_parameter('brake_filter_alpha', 0.10)

        # ===== 상태별 throttle 프로필 =====
        # LANE_KEEPING: 정상 주행
        self.declare_parameter('lk_throttle_fast', 0.40)
        self.declare_parameter('lk_throttle_medium', 0.32)
        self.declare_parameter('lk_throttle_hold', 0.22)
        self.declare_parameter('lk_throttle_trim', 0.10)
        self.declare_parameter('lk_throttle_min', 0.04)

        # FOLLOW_VEHICLE: 추종 (throttle 최소화)
        self.declare_parameter('fv_throttle_fast', 0.30)
        self.declare_parameter('fv_throttle_medium', 0.24)
        self.declare_parameter('fv_throttle_hold', 0.16)
        self.declare_parameter('fv_throttle_trim', 0.08)
        self.declare_parameter('fv_throttle_min', 0.02)

        # PREPARE_LANE_CHANGE: 감속 준비
        self.declare_parameter('plc_throttle_fast', 0.25)
        self.declare_parameter('plc_throttle_medium', 0.18)
        self.declare_parameter('plc_throttle_hold', 0.12)
        self.declare_parameter('plc_throttle_trim', 0.06)
        self.declare_parameter('plc_throttle_min', 0.02)

        # LANE_CHANGE: 안전한 낮은 속도
        self.declare_parameter('lc_throttle_fast', 0.20)
        self.declare_parameter('lc_throttle_medium', 0.14)
        self.declare_parameter('lc_throttle_hold', 0.10)
        self.declare_parameter('lc_throttle_trim', 0.05)
        self.declare_parameter('lc_throttle_min', 0.02)

        # RETURN_TO_LANE: 차선 복귀
        self.declare_parameter('rtl_throttle_fast', 0.18)
        self.declare_parameter('rtl_throttle_medium', 0.12)
        self.declare_parameter('rtl_throttle_hold', 0.08)
        self.declare_parameter('rtl_throttle_trim', 0.04)
        self.declare_parameter('rtl_throttle_min', 0.02)

        # ===== 정지 브레이크 제어 =====
        self.declare_parameter('stop_brake_high_speed', 0.35)
        self.declare_parameter('stop_brake_low_speed', 0.75)

        # ===== 상태별 브레이크 프로필 =====
        # 추종 상태: brake 최소화
        self.declare_parameter('fv_brake_normal', 0.0)
        # 차선 변경: 약간의 brake
        self.declare_parameter('lc_brake_normal', 0.05)
        # 정상 상태: brake 최소
        self.declare_parameter('normal_brake_overspeed', 0.0)

        # ===== Soft-start (부드러운 출발) =====
        self.declare_parameter('launch_speed_threshold', 0.15)
        self.declare_parameter('launch_throttle', 0.08)
        self.declare_parameter('launch_throttle_max', 0.12)
        self.declare_parameter('throttle_filter_alpha_launch', 0.06)
        self.declare_parameter('launch_duration_s', 1.0)

        # 파라미터 로드
        self.control_period_s = float(self.get_parameter('control_period_s').value)
        self.sensor_timeout_s = float(self.get_parameter('sensor_timeout_s').value)
        self.throttle_filter_alpha = float(self.get_parameter('throttle_filter_alpha').value)
        self.brake_filter_alpha = float(self.get_parameter('brake_filter_alpha').value)

        # State-specific throttle profiles
        self.throttle_profiles = {
            'LANE_KEEPING': {
                'fast': float(self.get_parameter('lk_throttle_fast').value),
                'medium': float(self.get_parameter('lk_throttle_medium').value),
                'hold': float(self.get_parameter('lk_throttle_hold').value),
                'trim': float(self.get_parameter('lk_throttle_trim').value),
                'min': float(self.get_parameter('lk_throttle_min').value),
            },
            'FOLLOW_VEHICLE': {
                'fast': float(self.get_parameter('fv_throttle_fast').value),
                'medium': float(self.get_parameter('fv_throttle_medium').value),
                'hold': float(self.get_parameter('fv_throttle_hold').value),
                'trim': float(self.get_parameter('fv_throttle_trim').value),
                'min': float(self.get_parameter('fv_throttle_min').value),
            },
            'PREPARE_LANE_CHANGE_LEFT': {
                'fast': float(self.get_parameter('plc_throttle_fast').value),
                'medium': float(self.get_parameter('plc_throttle_medium').value),
                'hold': float(self.get_parameter('plc_throttle_hold').value),
                'trim': float(self.get_parameter('plc_throttle_trim').value),
                'min': float(self.get_parameter('plc_throttle_min').value),
            },
            'PREPARE_LANE_CHANGE_RIGHT': {
                'fast': float(self.get_parameter('plc_throttle_fast').value),
                'medium': float(self.get_parameter('plc_throttle_medium').value),
                'hold': float(self.get_parameter('plc_throttle_hold').value),
                'trim': float(self.get_parameter('plc_throttle_trim').value),
                'min': float(self.get_parameter('plc_throttle_min').value),
            },
            'LANE_CHANGE_LEFT': {
                'fast': float(self.get_parameter('lc_throttle_fast').value),
                'medium': float(self.get_parameter('lc_throttle_medium').value),
                'hold': float(self.get_parameter('lc_throttle_hold').value),
                'trim': float(self.get_parameter('lc_throttle_trim').value),
                'min': float(self.get_parameter('lc_throttle_min').value),
            },
            'LANE_CHANGE_RIGHT': {
                'fast': float(self.get_parameter('lc_throttle_fast').value),
                'medium': float(self.get_parameter('lc_throttle_medium').value),
                'hold': float(self.get_parameter('lc_throttle_hold').value),
                'trim': float(self.get_parameter('lc_throttle_trim').value),
                'min': float(self.get_parameter('lc_throttle_min').value),
            },
            'RETURN_TO_LANE': {
                'fast': float(self.get_parameter('rtl_throttle_fast').value),
                'medium': float(self.get_parameter('rtl_throttle_medium').value),
                'hold': float(self.get_parameter('rtl_throttle_hold').value),
                'trim': float(self.get_parameter('rtl_throttle_trim').value),
                'min': float(self.get_parameter('rtl_throttle_min').value),
            },
        }

        self.stop_brake_high_speed = float(self.get_parameter('stop_brake_high_speed').value)
        self.stop_brake_low_speed = float(self.get_parameter('stop_brake_low_speed').value)
        self.fv_brake_normal = float(self.get_parameter('fv_brake_normal').value)
        self.lc_brake_normal = float(self.get_parameter('lc_brake_normal').value)
        self.normal_brake_overspeed = float(self.get_parameter('normal_brake_overspeed').value)

        self.launch_speed_threshold = float(self.get_parameter('launch_speed_threshold').value)
        self.launch_throttle = float(self.get_parameter('launch_throttle').value)
        self.launch_throttle_max = float(self.get_parameter('launch_throttle_max').value)
        self.throttle_filter_alpha_launch = float(self.get_parameter('throttle_filter_alpha_launch').value)
        self.launch_duration_s = float(self.get_parameter('launch_duration_s').value)

        self.pid = PIDController(kp=0.5, ki=0.0, kd=0.1)

        self.speed_sub = self.create_subscription(Float32, '/carla/ego_vehicle/speedometer', self.speed_callback, 10)
        self.desired_speed_sub = self.create_subscription(Float64, '/desired_speed', self.desired_speed_callback, 10)
        self.behavior_state_sub = self.create_subscription(String, '/behavior_state', self.behavior_state_callback, 10)

        self.speed_pub = self.create_publisher(Float64, '/speed_command', 10)
        self.brake_pub = self.create_publisher(Float64, '/brake_command', 10)
        self.debug_pub = self.create_publisher(String, '/speed_control_debug_text', 10)

        self.current_speed = 0.0
        self.target_speed = 0.0
        self.behavior_state = 'STOP'

        self.speed_ready = False
        self.target_ready = False
        self.state_ready = False
        self.waiting_log_printed = False

        now = monotonic()
        self.last_speed_update = now
        self.last_desired_speed_update = now
        self.last_behavior_state_update = now

        self.filtered_throttle_cmd = 0.0
        self.filtered_brake_cmd = 0.0
        self.launch_start_time = None

        self.timer = self.create_timer(self.control_period_s, self.periodic_control)

        self.get_logger().info(
            f'🚗 speed_control_node v2: state-based longitudinal profile + advanced smoothing'
        )

    def speed_callback(self, msg: Float32):
        self.current_speed = float(msg.data)
        self.speed_ready = True
        self.last_speed_update = monotonic()

    def desired_speed_callback(self, msg: Float64):
        self.target_speed = max(0.0, float(msg.data))
        self.target_ready = True
        self.last_desired_speed_update = monotonic()

    def behavior_state_callback(self, msg: String):
        self.behavior_state = msg.data.strip().upper()
        self.state_ready = True
        self.last_behavior_state_update = monotonic()

    def periodic_control(self):
        if not (self.speed_ready and self.target_ready and self.state_ready):
            if not self.waiting_log_printed:
                self.get_logger().warning(
                    '⏳ 입력 대기 중: speedometer / desired_speed / behavior_state 첫 수신 대기'
                )
                self.waiting_log_printed = True
            self.publish_safe_stop(reason='waiting_for_first_input')
            return

        self.waiting_log_printed = False

        now = monotonic()
        speed_timeout = (now - self.last_speed_update) > self.sensor_timeout_s
        target_timeout = (now - self.last_desired_speed_update) > self.sensor_timeout_s
        state_timeout = (now - self.last_behavior_state_update) > self.sensor_timeout_s

        if speed_timeout or target_timeout or state_timeout:
            reason = (
                f'speed_timeout={speed_timeout}, '
                f'target_timeout={target_timeout}, '
                f'state_timeout={state_timeout}'
            )
            self.emergency_stop(reason=reason)
            return

        self.update_speed_control()

    def get_throttle_profile_for_state(self):
        """현재 state에 해당하는 throttle profile 반환"""
        if self.behavior_state in self.throttle_profiles:
            return self.throttle_profiles[self.behavior_state]
        # Default to LANE_KEEPING
        return self.throttle_profiles['LANE_KEEPING']

    def update_speed_control(self):
        now = monotonic()
        error = self.target_speed - self.current_speed

        raw_throttle = 0.0
        raw_brake = 0.0
        launch_active = False
        launch_elapsed = 0.0

        # ===== 상태별 제어 =====

        if self.behavior_state == 'EMERGENCY_STOP':
            self.pid.reset()
            raw_throttle = 0.0
            raw_brake = self.stop_brake_low_speed
            self.launch_start_time = None

        elif self.behavior_state == 'STOP' or self.target_speed <= 0.01:
            self.pid.reset()
            raw_throttle = 0.0
            raw_brake = (
                self.stop_brake_high_speed if self.current_speed > 0.25 else self.stop_brake_low_speed
            )
            self.launch_start_time = None

        else:
            # Soft-start 판정
            if self.launch_start_time is None and self.current_speed < self.launch_speed_threshold:
                self.launch_start_time = now

            if self.launch_start_time is not None:
                launch_elapsed = now - self.launch_start_time
                launch_active = (
                    launch_elapsed < self.launch_duration_s
                    and self.current_speed < (self.launch_speed_threshold + 0.10)
                )

            if launch_active:
                if error > 0.30:
                    launch_cmd = self.launch_throttle + 0.03
                elif error > 0.15:
                    launch_cmd = self.launch_throttle + 0.02
                elif error >= -0.05:
                    launch_cmd = self.launch_throttle + 0.01
                else:
                    launch_cmd = 0.05  # trim

                raw_throttle = max(self.launch_throttle * 0.5, min(self.launch_throttle_max, launch_cmd))
                raw_brake = 0.0
            else:
                self.launch_start_time = None

                # 상태별 throttle profile 적용
                profile = self.get_throttle_profile_for_state()

                if error > 0.30:
                    raw_throttle = profile['fast']
                elif error > 0.15:
                    raw_throttle = profile['medium']
                elif error >= -0.05:
                    raw_throttle = profile['hold']
                elif error >= -0.20:
                    raw_throttle = profile['trim']
                else:
                    raw_throttle = profile['min']

                # 상태별 brake profile 적용
                if self.behavior_state == 'FOLLOW_VEHICLE':
                    raw_brake = self.fv_brake_normal
                elif self.behavior_state in ('LANE_CHANGE_LEFT', 'LANE_CHANGE_RIGHT', 'RETURN_TO_LANE', 
                                             'PREPARE_LANE_CHANGE_LEFT', 'PREPARE_LANE_CHANGE_RIGHT'):
                    raw_brake = self.lc_brake_normal
                else:
                    raw_brake = self.normal_brake_overspeed

        # 안전 범위 clamp
        raw_throttle = max(0.0, min(1.0, raw_throttle))
        raw_brake = max(0.0, min(1.0, raw_brake))

        # ===== Exponential smoothing =====
        throttle_alpha_normal = max(0.0, min(1.0, self.throttle_filter_alpha))
        throttle_alpha_launch = max(0.0, min(1.0, self.throttle_filter_alpha_launch))
        throttle_alpha = throttle_alpha_launch if launch_active else throttle_alpha_normal
        brake_alpha = max(0.0, min(1.0, self.brake_filter_alpha))

        target_throttle = (
            (1.0 - throttle_alpha) * self.filtered_throttle_cmd + throttle_alpha * raw_throttle
        )
        target_brake = (
            (1.0 - brake_alpha) * self.filtered_brake_cmd + brake_alpha * raw_brake
        )

        self.filtered_throttle_cmd = target_throttle
        self.filtered_brake_cmd = target_brake

        self.publish_commands(
            self.filtered_throttle_cmd,
            self.filtered_brake_cmd,
            error,
            launch_active=launch_active,
            launch_elapsed=launch_elapsed,
        )

    def publish_commands(self, throttle_cmd, brake_cmd, error, launch_active=False, launch_elapsed=0.0):
        speed_msg = Float64()
        speed_msg.data = float(max(0.0, min(1.0, throttle_cmd)))
        self.speed_pub.publish(speed_msg)

        brake_msg = Float64()
        brake_msg.data = float(max(0.0, min(1.0, brake_cmd)))
        self.brake_pub.publish(brake_msg)

        debug_msg = String()
        debug_msg.data = (
            f'state={self.behavior_state} | '
            f'current_speed={self.current_speed:.2f} | '
            f'target_speed={self.target_speed:.2f} | '
            f'error={error:.2f} | '
            f'throttle={throttle_cmd:.2f} | '
            f'brake={brake_cmd:.2f} | '
            f'launch_active={launch_active} | '
            f'launch_elapsed={launch_elapsed:.2f} | '
            f'filtered=({self.filtered_throttle_cmd:.2f},{self.filtered_brake_cmd:.2f})'
        )
        self.debug_pub.publish(debug_msg)

    def emergency_stop(self, reason='timeout'):
        self.pid.reset()
        self.get_logger().error(f'🚨 비상 정지 작동! reason={reason}')
        self.publish_commands(0.0, 1.0, error=0.0)

    def publish_safe_stop(self, reason='safe_stop'):
        self.pid.reset()
        self.publish_commands(0.0, 1.0, error=0.0)
        self.get_logger().debug(f'🛑 안전 정지 유지: reason={reason}')


def main(args=None):
    rclpy.init(args=args)
    node = SpeedControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
