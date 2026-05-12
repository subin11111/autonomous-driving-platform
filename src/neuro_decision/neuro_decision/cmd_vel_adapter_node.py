import math
from time import monotonic
from typing import List

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64, String


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class CmdVelAdapterNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_adapter_node')

        self.declare_parameter('desired_speed_topic', '/desired_speed')
        self.declare_parameter('desired_steering_normalized_topic', '/desired_steering_normalized')
        self.declare_parameter('desired_steering_angle_deg_topic', '/desired_steering_angle_deg')
        self.declare_parameter('behavior_state_topic', '/behavior_state')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_period_s', 0.05)
        self.declare_parameter('speed_deadband', 0.01)
        self.declare_parameter('steering_deadband', 0.05)
        self.declare_parameter('linear_cmd_value', 0.2)
        self.declare_parameter('angular_cmd_value', 0.5)
        self.declare_parameter('max_steering_angle_deg', 30.0)
        self.declare_parameter('speed_input_timeout_s', 0.5)
        self.declare_parameter('steering_input_timeout_s', 0.5)
        self.declare_parameter(
            'stop_states',
            ['STOP', 'EMERGENCY_STOP', 'ESTOP', 'RED_LIGHT', 'OBSTACLE_STOP'],
        )

        self.desired_speed_topic = str(self.get_parameter('desired_speed_topic').value)
        self.desired_steering_normalized_topic = str(self.get_parameter('desired_steering_normalized_topic').value)
        self.desired_steering_angle_deg_topic = str(self.get_parameter('desired_steering_angle_deg_topic').value)
        self.behavior_state_topic = str(self.get_parameter('behavior_state_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.publish_period_s = _safe_float(self.get_parameter('publish_period_s').value, 0.05)
        self.speed_deadband = abs(_safe_float(self.get_parameter('speed_deadband').value, 0.01))
        self.steering_deadband = abs(_safe_float(self.get_parameter('steering_deadband').value, 0.05))
        self.linear_cmd_value = abs(_safe_float(self.get_parameter('linear_cmd_value').value, 0.2))
        self.angular_cmd_value = abs(_safe_float(self.get_parameter('angular_cmd_value').value, 0.5))
        self.max_steering_angle_deg = abs(_safe_float(self.get_parameter('max_steering_angle_deg').value, 30.0))
        self.speed_input_timeout_s = max(0.0, _safe_float(self.get_parameter('speed_input_timeout_s').value, 0.5))
        self.steering_input_timeout_s = max(0.0, _safe_float(self.get_parameter('steering_input_timeout_s').value, 0.5))
        self.stop_states = {
            str(state).strip().upper() for state in self.get_parameter('stop_states').value if str(state).strip()
        }

        now = monotonic()
        self.last_speed_time = now
        self.last_steering_norm_time = now
        self.last_steering_deg_time = now
        self.last_state_time = now

        self.last_speed = 0.0
        self.last_steering_norm = 0.0
        self.last_steering_deg = 0.0
        self.last_behavior_state = 'STOP'

        self.speed_timed_out = False
        self.steering_timed_out = False
        self.stop_state_active = True

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.create_subscription(Float64, self.desired_speed_topic, self._on_desired_speed, 10)
        self.create_subscription(Float64, self.desired_steering_normalized_topic, self._on_desired_steering_normalized, 10)
        self.create_subscription(Float64, self.desired_steering_angle_deg_topic, self._on_desired_steering_angle_deg, 10)
        self.create_subscription(String, self.behavior_state_topic, self._on_behavior_state, 10)

        self.timer = self.create_timer(self.publish_period_s, self._publish_cmd_vel)

        self.get_logger().info(
            f'cmd_vel_adapter_node started: speed_topic={self.desired_speed_topic}, '
            f'steer_norm_topic={self.desired_steering_normalized_topic}, '
            f'steer_deg_topic={self.desired_steering_angle_deg_topic}, '
            f'behavior_state_topic={self.behavior_state_topic}, cmd_vel_topic={self.cmd_vel_topic}'
        )

    def _on_desired_speed(self, msg: Float64):
        self.last_speed = _safe_float(msg.data, 0.0)
        self.last_speed_time = monotonic()

    def _on_desired_steering_normalized(self, msg: Float64):
        self.last_steering_norm = _clamp(_safe_float(msg.data, 0.0), -1.0, 1.0)
        self.last_steering_norm_time = monotonic()

    def _on_desired_steering_angle_deg(self, msg: Float64):
        self.last_steering_deg = _safe_float(msg.data, 0.0)
        self.last_steering_deg_time = monotonic()

    def _on_behavior_state(self, msg: String):
        self.last_behavior_state = str(msg.data).strip().upper()
        self.last_state_time = monotonic()

    def _is_stop_state(self) -> bool:
        return self.last_behavior_state in self.stop_states

    def _compute_linear_x(self, now: float) -> float:
        if (now - self.last_speed_time) > self.speed_input_timeout_s:
            if not self.speed_timed_out:
                self.speed_timed_out = True
                self.get_logger().warn('desired_speed timeout. linear.x set to 0.0')
            return 0.0

        if self.speed_timed_out:
            self.speed_timed_out = False
            self.get_logger().info('desired_speed recovered from timeout.')

        if self.last_speed > self.speed_deadband:
            return self.linear_cmd_value
        if self.last_speed < -self.speed_deadband:
            return -self.linear_cmd_value
        return 0.0

    def _compute_steering_normalized(self, now: float) -> float:
        norm_fresh = (now - self.last_steering_norm_time) <= self.steering_input_timeout_s
        if norm_fresh:
            return _clamp(self.last_steering_norm, -1.0, 1.0)

        deg_fresh = (now - self.last_steering_deg_time) <= self.steering_input_timeout_s
        if deg_fresh and self.max_steering_angle_deg > 1e-6:
            return _clamp(self.last_steering_deg / self.max_steering_angle_deg, -1.0, 1.0)

        return 0.0

    def _compute_angular_z(self, now: float) -> float:
        norm_fresh = (now - self.last_steering_norm_time) <= self.steering_input_timeout_s
        deg_fresh = (now - self.last_steering_deg_time) <= self.steering_input_timeout_s

        if (not norm_fresh) and (not deg_fresh):
            if not self.steering_timed_out:
                self.steering_timed_out = True
                self.get_logger().warn('steering input timeout. angular.z set to 0.0')
            return 0.0

        if self.steering_timed_out:
            self.steering_timed_out = False
            self.get_logger().info('steering input recovered from timeout.')

        steering_norm = self._compute_steering_normalized(now)
        if steering_norm > self.steering_deadband:
            return self.angular_cmd_value
        if steering_norm < -self.steering_deadband:
            return -self.angular_cmd_value
        return 0.0

    def _publish_cmd_vel(self):
        now = monotonic()
        stop_active = self._is_stop_state()
        if stop_active != self.stop_state_active:
            self.stop_state_active = stop_active
            if stop_active:
                self.get_logger().warn(f'stop state active: {self.last_behavior_state}. cmd_vel forced to zero.')
            else:
                self.get_logger().info(f'stop state released: {self.last_behavior_state}.')

        msg = Twist()
        if stop_active:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        else:
            msg.linear.x = self._compute_linear_x(now)
            msg.angular.z = self._compute_angular_z(now)

        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelAdapterNode()
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