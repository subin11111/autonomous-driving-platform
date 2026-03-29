import math
import rclpy
from std_msgs.msg import String
from neuro_decision.behavior_node import BehaviorNode, BehaviorState
from neuro_decision.speed_control_node import PIDController


def test_find_lookahead_point_further_than_lookahead():
    rclpy.init()
    node = BehaviorNode()
    points = [(1.0, 0.5), (4.0, 3.0), (6.0, 0.0)]
    x, y = node.find_lookahead_point(points)
    assert math.isclose(x, 6.0, rel_tol=1e-6)
    assert math.isclose(y, 0.0, rel_tol=1e-6)
    node.destroy_node()
    rclpy.shutdown()


def test_find_lookahead_point_uses_last_if_no_far_point():
    rclpy.init()
    node = BehaviorNode()
    points = [(1.0, 0.5), (2.0, 0.5)]
    x, y = node.find_lookahead_point(points)
    assert math.isclose(x, 2.0, rel_tol=1e-6)
    assert math.isclose(y, 0.5, rel_tol=1e-6)
    node.destroy_node()
    rclpy.shutdown()


def test_pid_controller_integral_derivative():
    pid = PIDController(kp=0.5, ki=0.1, kd=0.05)
    out1 = pid.compute(1.0, dt=0.1)
    out2 = pid.compute(2.0, dt=0.1)
    assert out2 > out1


def test_traffic_light_red_forces_stop_state():
    rclpy.init()
    node = BehaviorNode()
    node.traffic_light_callback(String(data='RED'))
    assert node.current_state == BehaviorState.STOP
    node.traffic_light_callback(String(data='GREEN'))
    assert node.current_state in (BehaviorState.LANE_KEEPING, BehaviorState.STOP)
    node.destroy_node()
    rclpy.shutdown()
