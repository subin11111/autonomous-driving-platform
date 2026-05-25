import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String

try:
    import serial
    from serial import SerialException
except Exception:  # pragma: no cover - handled by runtime checks
    serial = None

    class SerialException(Exception):
        pass


class MockSerial:
    def __init__(self, node_logger):
        self._logger = node_logger
        self._closed = False

    @property
    def in_waiting(self):
        return 0

    def write(self, data: bytes):
        self._logger.info(f'[mock-serial] write: {data!r}')

    def readline(self):
        return b''

    def close(self):
        if not self._closed:
            self._closed = True
            self._logger.info('[mock-serial] closed')


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sanitize_state(value) -> str:
    raw_state = str(value).strip().upper()
    if not raw_state:
        return 'UNKNOWN'

    sanitized = []
    previous_was_underscore = False
    for char in raw_state:
        if char.isalnum():
            sanitized.append(char)
            previous_was_underscore = False
        elif char.isspace() or char in ('-', '_'):
            if not previous_was_underscore:
                sanitized.append('_')
                previous_was_underscore = True

    state = ''.join(sanitized).strip('_')
    return state or 'UNKNOWN'


class McuSerialBridge(Node):
    VALID_INPUT_MODES = {'numeric_direct', 'legacy_cmd_vel'}

    def __init__(self):
        super().__init__('mcu_serial_bridge')

        self.declare_parameter('input_mode', 'numeric_direct')
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('mock_serial', False)
        self.declare_parameter('watchdog_timeout', 0.5)
        self.declare_parameter('serial_timeout', 0.01)
        self.declare_parameter('write_timeout', 0.01)
        self.declare_parameter('command_publish_period_s', 0.05)

        self.declare_parameter('desired_speed_topic', '/desired_speed')
        self.declare_parameter('desired_steering_angle_deg_topic', '/desired_steering_angle_deg')
        self.declare_parameter('behavior_state_topic', '/behavior_state')
        self.declare_parameter('max_abs_speed_mps', 1.40)
        self.declare_parameter('max_abs_steering_deg', 20.0)
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter(
            'stop_states',
            ['STOP', 'EMERGENCY_STOP', 'ESTOP', 'RED_LIGHT', 'OBSTACLE_STOP'],
        )
        self.declare_parameter('speed_input_timeout_s', 0.5)
        self.declare_parameter('steering_input_timeout_s', 0.5)
        self.declare_parameter('state_input_timeout_s', 0.5)

        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('linear_deadband', 0.05)
        self.declare_parameter('angular_deadband', 0.05)
        self.declare_parameter('send_center_command', True)

        self.input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        if self.input_mode not in self.VALID_INPUT_MODES:
            self.get_logger().warn(
                f'unknown input_mode={self.input_mode!r}; falling back to numeric_direct'
            )
            self.input_mode = 'numeric_direct'

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.mock_serial = bool(self.get_parameter('mock_serial').value)
        self.watchdog_timeout = max(0.0, _safe_float(self.get_parameter('watchdog_timeout').value, 0.5))
        self.serial_timeout = max(0.0, _safe_float(self.get_parameter('serial_timeout').value, 0.01))
        self.write_timeout = max(0.0, _safe_float(self.get_parameter('write_timeout').value, 0.01))
        self.command_publish_period_s = max(
            0.001,
            _safe_float(self.get_parameter('command_publish_period_s').value, 0.05),
        )

        self.desired_speed_topic = self.get_parameter(
            'desired_speed_topic'
        ).get_parameter_value().string_value
        self.desired_steering_angle_deg_topic = self.get_parameter(
            'desired_steering_angle_deg_topic'
        ).get_parameter_value().string_value
        self.behavior_state_topic = self.get_parameter(
            'behavior_state_topic'
        ).get_parameter_value().string_value
        self.max_abs_speed_mps = abs(_safe_float(self.get_parameter('max_abs_speed_mps').value, 1.40))
        self.max_abs_steering_deg = abs(
            _safe_float(self.get_parameter('max_abs_steering_deg').value, 20.0)
        )
        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self.stop_states = {
            _sanitize_state(state) for state in self.get_parameter('stop_states').value
        }
        self.speed_input_timeout_s = max(
            0.0,
            _safe_float(self.get_parameter('speed_input_timeout_s').value, 0.5),
        )
        self.steering_input_timeout_s = max(
            0.0,
            _safe_float(self.get_parameter('steering_input_timeout_s').value, 0.5),
        )
        self.state_input_timeout_s = max(
            0.0,
            _safe_float(self.get_parameter('state_input_timeout_s').value, 0.5),
        )

        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.linear_deadband = abs(_safe_float(self.get_parameter('linear_deadband').value, 0.05))
        self.angular_deadband = abs(_safe_float(self.get_parameter('angular_deadband').value, 0.05))
        self.send_center_command = bool(self.get_parameter('send_center_command').value)

        self.status_pub = self.create_publisher(String, '/vehicle/mcu_status', 10)
        self.tx_pub = self.create_publisher(String, '/vehicle/mcu_tx', 10)

        self.serial_conn = self._open_serial()

        self.estop_active = False
        self.read_timer = self.create_timer(0.02, self._read_serial)
        self.estop_sub = self.create_subscription(Bool, '/vehicle/estop', self.estop_callback, 10)

        self._init_numeric_state()
        self._init_legacy_state()
        self._init_mode_interfaces()

        # Start from a safe stopped state for the selected protocol.
        if self.input_mode == 'numeric_direct':
            self._send_numeric_stop(reason='startup')
        else:
            self._send_legacy_stop(reason='startup', force=True)

        self.get_logger().info(
            f'mcu_serial_bridge started: input_mode={self.input_mode}, port={self.port}, '
            f'baudrate={self.baudrate}, mock_serial={self.mock_serial}'
        )

    def _init_numeric_state(self):
        self.last_desired_speed = 0.0
        self.last_steering_deg = 0.0
        self.last_behavior_state = 'STOP'
        self.last_speed_time: Optional[float] = None
        self.last_steering_time: Optional[float] = None
        self.last_state_time: Optional[float] = None
        self.last_numeric_safety_reason: Optional[str] = None
        self.numeric_command_timer = None

    def _init_legacy_state(self):
        self.last_cmd_time: Optional[float] = None
        self.last_sent_drive: Optional[str] = None
        self.last_sent_steer: Optional[str] = None
        self.watchdog_triggered = False
        self.cmd_sub = None
        self.watchdog_timer = None

    def _init_mode_interfaces(self):
        if self.input_mode == 'numeric_direct':
            self.speed_sub = self.create_subscription(
                Float64,
                self.desired_speed_topic,
                self.desired_speed_callback,
                10,
            )
            self.steering_sub = self.create_subscription(
                Float64,
                self.desired_steering_angle_deg_topic,
                self.desired_steering_angle_deg_callback,
                10,
            )
            self.behavior_state_sub = self.create_subscription(
                String,
                self.behavior_state_topic,
                self.behavior_state_callback,
                10,
            )
            self.numeric_command_timer = self.create_timer(
                self.command_publish_period_s,
                self._publish_numeric_command,
            )
            self.get_logger().info(
                'numeric_direct mode: subscribing to '
                f'{self.desired_speed_topic}, {self.desired_steering_angle_deg_topic}, '
                f'{self.behavior_state_topic}'
            )
            return

        self.cmd_sub = self.create_subscription(Twist, self.cmd_topic, self.cmd_callback, 10)
        self.watchdog_timer = self.create_timer(0.05, self._watchdog_check)
        self.get_logger().info(
            f'legacy_cmd_vel mode: subscribing to {self.cmd_topic}; '
            'sending W/S/A/D/C/Space commands'
        )

    def _open_serial(self):
        if self.mock_serial:
            self.get_logger().info('mock_serial=true: using mock serial backend')
            return MockSerial(self.get_logger())

        if serial is None:
            raise RuntimeError('pyserial is not installed. Install python3-serial or set mock_serial=true.')

        try:
            conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.serial_timeout,
                write_timeout=self.write_timeout,
            )
            return conn
        except Exception as exc:
            raise RuntimeError(f'failed to open serial port {self.port}: {exc}') from exc

    def _publish_tx(self, payload: str, reason: str):
        display = payload.rstrip('\n')
        if display == ' ':
            display = 'Space'

        msg = String()
        msg.data = f'{display} ({reason})'
        self.tx_pub.publish(msg)

    def _write_serial(self, payload: str, reason: str):
        try:
            self.serial_conn.write(payload.encode('ascii'))
            self._publish_tx(payload, reason)
        except (SerialException, OSError) as exc:
            self.get_logger().error(f'serial write failed for {payload!r}: {exc}')

    def _send_numeric_command(self, speed_mps: float, steering_deg: float, behavior_state: str):
        payload = f'CMD,{speed_mps:.3f},{steering_deg:.2f},{behavior_state}\n'
        self._write_serial(payload, reason='numeric_direct')

    def _send_numeric_stop(self, reason: str):
        self._write_serial('STOP\n', reason=reason)

    def _send_numeric_estop(self, reason: str):
        self._write_serial('ESTOP\n', reason=reason)

    def _write_legacy_command(self, cmd_char: str, reason: str, force: bool = False):
        if (not force) and cmd_char == ' ' and self.last_sent_drive == ' ':
            return

        self._write_serial(cmd_char, reason=reason)

    def _send_legacy_stop(self, reason: str, force: bool = False):
        self._write_legacy_command(' ', reason=reason, force=force)
        self.last_sent_drive = ' '
        self.last_sent_steer = 'C'

    def desired_speed_callback(self, msg: Float64):
        self.last_desired_speed = _safe_float(msg.data, 0.0)
        self.last_speed_time = time.monotonic()

    def desired_steering_angle_deg_callback(self, msg: Float64):
        self.last_steering_deg = _safe_float(msg.data, 0.0)
        self.last_steering_time = time.monotonic()

    def behavior_state_callback(self, msg: String):
        self.last_behavior_state = _sanitize_state(msg.data)
        self.last_state_time = time.monotonic()

    def _is_timeout(self, last_time: Optional[float], timeout_s: float, now: float) -> bool:
        if last_time is None:
            return True
        return (now - last_time) > timeout_s

    def _numeric_stop_reason(self, now: float) -> Optional[str]:
        if self._is_timeout(self.last_speed_time, self.speed_input_timeout_s, now):
            return 'speed_timeout'
        if self._is_timeout(self.last_steering_time, self.steering_input_timeout_s, now):
            return 'steering_timeout'
        if self._is_timeout(self.last_state_time, self.state_input_timeout_s, now):
            return 'state_timeout'
        if self.last_behavior_state in self.stop_states:
            return 'state_stop'
        return None

    def _log_numeric_safety_reason(self, reason: Optional[str]):
        if reason == self.last_numeric_safety_reason:
            return

        if reason in ('speed_timeout', 'steering_timeout', 'state_timeout'):
            self.get_logger().warn(f'numeric_direct safety stop: {reason}')
        elif reason == 'state_stop':
            self.get_logger().info(f'numeric_direct STOP state: {self.last_behavior_state}')

        self.last_numeric_safety_reason = reason

    def _publish_numeric_command(self):
        if self.estop_active:
            self._send_numeric_estop(reason='estop_hold')
            return

        now = time.monotonic()
        stop_reason = self._numeric_stop_reason(now)
        if stop_reason is not None:
            self._log_numeric_safety_reason(stop_reason)
            self._send_numeric_stop(reason=stop_reason)
            return

        self._log_numeric_safety_reason(None)
        speed_mps = _clamp(
            self.last_desired_speed,
            -self.max_abs_speed_mps,
            self.max_abs_speed_mps,
        )
        if not self.allow_reverse and speed_mps < 0.0:
            speed_mps = 0.0

        steering_deg = _clamp(
            self.last_steering_deg,
            -self.max_abs_steering_deg,
            self.max_abs_steering_deg,
        )
        self._send_numeric_command(speed_mps, steering_deg, self.last_behavior_state)

    def _map_drive(self, linear_x: float) -> str:
        if linear_x > self.linear_deadband:
            return 'W'
        if linear_x < -self.linear_deadband:
            return 'S'
        return ' '

    def _map_steer(self, angular_z: float) -> Optional[str]:
        if angular_z > self.angular_deadband:
            return 'A'
        if angular_z < -self.angular_deadband:
            return 'D'
        if self.send_center_command:
            return 'C'
        return None

    def estop_callback(self, msg: Bool):
        new_state = bool(msg.data)
        if new_state:
            if not self.estop_active:
                self.get_logger().warn('E-stop activated. Sending immediate E-stop command.')
            if self.input_mode == 'numeric_direct':
                self._send_numeric_estop(reason='estop')
            else:
                self._send_legacy_stop(reason='estop', force=True)
        elif self.estop_active:
            self.get_logger().info('E-stop released.')

        self.estop_active = new_state

    def cmd_callback(self, msg: Twist):
        self.last_cmd_time = time.monotonic()
        self.watchdog_triggered = False

        if self.estop_active:
            # In E-stop mode, never allow motion commands to pass through.
            self._send_legacy_stop(reason='estop_hold', force=True)
            return

        drive_cmd = self._map_drive(msg.linear.x)
        steer_cmd = self._map_steer(msg.angular.z)

        if drive_cmd == ' ':
            if self.last_sent_drive != ' ':
                self._send_legacy_stop(reason='cmd_vel_stop', force=False)
            return

        state_changed = (drive_cmd != self.last_sent_drive) or (steer_cmd != self.last_sent_steer)
        if not state_changed:
            return

        # Send drive first, steer second to preserve the previous firmware protocol.
        self._write_legacy_command(drive_cmd, reason='cmd_vel_drive', force=False)
        self.last_sent_drive = drive_cmd

        if steer_cmd is not None:
            self._write_legacy_command(steer_cmd, reason='cmd_vel_steer', force=False)
            self.last_sent_steer = steer_cmd

    def _watchdog_check(self):
        if self.estop_active:
            return

        now = time.monotonic()
        if self.last_cmd_time is None:
            if self.last_sent_drive != ' ':
                self._send_legacy_stop(reason='watchdog_init', force=True)
            return

        elapsed = now - self.last_cmd_time
        if elapsed > self.watchdog_timeout:
            if not self.watchdog_triggered:
                self.get_logger().warn(f'Watchdog timeout ({elapsed:.3f}s). Sending stop command.')
                self._send_legacy_stop(reason='watchdog_timeout', force=True)
                self.watchdog_triggered = True

    def _read_serial(self):
        try:
            waiting = int(getattr(self.serial_conn, 'in_waiting', 0))
        except Exception:
            waiting = 0

        if waiting <= 0:
            return

        # Drain all currently available lines with non-blocking serial timeout.
        while waiting > 0:
            try:
                raw = self.serial_conn.readline()
            except (SerialException, OSError) as exc:
                self.get_logger().error(f'serial read failed: {exc}')
                return

            if not raw:
                return

            line = raw.decode('utf-8', errors='ignore').strip()
            if line:
                msg = String()
                msg.data = line
                self.status_pub.publish(msg)

            try:
                waiting = int(getattr(self.serial_conn, 'in_waiting', 0))
            except Exception:
                waiting = 0

    def shutdown(self):
        if self.input_mode == 'numeric_direct':
            self._send_numeric_stop(reason='shutdown')
        else:
            self._send_legacy_stop(reason='shutdown', force=True)

        try:
            self.serial_conn.close()
        except Exception as exc:
            self.get_logger().warn(f'serial close failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = McuSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.shutdown()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
